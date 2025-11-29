#!/usr/bin/env python3
"""
Build a Chroma vector database using embeddings stored in:
data/processed/medquad_with_embeddings.parquet

Creates 3 persistent HNSW-indexed collections:
- medquad_questions
- medquad_answers
- medquad_combined
"""

import os
import shutil
import pandas as pd
import chromadb
from chromadb import PersistentClient

PARQUET_PATH = "data/processed/medquad_with_embeddings.parquet"
CHROMA_DIR = "data/vectorstore"


def reset_vectorstore():
    """Remove old vectorstore to avoid stale or corrupted indexes."""
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print("Old Chroma vectorstore removed.")
    os.makedirs(CHROMA_DIR, exist_ok=True)


def main():
    print("Loading embeddings parquet...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded {len(df)} rows.")

    # Convert numpy arrays → python lists (for Chroma)
    df["question_embedding"] = df["question_embedding"].apply(lambda x: list(x))
    df["answer_embedding"] = df["answer_embedding"].apply(lambda x: list(x))
    df["combined_embedding"] = df["combined_embedding"].apply(lambda x: list(x))

    # Clean metadata - only include columns that actually exist
    meta_cols = []
    for col in ["source", "focus_area"]:
        if col in df.columns:
            meta_cols.append(col)

    if meta_cols:
        metadata = (
            df[meta_cols]
            .fillna("")
            .astype(str)
            .to_dict(orient="records")
        )
    else:
        # Fallback: empty metadata dict per row
        metadata = [{} for _ in range(len(df))]

    # Reset vectorstore (ensures HNSW index is built cleanly)
    reset_vectorstore()

    # Create persistent Chroma client
    chroma_client = PersistentClient(path=CHROMA_DIR)

    print(f"Using persistent Chroma store at: {CHROMA_DIR}")

    # Ensure HNSW is used — very important
    hnsw_cfg = {"hnsw:space": "cosine"}

    # Create collections with ANN indexing enabled
    col_q = chroma_client.create_collection(
        name="medquad_questions",
        metadata=hnsw_cfg,
        embedding_function=None,
    )

    col_a = chroma_client.create_collection(
        name="medquad_answers",
        metadata=hnsw_cfg,
        embedding_function=None,
    )

    col_c = chroma_client.create_collection(
        name="medquad_combined",
        metadata=hnsw_cfg,
        embedding_function=None,
    )

    # IDs must be strings
    if "question_id" not in df.columns:
        raise KeyError("Expected column 'question_id' in parquet for Chroma IDs.")
    ids = df["question_id"].astype(str).tolist()

    def add_batched(col, ids_list, docs, embeds, metas, batch_size=2000, label=""):
        for start in range(0, len(ids_list), batch_size):
            end = start + batch_size
            col.add(
                ids=ids_list[start:end],
                documents=docs[start:end],
                embeddings=embeds[start:end],
                metadatas=metas[start:end],
            )
        if label:
            print(f"Indexed {label}.")

    print("Indexing questions...")
    add_batched(
        col_q,
        ids,
        df["question"].astype(str).tolist(),
        df["question_embedding"].tolist(),
        metadata,
        label="questions",
    )

    print("Indexing answers...")
    add_batched(
        col_a,
        ids,
        df["answer"].astype(str).tolist(),
        df["answer_embedding"].tolist(),
        metadata,
        label="answers",
    )

    print("Indexing combined text...")
    add_batched(
        col_c,
        ids,
        df["combined_text"].astype(str).tolist(),
        df["combined_embedding"].tolist(),
        metadata,
        label="combined text",
    )

    # PersistentClient writes immediately; no explicit persist() method needed
    print("Chroma index built successfully.")


if __name__ == "__main__":
    main()
