#!/usr/bin/env python3
"""
Build a Chroma vector index over PubMed QA sentence contexts.

Inputs:
    data/raw/pubmed_qa_pga_labeled.parquet
    data/raw/pubmed_qa_pga_artificial.parquet

Each row has:
    - pubid
    - question
    - context (dict of section -> text or list/array of sentences)

We stream contexts → sentences → embeddings to keep memory reasonable.
"""

import os
import re
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

RAW_FILES = [
    "data/raw/pubmed_qa_pga_labeled.parquet",
    "data/raw/pubmed_qa_pga_artificial.parquet",
]

CHROMA_DIR = "data/vectorstore_pubmed"
COLLECTION_NAME = "pubmed_sentences"

MAX_SENTENCES = int(os.getenv("PUBMED_MAX_SENTENCES", "400000"))
BATCH_SIZE = int(os.getenv("PUBMED_BATCH_SIZE", "256"))


def simple_sent_split(text: str):
    """Lightweight sentence splitter."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def normalize_text(section_text):
    if section_text is None:
        return ""
    if isinstance(section_text, (list, tuple)):
        return " ".join(str(t) for t in section_text if str(t).strip())
    if isinstance(section_text, np.ndarray):
        return " ".join(str(t) for t in section_text.flatten().tolist() if str(t).strip())
    return str(section_text or "")


def sentence_stream():
    """Yield dicts {pubid, sentence} one-by-one from the parquet files."""
    total = 0
    for path in RAW_FILES:
        if not Path(path).exists():
            continue
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            pubid = str(row.get("pubid", "")).strip()
            ctx = row.get("context", {})
            ctx_texts = list(ctx.values()) if isinstance(ctx, dict) else [ctx]
            for section_text in ctx_texts:
                text = normalize_text(section_text).strip()
                if not text:
                    continue
                for sent in simple_sent_split(text):
                    yield {"pubid": pubid, "sentence": sent}
                    total += 1
                    if MAX_SENTENCES and total >= MAX_SENTENCES:
                        return


def reset_collection(client: PersistentClient):
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_index():
    model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")
    client = PersistentClient(path=CHROMA_DIR)
    col = reset_collection(client)

    ids, texts, metas = [], [], []
    total = 0

    def flush():
        nonlocal ids, texts, metas, total
        if not texts:
            return
        emb = model.encode(texts, convert_to_numpy=True)
        col.add(
            ids=ids,
            documents=texts,
            embeddings=emb.tolist(),
            metadatas=metas,
        )
        total += len(texts)
        print(f"Indexed {total} sentences...")
        ids, texts, metas = [], [], []

    for row in sentence_stream():
        sid = f"{row['pubid']}:{uuid.uuid4().hex[:8]}"
        ids.append(sid)
        texts.append(row["sentence"])
        metas.append({"pubid": row["pubid"]})
        if len(texts) >= BATCH_SIZE:
            flush()
    flush()
    print("PubMed sentence index built.")


def main():
    print(f"Building PubMed sentence index (max {MAX_SENTENCES} sentences, batch {BATCH_SIZE})...")
    build_index()


if __name__ == "__main__":
    main()
