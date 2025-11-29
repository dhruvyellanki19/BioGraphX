#!/usr/bin/env python3
"""
Generate embeddings for MedQuAD questions/answers using a biomedical
Sentence-Transformer model.

Input:
    data/processed/medquad_with_entities_normalized.csv

Output:
    data/processed/medquad_with_embeddings.parquet

All original columns (including diseases/chemicals) are preserved.
"""

import os
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT = "data/processed/medquad_with_entities_normalized.csv"
OUTPUT = "data/processed/medquad_with_embeddings.parquet"


def load_model():
    print("Loading biomedical embedding model (NeuML/pubmedbert-base-embeddings)...")
    model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")
    print("Model loaded.")
    return model


def load_data():
    print(f"Loading dataset: {INPUT}")
    df = pd.read_csv(INPUT)
    print(f"Loaded {len(df)} rows.")
    return df


def generate_text(df):
    """
    Create a single combined_text field for embeddings.
    """
    print("Creating combined text column...")
    df["question"] = df["question"].astype(str)
    df["answer"] = df["answer"].astype(str)
    df["combined_text"] = df["question"] + " " + df["answer"]
    return df


def embed(df, model):
    """
    Generate embeddings for question, answer, and combined text.
    """
    print("Generating embeddings...")

    questions = df["question"].tolist()
    answers = df["answer"].tolist()
    combined = df["combined_text"].tolist()

    print("Embedding questions...")
    q_emb = model.encode(questions, batch_size=16, show_progress_bar=True)

    print("Embedding answers...")
    a_emb = model.encode(answers, batch_size=16, show_progress_bar=True)

    print("Embedding combined text...")
    c_emb = model.encode(combined, batch_size=16, show_progress_bar=True)

    df["question_embedding"] = q_emb.tolist()
    df["answer_embedding"] = a_emb.tolist()
    df["combined_embedding"] = c_emb.tolist()

    return df


def save(df):
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    df.to_parquet(OUTPUT, index=False)
    print(f"Saved embeddings → {OUTPUT}")


def main():
    model = load_model()
    df = load_data()
    df = generate_text(df)
    df = embed(df, model)
    save(df)
    print("Embedding generation completed.")


if __name__ == "__main__":
    main()