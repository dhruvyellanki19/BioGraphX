import os
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Your processed file from Sprint 3 ETL:
# data/processed/pubmed_sentences.csv  OR  pubmed_sentences.parquet
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "pubmed_sentences.csv"
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "pubmed_sentences.parquet"

CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "pubmed_index"


# ---------------------------------------------------------------------
# Load Embedding Model (most widely used open-source model)
# ---------------------------------------------------------------------

def get_model():
    """
    Use the most downloaded open-source embedding model.
    Perfect for semantic retrieval. Most stable choice for BioGraphX.
    """
    model_name = "sentence-transformers/all-mpnet-base-v2"
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


# ---------------------------------------------------------------------
# Initialize ChromaDB collection
# ---------------------------------------------------------------------

def get_collection():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


# ---------------------------------------------------------------------
# Load input PubMed sentences
# ---------------------------------------------------------------------

def load_sentences():
    if CSV_PATH.exists():
        print(f"Loading sentences from CSV: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH, dtype=str)
    elif PARQUET_PATH.exists():
        print(f"Loading sentences from Parquet: {PARQUET_PATH}")
        df = pd.read_parquet(PARQUET_PATH)
    else:
        raise FileNotFoundError(
            "Neither pubmed_sentences.csv nor pubmed_sentences.parquet found."
        )

    if "pmid" not in df.columns or "sentence" not in df.columns:
        raise ValueError("File must contain 'pmid' and 'sentence' columns")

    return df


# ---------------------------------------------------------------------
# Embed sentences + push into Chroma
# ---------------------------------------------------------------------

def embed_pubmed(batch_size=500):
    df = load_sentences()
    model = get_model()
    collection = get_collection()

    total = len(df)
    print(f"Embedding {total} PubMed sentences...")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        chunk = df.iloc[start:end]

        sentences = chunk["sentence"].tolist()
        pmids = chunk["pmid"].tolist()

        ids = [f"{pmids[i]}_{start+i}" for i in range(len(chunk))]

        print(f"Encoding batch {start} → {end}")
        embeddings = model.encode(sentences, show_progress_bar=False)

        metadata = [{"pmid": pmids[i]} for i in range(len(chunk))]

        collection.add(
            ids=ids,
            documents=sentences,
            embeddings=embeddings.tolist(),
            metadatas=metadata
        )

    print("\nChroma index creation complete.")
    print("Saved to:", CHROMA_DIR)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    embed_pubmed()
