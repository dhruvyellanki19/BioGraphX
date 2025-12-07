import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from pathlib import Path

DATA_PATH = Path("data/processed/pubmed_sentences.parquet")
CHROMA_DIR = Path("data/chroma")


def main():
    print("\n[Chroma] Loading PubMed sentences...")
    df = pd.read_parquet(DATA_PATH)

    print(f"[Chroma] Loaded {len(df)} sentences.")

    print("[Chroma] Loading embedding model (all-mpnet-base-v2)...")
    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    print("[Chroma] Initializing client...")
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )

    try:
        client.delete_collection("pubmed_index")
    except:
        pass

    collection = client.create_collection(
        name="pubmed_index",
        metadata={"hnsw:space": "cosine"}
    )

    print("[Chroma] Embedding and inserting...")

    batch_size = 512
    ids, texts, metadatas = [], [], []

    for idx, row in df.iterrows():
        ids.append(f"id_{idx}")
        texts.append(row["sentence"])
        metadatas.append({"pmid": str(row["pmid"])})

        if len(ids) == batch_size:
            embs = embedder.encode(texts).tolist()
            collection.add(
                ids=ids, embeddings=embs,
                documents=texts, metadatas=metadatas
            )
            ids, texts, metadatas = [], [], []

    if ids:
        embs = embedder.encode(texts).tolist()
        collection.add(
            ids=ids, embeddings=embs,
            documents=texts, metadatas=metadatas
        )

    print("\n[Chroma] DONE building index.")
    print(f"Stored at: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
