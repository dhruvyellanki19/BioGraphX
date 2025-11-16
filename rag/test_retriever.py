from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"

def main():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection("pubmed_index")

    embed_model = SentenceTransformer("all-mpnet-base-v2")

    while True:
        q = input("\nEnter biomedical query (or 'exit'): ")
        if q.lower() == "exit":
            break

        emb = embed_model.encode([q])[0].tolist()

        results = collection.query(query_embeddings=[emb], n_results=5)

        print("\nTop Evidence:")
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            print(f"- PMID {meta['pmid']}: {doc}")

if __name__ == "__main__":
    main()
