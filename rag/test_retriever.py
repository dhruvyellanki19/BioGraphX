import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

def main():
    client = chromadb.PersistentClient(
        path="data/chroma",
        settings=Settings(anonymized_telemetry=False)
    )

    collection = client.get_collection("pubmed_index")
    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    q = input("Enter a biomedical query: ")

    q_emb = embedder.encode([q])[0].tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=5)

    print("\nTop-k Results:")
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"- PMID {meta['pmid']}: {doc}")

if __name__ == "__main__":
    main()
