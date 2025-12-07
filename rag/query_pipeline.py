from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


class QueryPipeline:
    """
    Simple Sprint 3 retrieval pipeline:
    - Loads all-mpnet-base-v2 embedding model
    - Connects to ChromaDB pubmed_index
    - Executes top-k retrieval for any query
    """

    def __init__(self):
        # Same embedding model used to build index
        self.embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

        # Connect to persistent Chroma index
        self.client = chromadb.PersistentClient(
            path="data/chroma",
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_collection("pubmed_index")

    def search(self, query: str, k: int = 5):
        """
        Retrieve top-k evidence sentences for a query.
        """
        # Generate embedding
        emb = self.embedder.encode([query])[0].tolist()

        # Query Chroma vector DB
        hits = self.collection.query(
            query_embeddings=[emb],
            n_results=k
        )

        # Format results
        out = []
        for sentence, meta in zip(hits["documents"][0], hits["metadatas"][0]):
            out.append({
                "pmid": meta["pmid"],
                "sentence": sentence
            })

        return out


# ----------------------------------------------------------------------
# Manual test entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    qp = QueryPipeline()

    q = input("Enter biomedical query: ")

    results = qp.search(q, k=5)

    print("\nTop-k Evidence:\n")
    for r in results:
        print(f"[PMID {r['pmid']}] {r['sentence']}")
