#!/usr/bin/env python3
# agents/vector_agent.py

import os
import ast
import pandas as pd
import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

PARQUET_PATH = "data/processed/medquad_with_embeddings.parquet"
CHROMA_DIR = "data/vectorstore"
COLLECTION_NAME = "medquad_combined"
PUBMED_CHROMA_DIR = "data/vectorstore_pubmed"
PUBMED_COLLECTION_NAME = "pubmed_sentences"
TOP_K = 10

_DF = None
_COLLECTION = None
_PUBMED_COLLECTION = None
_EMBEDDER = None


def _ensure_list(value):
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        if "," in value:
            return [v.strip() for v in value.split(",")]
        return [value]
    return [str(value)]


def _load_df():
    global _DF
    if _DF is None:
        if not os.path.exists(PARQUET_PATH):
            raise FileNotFoundError(f"Missing parquet: {PARQUET_PATH}")
        df = pd.read_parquet(PARQUET_PATH)
        df["question_id"] = df["question_id"].astype(str)
        # Keep raw strings for evidence snippets
        for col in ["question", "answer", "combined_text"]:
            if col in df.columns:
                df[col] = df[col].astype(str)
        if "diseases" in df.columns:
            df["diseases"] = df["diseases"].apply(_ensure_list)
        else:
            df["diseases"] = [[] for _ in range(len(df))]
        _DF = df
    return _DF



def _load_collection():
    global _COLLECTION
    if _COLLECTION is None:
        # Create a persistent Chroma client at your directory
        client = PersistentClient(path=CHROMA_DIR)

        # Load existing collection or create if missing
        _COLLECTION = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    return _COLLECTION


def _load_pubmed_collection():
    global _PUBMED_COLLECTION
    if _PUBMED_COLLECTION is None:
        client = PersistentClient(path=PUBMED_CHROMA_DIR)
        _PUBMED_COLLECTION = client.get_or_create_collection(
            name=PUBMED_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _PUBMED_COLLECTION


def _load_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        print("Loading PubMedBERT sentence embedder...")
        _EMBEDDER = SentenceTransformer("NeuML/pubmedbert-base-embeddings")
    return _EMBEDDER


def vector_agent(state: dict) -> dict:
    """
    Semantic search over MedQuAD using PubMedBERT + Chroma.
    Always falls back to the user query.
    """
    # FIX: Always use user query if vector_query empty
    raw_query = state.get("vector_query") or state.get("query") or ""
    query = raw_query.strip()

    if not query:
        print("[VECTOR_AGENT] Empty query; skipping vector search.")
        return {"vector_results": []}

    df = _load_df()
    collection = _load_collection()
    pubmed_collection = _load_pubmed_collection()
    embedder = _load_embedder()

    print("[VECTOR_AGENT] Running semantic search...")
    emb = embedder.encode(query, convert_to_numpy=True)

    results = collection.query(
        query_embeddings=[emb.tolist()],
        n_results=TOP_K,
    )

    ids = results["ids"][0]
    docs = results["documents"][0]
    distances = results["distances"][0]

    vector_results = []
    for qid, doc, dist in zip(ids, docs, distances):
        row = df[df["question_id"] == str(qid)]
        diseases = row["diseases"].iloc[0] if not row.empty else []
        question = row["question"].iloc[0] if not row.empty and "question" in row else ""
        answer = row["answer"].iloc[0] if not row.empty and "answer" in row else ""
        combined = row["combined_text"].iloc[0] if not row.empty and "combined_text" in row else doc

        vector_results.append(
            {
                "qid": str(qid),
                "text": doc,
                "score": float(dist),
                "diseases": sorted(set(diseases)),
                "question": question,
                "answer": answer,
                "combined": combined,
                "source": "medquad",
            }
        )

    # Re-rank to prefer snippets that mention detected diseases
    diseases_in_query = {d.lower() for d in (state.get("diseases") or []) if d}
    if diseases_in_query:
        def _score(hit):
            text = (hit.get("combined") or hit.get("question") or "").lower()
            hits = sum(1 for d in diseases_in_query if d in text)
            return hits
        vector_results = sorted(
            vector_results,
            key=lambda h: (_score(h), -h["score"]),
            reverse=True,
        )
    # Keep only top-K after re-rank to avoid prompt bloat
    vector_results = vector_results[:TOP_K]

    # PubMed sentences retrieval
    pubmed_results = []
    pubmed_query = pubmed_collection.query(
        query_embeddings=[emb.tolist()],
        n_results=TOP_K,
    )
    if pubmed_query and pubmed_query.get("ids"):
        metas = pubmed_query.get("metadatas") or [[]]
        for pid, sent, dist, meta in zip(
            pubmed_query["ids"][0],
            pubmed_query["documents"][0],
            pubmed_query["distances"][0],
            metas[0],
        ):
            pubid = meta.get("pubid") if isinstance(meta, dict) else ""
            pubmed_results.append(
                {
                    "qid": "",
                    "id": str(pid),
                    "pubid": str(pubid),
                    "text": sent,
                    "score": float(dist),
                    "diseases": [],
                    "question": "",
                    "answer": "",
                    "combined": sent,
                    "source": "pubmed",
                }
            )

    print("\n[VECTOR_AGENT] Vector search top MedQuAD matches:")
    for r in vector_results:
        print(f"  Q{r['qid']}  score={r['score']:.4f}  diseases={r['diseases']}")
    print("\n[VECTOR_AGENT] PubMed sentence matches:")
    for r in pubmed_results[:5]:
        print(f"  PMID {r.get('pubid','')}  score={r['score']:.4f}")
    print()

    if not vector_results and not pubmed_results:
        print("[VECTOR_AGENT] No vector matches returned.")

    combined_results = vector_results + pubmed_results

    return {
        "vector_results": combined_results,
        "vector_results_medquad": vector_results,
        "vector_results_pubmed": pubmed_results,
    }
