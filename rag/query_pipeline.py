import os
import subprocess
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from py2neo import Graph
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# -----------------------------
# Load Environment
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"

# -----------------------------
# Connect to Neo4j
# -----------------------------
graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# -----------------------------
# Load Embedding Model
# -----------------------------
embed_model = SentenceTransformer("all-mpnet-base-v2")

# -----------------------------
# ChromaDB Client
# -----------------------------
client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_collection("pubmed_index")


# ================================================================
# 1. GET QUESTION FROM NEO4J
# ================================================================
def get_question_from_graph(qid: int):
    res = graph.run(
        """
        MATCH (q:Question {id: $id})
        RETURN q.text AS text
        """,
        id=qid
    ).data()

    if not res:
        raise ValueError(f"No Question with id={qid} found in Neo4j")

    return res[0]["text"]


# ================================================================
# 2. GET ENTITIES FROM NEO4J
# ================================================================
def get_entities(qid: int):
    res = graph.run(
        """
        MATCH (q:Question {id: $qid})-[:ABOUT]->(e)
        RETURN e.name AS name, labels(e)[0] AS type
        """,
        qid=qid
    ).data()

    return [{"name": r["name"], "type": r["type"]} for r in res]


# ================================================================
# 3. RETRIEVE EVIDENCE
# ================================================================
def retrieve_evidence(query: str, k: int = 5):
    q_emb = embed_model.encode([query])[0].tolist()

    result = collection.query(
        query_embeddings=[q_emb],
        n_results=k
    )

    sentences = result["documents"][0]
    metadata = result["metadatas"][0]

    evidence = []
    for s, m in zip(sentences, metadata):
        evidence.append({"sentence": s, "pmid": m["pmid"]})

    return evidence


# ================================================================
# 4. RUN OLLAMA LLM
# ================================================================
def ollama_llm(prompt: str) -> str:
    p = subprocess.Popen(
        ["ollama", "run", "llama3.1:8b"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, err = p.communicate(prompt)

    if err:
        print("OLLAMA ERROR:", err)

    return out.strip()


# ================================================================
# 5. BUILD FINAL ANSWER
# ================================================================
def build_prompt(question: str, entities, evidence):
    ent_str = ", ".join([e["name"] for e in entities]) if entities else "None"

    ev_str = "\n".join(
        [f"PMID {ev['pmid']}: {ev['sentence']}" for ev in evidence]
    )

    prompt = f"""
You are a biomedical reasoning LLM.

Question:
{question}

Graph Entities:
{ent_str}

Evidence Sentences:
{ev_str}

Using ONLY the evidence above, produce a medically accurate answer.
Include PMIDs where relevant.
Keep the answer under 150 words.
"""
    return prompt.strip()


# ================================================================
# MAIN PIPELINE
# ================================================================
def main():
    qid = int(input("Enter Question ID: "))

    print("\n[1] Fetching question from Neo4j...")
    question = get_question_from_graph(qid)
    print("QUESTION:", question)

    print("\n[2] Fetching entities from Neo4j...")
    entities = get_entities(qid)
    print("ENTITIES:", entities)

    print("\n[3] Retrieving evidence from ChromaDB...")
    retrieval_query = question + " " + " ".join([e["name"] for e in entities])
    evidence = retrieve_evidence(retrieval_query)
    print("EVIDENCE:")
    for ev in evidence:
        print(f"- PMID {ev['pmid']}: {ev['sentence']}")

    print("\n[4] Running Ollama model (llama3.1:8b)")
    prompt = build_prompt(question, entities, evidence)
    answer = ollama_llm(prompt)

    print("\n====================")
    print("FINAL ANSWER:")
    print("====================")
    print(answer)


if __name__ == "__main__":
    main()
