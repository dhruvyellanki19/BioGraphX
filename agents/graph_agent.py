#!/usr/bin/env python3
# agents/graph_agent.py

import os
import dotenv
from neo4j import GraphDatabase

dotenv.load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

_DRIVER = None

GRAPH_QUERY = """
MATCH (d:Disease {name: $name})<-[:MENTIONS_DISEASE]-(q:Question)
MATCH (q)-[:MENTIONS_DISEASE]->(other:Disease)
WHERE other <> d
RETURN other.name AS related, count(DISTINCT q) AS weight
ORDER BY weight DESC
LIMIT $limit
"""


def _get_driver():
    global _DRIVER
    if _DRIVER is None:
        if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
            raise ValueError("Missing Neo4j credentials in .env")
        print("Opening Neo4j driver...")
        _DRIVER = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _DRIVER


def graph_agent(state: dict) -> dict:
    """
    Given extracted diseases, use Neo4j to find graph-related diseases.
    Uses the first disease as the seed (as in your earlier script).
    """
    diseases = state.get("diseases") or []
    if not diseases:
        print("[GRAPH_AGENT] No diseases from NER; skipping Neo4j lookup.")
        return {"graph_results": []}

    seed = diseases[0]
    driver = _get_driver()

    print(f"[GRAPH_AGENT] Searching Neo4j for diseases related to: {seed}")
    with driver.session() as session:
        results = session.run(
            GRAPH_QUERY,
            name=seed,
            limit=20,
        ).data()

    graph_results = [(r["related"], r["weight"]) for r in results]
    if not graph_results:
        print("[GRAPH_AGENT] No related diseases found in Neo4j.")
        return {"graph_results": []}

    print("\n[GRAPH_AGENT] Top graph-related diseases:")
    for d, w in graph_results[:15]:
        print(f"  - {d} (weight {w})")
    print()

    return {"graph_results": graph_results}