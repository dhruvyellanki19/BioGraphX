#!/usr/bin/env python3
"""
Simple Neo4j Graph Query Utility
--------------------------------

Usage:
    python graph_query.py "glaucoma"

This uses the main Question–Disease KG:

    (q:Question)-[:MENTIONS_DISEASE]->(d:Disease)

and finds diseases that frequently co-occur with the seed disease
through shared questions.
"""

import os
import sys
import dotenv
from neo4j import GraphDatabase

dotenv.load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def get_driver():
    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        raise RuntimeError(
            "Missing Neo4j credentials.\n"
            "Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in .env"
        )
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )


QUERY = """
MATCH (d:Disease {name: $name})<-[:MENTIONS_DISEASE]-(q:Question)
MATCH (q)-[:MENTIONS_DISEASE]->(other:Disease)
WHERE other <> d
RETURN other.name AS related, count(DISTINCT q) AS weight
ORDER BY weight DESC
LIMIT $limit
"""


def get_related_diseases(disease, limit=20):
    driver = get_driver()
    with driver.session() as session:
        results = session.run(
            QUERY,
            name=disease,
            limit=limit
        ).data()
    return [(r["related"], r["weight"]) for r in results]


def expand_diseases(seeds, k=10):
    """
    Given ['glaucoma'], expand to top-k related diseases.
    """
    all_related = {}
    for d in seeds:
        neighbors = get_related_diseases(d, limit=k)
        all_related[d] = neighbors
    return all_related


def main():
    if len(sys.argv) < 2:
        print("Usage: python graph_query.py \"disease name\"")
        sys.exit(1)

    disease = sys.argv[1].strip().lower()
    print(f"\nLooking up diseases related to: {disease}\n")

    out = get_related_diseases(disease)
    if not out:
        print("No related diseases found.")
        return

    print("Top related diseases:")
    for d, w in out:
        print(f"  - {d} (weight {w})")


if __name__ == "__main__":
    main()