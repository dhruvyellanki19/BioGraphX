import os
import pandas as pd
from py2neo import Graph
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

BASE = "data/processed/graph_data/"


def load_questions():
    df = pd.read_csv(BASE + "nodes_question.csv")
    df = df.dropna(subset=["id", "text"])
    df["id"] = df["id"].astype(int)
    df["text"] = df["text"].astype(str).str.strip()

    for _, row in df.iterrows():
        graph.run(
            """
            MERGE (q:Question {id: $id})
            SET q.text = $text
            """,
            id=row["id"],
            text=row["text"],
        )
    print(f"Loaded {len(df)} Question nodes.")


def load_diseases():
    df = pd.read_csv(BASE + "nodes_disease.csv")
    df = df.dropna(subset=["name"])
    df["name"] = df["name"].astype(str).str.strip()

    for _, row in df.iterrows():
        graph.run(
            "MERGE (d:Disease {name: $name})",
            name=row["name"],
        )
    print(f"Loaded {len(df)} Disease nodes.")


def load_drugs():
    df = pd.read_csv(BASE + "nodes_drug.csv")
    df = df.dropna(subset=["name"])
    df["name"] = df["name"].astype(str).str.strip()

    for _, row in df.iterrows():
        graph.run(
            "MERGE (d:Drug {name: $name})",
            name=row["name"],
        )
    print(f"Loaded {len(df)} Drug nodes.")


def load_relationships():
    df = pd.read_csv(BASE + "rels_about.csv")
    print("rels_about.csv columns:", df.columns.tolist())

    # Normalize relationship CSV
    df = df.rename(columns={
        "start_id": "question_id",
        "end_name": "entity_name"
    })

    df = df.dropna(subset=["question_id", "entity_name"])
    df["question_id"] = df["question_id"].astype(int)
    df["entity_name"] = df["entity_name"].astype(str).str.strip()

    # Load drug names separately
    drug_df = pd.read_csv(BASE + "nodes_drug.csv")
    drug_df["name"] = drug_df["name"].astype(str).str.strip()
    drug_names = set(drug_df["name"].unique())

    rel_count = 0

    for _, row in df.iterrows():
        qid = row["question_id"]
        name = row["entity_name"]

        if name in drug_names:
            graph.run(
                """
                MATCH (q:Question {id: $qid})
                MATCH (d:Drug {name: $name})
                MERGE (q)-[:ABOUT]->(d)
                """,
                qid=qid, name=name
            )
        else:
            graph.run(
                """
                MATCH (q:Question {id: $qid})
                MATCH (d:Disease {name: $name})
                MERGE (q)-[:ABOUT]->(d)
                """,
                qid=qid, name=name
            )

        rel_count += 1

    print(f"Loaded {rel_count} ABOUT relationships.")


def main():
    print("Loading BioGraphX Knowledge Graph into Neo4j")
    load_questions()
    load_diseases()
    load_drugs()
    load_relationships()
    print("Graph import complete!")


if __name__ == "__main__":
    main()
