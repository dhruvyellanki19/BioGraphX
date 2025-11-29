#!/usr/bin/env python3
"""
Prepare Neo4j-compatible CSV node and edge files from the normalized MedQuAD dataset.

Expected input: data/processed/medquad_with_entities_normalized.csv

Columns expected ideally:
- question_id
- question
- answer
- source        (optional, will be created as empty if missing)
- focus_area    (optional, will be created as empty if missing)
- diseases
- chemicals

Output CSVs (for Cypher-based import):
- data/neo4j/nodes_questions.csv       (question_id, question, answer, source, focus_area)
- data/neo4j/nodes_diseases.csv        (disease)
- data/neo4j/nodes_chemicals.csv       (chemical)
- data/neo4j/rels_question_disease.csv (question_id, disease)
- data/neo4j/rels_question_chemical.csv(question_id, chemical)
"""

import os
import ast
import pandas as pd

INPUT = "data/processed/medquad_with_entities_normalized.csv"
OUT_DIR = "data/neo4j"


def _parse_list_cell(cell):
    """
    Cells are stored as strings like "['glaucoma', 'blindness']".
    Convert them back to Python lists safely.
    """
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        value = ast.literal_eval(cell)
        if isinstance(value, list):
            return value
        return []
    except Exception:
        return []


def load_medquad():
    if not os.path.exists(INPUT):
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    print(f"Loading: {INPUT}")
    df = pd.read_csv(INPUT)

    # Ensure optional columns exist
    for col in ["source", "focus_area"]:
        if col not in df.columns:
            print(f"Column '{col}' not found. Creating empty column.")
            df[col] = ""

    required_cols = [
        "question_id",
        "question",
        "answer",
        "source",
        "focus_area",
        "diseases",
        "chemicals",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in input CSV: {missing}")

    # Parse diseases and chemicals back to lists
    df["diseases"] = df["diseases"].apply(_parse_list_cell)
    df["chemicals"] = df["chemicals"].apply(_parse_list_cell)

    return df


def save_nodes(df: pd.DataFrame):
    os.makedirs(OUT_DIR, exist_ok=True)

    # Question nodes
    q_cols = ["question_id", "question", "answer", "source", "focus_area"]
    df_nodes = df[q_cols].copy()
    df_nodes.to_csv(f"{OUT_DIR}/nodes_questions.csv", index=False)
    print("Saved Question nodes → data/neo4j/nodes_questions.csv")

    # Disease nodes
    diseases = sorted({d for sub in df["diseases"] for d in sub})
    df_diseases = pd.DataFrame({"disease": diseases})
    df_diseases.to_csv(f"{OUT_DIR}/nodes_diseases.csv", index=False)
    print("Saved Disease nodes → data/neo4j/nodes_diseases.csv")

    # Chemical nodes
    chemicals = sorted({c for sub in df["chemicals"] for c in sub})
    df_chems = pd.DataFrame({"chemical": chemicals})
    df_chems.to_csv(f"{OUT_DIR}/nodes_chemicals.csv", index=False)
    print("Saved Chemical nodes → data/neo4j/nodes_chemicals.csv")


def save_relationships(df: pd.DataFrame):
    # Question–Disease relationships
    rel_q_d = []
    for _, row in df.iterrows():
        qid = row["question_id"]
        for d in row["diseases"]:
            rel_q_d.append({"question_id": qid, "disease": d})

    df_qd = pd.DataFrame(rel_q_d, columns=["question_id", "disease"])
    df_qd.to_csv(f"{OUT_DIR}/rels_question_disease.csv", index=False)
    print("Saved Disease relationships → data/neo4j/rels_question_disease.csv")

    # Question–Chemical relationships
    rel_q_c = []
    for _, row in df.iterrows():
        qid = row["question_id"]
        for c in row["chemicals"]:
            rel_q_c.append({"question_id": qid, "chemical": c})

    df_qc = pd.DataFrame(rel_q_c, columns=["question_id", "chemical"])
    df_qc.to_csv(f"{OUT_DIR}/rels_question_chemical.csv", index=False)
    print("Saved Chemical relationships → data/neo4j/rels_question_chemical.csv")


def main():
    df = load_medquad()
    save_nodes(df)
    save_relationships(df)
    print("\nNeo4j CSV generation completed.\n")


if __name__ == "__main__":
    main()