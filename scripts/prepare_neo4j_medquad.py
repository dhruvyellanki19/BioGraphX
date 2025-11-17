#!/usr/bin/env python3
"""
Prepare Neo4j ingestion CSV files:
- disease nodes
- chemical nodes
- question nodes
- relationships:
    QUESTION_HAS_DISEASE
    QUESTION_HAS_CHEMICAL
"""

import os
import ast
import pandas as pd


INPUT_PATH = "data/processed/medquad_with_entities_normalized.csv"
OUT_DIR = "data/neo4j"


def parse_entities(cell):
    """Convert string dict to Python dict."""
    try:
        if isinstance(cell, str):
            return ast.literal_eval(cell)
    except Exception:
        pass

    return {"diseases": [], "chemicals": []}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    ensure_dir(OUT_DIR)

    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows.")

    disease_set = set()
    chemical_set = set()

    print("Extracting global entity sets...")
    for _, row in df.iterrows():
        ent = parse_entities(row["entities"])
        disease_set.update(ent.get("diseases", []))
        chemical_set.update(ent.get("chemicals", []))

    disease_list = sorted(list(disease_set))
    chemical_list = sorted(list(chemical_set))

    print("Assigning IDs...")
    disease_map = {name: f"D{idx+1:05d}" for idx, name in enumerate(disease_list)}
    chemical_map = {name: f"C{idx+1:05d}" for idx, name in enumerate(chemical_list)}

    print("Saving disease nodes...")
    pd.DataFrame({
        "disease_id:ID(Disease)": disease_map.values(),
        "name": disease_map.keys()
    }).to_csv(f"{OUT_DIR}/nodes_diseases.csv", index=False)

    print("Saving chemical nodes...")
    pd.DataFrame({
        "chemical_id:ID(Chemical)": chemical_map.values(),
        "name": chemical_map.keys()
    }).to_csv(f"{OUT_DIR}/nodes_chemicals.csv", index=False)

    print("Saving question nodes...")
    df_questions = pd.DataFrame({
        "question_id:ID(Question)": [f"Q{idx+1:05d}" for idx in range(len(df))],
        "text": df["question"]
    })
    df_questions.to_csv(f"{OUT_DIR}/nodes_questions.csv", index=False)

    print("Saving relationships...")
    rel_q_disease = []
    rel_q_chemical = []

    for idx, row in df.iterrows():
        qid = f"Q{idx+1:05d}"
        ent = parse_entities(row["entities"])

        for d in ent["diseases"]:
            rel_q_disease.append([qid, disease_map[d], "QUESTION_HAS_DISEASE"])

        for c in ent["chemicals"]:
            rel_q_chemical.append([qid, chemical_map[c], "QUESTION_HAS_CHEMICAL"])

    pd.DataFrame(rel_q_disease,
                 columns=[":START_ID(Question)", ":END_ID(Disease)", ":TYPE"])\
        .to_csv(f"{OUT_DIR}/rels_question_disease.csv", index=False)

    pd.DataFrame(rel_q_chemical,
                 columns=[":START_ID(Question)", ":END_ID(Chemical)", ":TYPE"])\
        .to_csv(f"{OUT_DIR}/rels_question_chemical.csv", index=False)

    print("Complete.")
    print(f"CSV files created in: {OUT_DIR}")


if __name__ == "__main__":
    main()
