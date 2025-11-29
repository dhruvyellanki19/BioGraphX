#!/usr/bin/env python3
"""
Normalize MedQuAD entities and split them into explicit columns.

Input:
  data/processed/medquad_with_entities.csv

  Expected columns:
    - question_id
    - question
    - answer
    - source
    - focus_area
    - entities  (stringified dict like:
                 {"diseases": ["glaucoma"], "chemicals": ["aspirin"]})

Output:
  data/processed/medquad_with_entities_normalized.csv

  Columns:
    - question_id
    - question
    - answer
    - source
    - focus_area
    - diseases   (list[str], normalized, deduplicated)
    - chemicals  (list[str], normalized, deduplicated)
"""

import os
import ast
import string
import pandas as pd

INPUT_PATH = "data/processed/medquad_with_entities.csv"
OUTPUT_PATH = "data/processed/medquad_with_entities_normalized.csv"


def normalize_entity(text: str) -> str:
    """Lowercase, strip punctuation and extra spaces from an entity string."""
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = text.strip(string.punctuation + " ")
    text = " ".join(text.split())
    return text


def parse_entities_cell(cell):
    """
    Parse the 'entities' cell back into a Python dict.

    The cell should look like:
      {"diseases": ["asthma"], "chemicals": ["aspirin"]}

    Returns a dict with keys 'diseases' and 'chemicals'
    (both lists of strings), even if parsing fails.
    """
    if not isinstance(cell, str) or not cell.strip():
        return {"diseases": [], "chemicals": []}

    try:
        data = ast.literal_eval(cell)
        if isinstance(data, dict):
            diseases = data.get("diseases", [])
            chemicals = data.get("chemicals", [])
            if not isinstance(diseases, list):
                diseases = []
            if not isinstance(chemicals, list):
                chemicals = []
            return {"diseases": diseases, "chemicals": chemicals}
    except Exception:
        pass

    return {"diseases": [], "chemicals": []}


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    print(f"Loading: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows.")
    print(f"Columns: {list(df.columns)}")

    if "entities" not in df.columns:
        raise ValueError("Expected column 'entities' not found in input CSV.")

    normalized_diseases = []
    normalized_chemicals = []

    print("Parsing and normalizing entities...")

    for _, row in df.iterrows():
        ent_dict = parse_entities_cell(row["entities"])

        # Normalize and deduplicate diseases
        seen_d = set()
        d_list = []
        for item in ent_dict.get("diseases", []):
            n = normalize_entity(item)
            if n and n not in seen_d:
                seen_d.add(n)
                d_list.append(n)

        # Normalize and deduplicate chemicals
        seen_c = set()
        c_list = []
        for item in ent_dict.get("chemicals", []):
            n = normalize_entity(item)
            if n and n not in seen_c:
                seen_c.add(n)
                c_list.append(n)

        normalized_diseases.append(d_list)
        normalized_chemicals.append(c_list)

    # Attach new columns
    df["diseases"] = normalized_diseases
    df["chemicals"] = normalized_chemicals

    # Drop the raw 'entities' column now that we have explicit lists
    df = df.drop(columns=["entities"])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved normalized dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()