#!/usr/bin/env python3
"""
Generate global unique disease and chemical lists from the normalized
MedQuAD dataset. Produces:

data/processed/unique_diseases.txt
data/processed/unique_chemicals.txt
"""

import os
import ast
import pandas as pd

INPUT_PATH = "data/processed/medquad_with_entities_normalized.csv"
OUT_DISEASES = "data/processed/unique_diseases.txt"
OUT_CHEMICALS = "data/processed/unique_chemicals.txt"


def parse_entities(cell):
    """Parse the JSON-like dict string into a Python dictionary."""
    if not isinstance(cell, str):
        return {"diseases": [], "chemicals": []}

    try:
        data = ast.literal_eval(cell)
        if isinstance(data, dict):
            return {
                "diseases": data.get("diseases", []),
                "chemicals": data.get("chemicals", [])
            }
    except Exception:
        pass

    return {"diseases": [], "chemicals": []}


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    print(f"Loading: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows.")

    diseases_set = set()
    chemicals_set = set()

    print("Extracting unique entities...")

    for idx, row in df.iterrows():
        ent_dict = parse_entities(row["entities"])
        diseases_set.update(ent_dict.get("diseases", []))
        chemicals_set.update(ent_dict.get("chemicals", []))

    # Sort alphabetically
    diseases_list = sorted(diseases_set)
    chemicals_list = sorted(chemicals_set)

    os.makedirs("data/processed", exist_ok=True)

    print("Saving unique disease list...")
    with open(OUT_DISEASES, "w") as f:
        for d in diseases_list:
            f.write(d + "\n")

    print("Saving unique chemical list...")
    with open(OUT_CHEMICALS, "w") as f:
        for c in chemicals_list:
            f.write(c + "\n")

    print("Finished.")
    print(f"Diseases saved to: {OUT_DISEASES}")
    print(f"Chemicals saved to: {OUT_CHEMICALS}")


if __name__ == "__main__":
    main()
