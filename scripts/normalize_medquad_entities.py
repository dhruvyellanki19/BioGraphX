import os
import ast
import json
import string
import pandas as pd

INPUT_PATH = "data/processed/medquad_with_entities.csv"
OUTPUT_PATH = "data/processed/medquad_with_entities_normalized.csv"


def normalize_entity(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = text.strip(string.punctuation + " ")
    text = " ".join(text.split())
    return text


def parse_entities(cell):
    """
    The cell contains a string like:
    {"diseases": ["asthma"], "chemicals": ["aspirin"]}
    This converts it back into a Python dict.
    """
    if not isinstance(cell, str):
        return {"diseases": [], "chemicals": []}

    try:
        data = ast.literal_eval(cell)
        if isinstance(data, dict):
            return {
                "diseases": data.get("diseases", []),
                "chemicals": data.get("chemicals", [])
            }
        return {"diseases": [], "chemicals": []}
    except Exception:
        return {"diseases": [], "chemicals": []}


def normalize_entity_dict(ent_dict):
    """
    Normalize disease and chemical lists inside the dictionary.
    """
    normalized = {
        "diseases": [],
        "chemicals": []
    }

    for key in ["diseases", "chemicals"]:
        seen = set()
        for item in ent_dict.get(key, []):
            n = normalize_entity(item)
            if n and n not in seen:
                seen.add(n)
                normalized[key].append(n)

    return normalized


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    print(f"Loading: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows.")

    if "entities" not in df.columns:
        raise ValueError("Column 'entities' not found in CSV.")

    print("Normalizing entity dictionaries...")
    normalized_list = []

    for idx, row in df.iterrows():
        ent_dict = parse_entities(row["entities"])
        norm_dict = normalize_entity_dict(ent_dict)
        normalized_list.append(json.dumps(norm_dict, ensure_ascii=False))

    df["entities"] = normalized_list

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved normalized dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
