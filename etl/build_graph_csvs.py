import pandas as pd
import os

ENTITY_PATH = "data/processed/entity_mappings.csv"
CLEAN_PATH  = "data/processed/medquad_clean.csv"
OUT_DIR     = "data/processed/graph_data"

os.makedirs(OUT_DIR, exist_ok=True)


def normalize(s: str):
    """Lowercase + strip + remove duplicate spaces."""
    return " ".join(str(s).strip().lower().split())


def main():
    entities = pd.read_csv(ENTITY_PATH)
    df = pd.read_csv(CLEAN_PATH)

    # Normalize entity text everywhere
    entities["entity_text"] = entities["entity_text"].apply(normalize)
    df["question"] = df["question"].apply(normalize)

    # --- Question nodes ---
    q_df = pd.DataFrame({
        "id": df.index,
        "text": df["question"]
    })
    q_df.to_csv(f"{OUT_DIR}/nodes_question.csv", index=False)

    # --- Disease nodes ---
    diseases = entities[entities["entity_type"] == "DISEASE"]["entity_text"].drop_duplicates()
    pd.DataFrame({"name": diseases}).to_csv(f"{OUT_DIR}/nodes_disease.csv", index=False)

    # --- Drug nodes ---
    drugs = entities[entities["entity_type"] == "CHEMICAL"]["entity_text"].drop_duplicates()
    pd.DataFrame({"name": drugs}).to_csv(f"{OUT_DIR}/nodes_drug.csv", index=False)

    # --- Relationship CSV ---
    rels = entities[["question_id", "entity_text", "entity_type"]].copy()
    rels.rename(columns={
        "question_id": "start_id",
        "entity_text": "end_name",
        "entity_type": "type"
    }, inplace=True)

    # Normalize
    rels["end_name"] = rels["end_name"].apply(normalize)
    rels["type"] = rels["type"].str.upper()

    # About relationship
    rels["rel_type"] = "ABOUT"

    rels.to_csv(f"{OUT_DIR}/rels_about.csv", index=False)

    print("Graph CSVs saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
