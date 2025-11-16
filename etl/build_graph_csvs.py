import pandas as pd
import os

ENTITY_PATH = "data/processed/entity_mappings.csv"
CLEAN_PATH  = "data/processed/medquad_clean.csv"
OUT_DIR     = "data/processed/graph"

os.makedirs(OUT_DIR, exist_ok=True)

def main():
    entities = pd.read_csv(ENTITY_PATH)
    df = pd.read_csv(CLEAN_PATH)

    # --- Question nodes ---
    q_df = pd.DataFrame({
        "id": df.index,
        "text": df["question"]
    })
    q_df.to_csv(f"{OUT_DIR}/nodes_question.csv", index=False)

    # --- Disease nodes ---
    diseases = entities[entities["entity_type"] == "DISEASE"]["entity_text"].str.lower().drop_duplicates()
    pd.DataFrame({"name": diseases}).to_csv(f"{OUT_DIR}/nodes_disease.csv", index=False)

    # --- Drug nodes ---
    drugs = entities[entities["entity_type"] == "CHEMICAL"]["entity_text"].str.lower().drop_duplicates()
    pd.DataFrame({"name": drugs}).to_csv(f"{OUT_DIR}/nodes_drug.csv", index=False)

    # --- Relations ---
    rels = entities.rename(columns={
        "question_id": "start_id",
        "entity_text": "end_name"
    })
    rels["rel_type"] = "ABOUT"
    rels = rels[["start_id", "end_name", "rel_type"]]
    rels.to_csv(f"{OUT_DIR}/rels_about.csv", index=False)

    print("Graph CSVs saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
