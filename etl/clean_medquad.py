import os
import pandas as pd

RAW_PATH = os.path.join("data", "raw", "medquad.csv")
OUT_PATH = os.path.join("data", "processed", "medquad_clean.csv")

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = text.strip()
    text = text.replace("\n", " ")
    text = " ".join(text.split())  # remove extra spaces
    return text

def main():
    print("Loading MedQuAD...")
    df = pd.read_csv(RAW_PATH)

    print("Cleaning text...")
    df["question"] = df["question"].astype(str).apply(clean_text)
    df["answer"] = df["answer"].astype(str).apply(clean_text)

    print("Dropping duplicates + NA rows...")
    df = df.dropna(subset=["question", "answer"])
    df = df.drop_duplicates(subset=["question", "answer"])

    print("Saving cleaned dataset...")
    df.to_csv(OUT_PATH, index=False)

    print("Done. Saved to:", OUT_PATH)

if __name__ == "__main__":
    main()
