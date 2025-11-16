import pandas as pd
import re
import os

RAW_PATH = "data/raw/medquad.csv"
OUT_PATH = "data/processed/medquad_clean.csv"

def clean_text(t):
    """Clean and normalize MedQuAD text safely."""
    if pd.isna(t):
        return ""

    t = str(t).strip()
    t = re.sub(r"<.*?>", "", t)        # remove HTML tags
    t = re.sub(r"\s+", " ", t)        # normalize whitespace
    return t

def main():
    # Load raw MedQuAD
    df = pd.read_csv(RAW_PATH)

    # Fill missing values BEFORE applying any string ops
    df["question"] = df["question"].fillna("")
    df["answer"]   = df["answer"].fillna("")

    # Convert everything to clean text
    df["question"] = df["question"].apply(clean_text)
    df["answer"]   = df["answer"].apply(clean_text)

    # Remove rows where question or answer is empty AFTER cleaning
    df = df[(df["question"].str.strip() != "") & (df["answer"].str.strip() != "")]

    # Remove duplicates
    df = df.drop_duplicates(subset=["question", "answer"])

    # Keep only necessary columns
    keep_cols = ["question", "answer"]
    df = df[keep_cols]

    # Save cleaned dataset
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Total rows after cleaning:", len(df))

if __name__ == "__main__":
    main()
