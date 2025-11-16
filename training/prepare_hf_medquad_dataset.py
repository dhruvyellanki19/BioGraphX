import pandas as pd
from datasets import Dataset

RAW = "data/processed/medquad_clean.csv"
OUT_TRAIN = "data/processed/evaluation/medquad_train.json"
OUT_VAL = "data/processed/evaluation/medquad_val.json"

def main():
    df = pd.read_csv(RAW)

    # HuggingFace dataset format
    df = df.rename(columns={"question": "input", "answer": "output"})
    df = df.sample(frac=1, random_state=42)

    val_size = int(len(df) * 0.1)
    train_df = df[:-val_size]
    val_df = df[-val_size:]

    Dataset.from_pandas(train_df).to_json(OUT_TRAIN)
    Dataset.from_pandas(val_df).to_json(OUT_VAL)

    print("Train size:", len(train_df))
    print("Val size:", len(val_df))

if __name__ == "__main__":
    main()
