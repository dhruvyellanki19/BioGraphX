import pandas as pd

def main():
    print("Loading PubMedQA parquet files...")

    df1 = pd.read_parquet("data/raw/pubmed_qa_pga_labeled.parquet")
    df2 = pd.read_parquet("data/raw/pubmed_qa_pga_artificial.parquet")

    df = pd.concat([df1, df2])

    df = df.rename(columns={
        "QUESTION": "question",
        "LONG_ANSWER": "long_answer"
    })

    df = df[["question", "long_answer"]].dropna()

    print("Saving - data/processed/pubmedqa_clean.csv")
    df.to_csv("data/processed/pubmedqa_clean.csv", index=False)


if __name__ == "__main__":
    main()
