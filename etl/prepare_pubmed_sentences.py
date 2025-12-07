import pandas as pd
import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "pubmed_abstracts.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_PATH = OUT_DIR / "pubmed_sentences.parquet"


def simple_sentence_split(text: str) -> list:
    if not isinstance(text, str):
        return []
    text = " ".join(text.split())  # normalize whitespace

    sentences = []
    buf = []

    for ch in text:
        buf.append(ch)
        if ch in [".", "?", "!"]:
            s = "".join(buf).strip()
            if s:
                sentences.append(s)
            buf = []

    last = "".join(buf).strip()
    if last:
        sentences.append(last)

    return sentences


def extract_abstract_text(cell):
    if pd.isna(cell):
        return None
    if not isinstance(cell, str):
        return None

    raw = cell.strip()
    if not raw:
        return None

    if raw.startswith("(") or raw.startswith("["):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                first = parsed[0]
                if isinstance(first, (list, tuple)) and len(first) > 0:
                    return str(first[0])
                return str(first)
        except Exception:
            return raw

    return raw


def prepare():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing file: {RAW_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} rows")

    abstract_columns = [
        c for c in df.columns
        if not c.lower().endswith("_links")
        and c != "Unnamed: 0"
        and c.strip() != ""
    ]

    print("Detected content columns:", abstract_columns)

    rows = []
    total_sentences = 0

    for row_idx, row in df.iterrows():
        for col in abstract_columns:
            cell = row[col]
            text = extract_abstract_text(cell)

            if not text:
                continue

            pmid = f"{col}_{row_idx}"
            sentences = simple_sentence_split(text)

            for sent in sentences:
                rows.append({"pmid": pmid, "sentence": sent})
                total_sentences += 1

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(OUT_PATH, index=False)

    print("Done.")
    print(f"Total sentences extracted: {total_sentences}")
    print(f"Output written to: {OUT_PATH}")


if __name__ == "__main__":
    prepare()
