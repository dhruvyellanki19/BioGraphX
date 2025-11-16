import csv
import ast
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "pubmed_abstracts.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_PATH = OUT_DIR / "pubmed_sentences.csv"


def simple_sentence_split(text: str) -> list:
    """
    Clean, minimal and safe sentence splitter.
    """
    if not isinstance(text, str):
        return []
    text = " ".join(text.split())  # normalize spacing

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
    """
    Safely extract abstract text from:
    - plain strings
    - Python list strings: "(['text'],)"
    - nested lists
    """
    if pd.isna(cell):
        return None
    if not isinstance(cell, str):
        return None

    raw = cell.strip()
    if not raw:
        return None

    # Try parsing list-like structures
    if raw.startswith("(") or raw.startswith("["):
        try:
            parsed = ast.literal_eval(raw)

            # Example structures:
            # ['abstract'], (['abstract'],), (['abstract'], ['kw'])
            if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                first = parsed[0]

                # If first element is itself a list
                if isinstance(first, (list, tuple)) and len(first) > 0:
                    return str(first[0])

                return str(first)

        except Exception:
            return raw  # fallback

    return raw


def prepare():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing file: {RAW_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        OUT_PATH.unlink()

    print(f"Reading: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} rows")

    # ----------------------------------------------------------------------
    # Detect content columns for this dataset:
    # Keep all columns EXCEPT:
    # - "_links" columns
    # - "Unnamed: 0"
    # ----------------------------------------------------------------------
    abstract_columns = [
        c for c in df.columns
        if not c.lower().endswith("_links")
        and c != "Unnamed: 0"
        and c.strip() != ""
    ]

    print("Detected content columns (these contain abstracts):")
    print(abstract_columns)

    total_abstracts = 0
    total_sentences = 0

    with OUT_PATH.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["pmid", "sentence"])

        for row_idx, row in df.iterrows():
            for col in abstract_columns:
                cell = row[col]
                text = extract_abstract_text(cell)

                if not text or not isinstance(text, str):
                    continue

                pmid = f"{col}_{row_idx}"

                sentences = simple_sentence_split(text)

                for sent in sentences:
                    if sent.strip():
                        writer.writerow([pmid, sent])
                        total_sentences += 1

                total_abstracts += 1

    print("Done.")
    print(f"Total abstract entries processed: {total_abstracts}")
    print(f"Total sentences extracted: {total_sentences}")
    print(f"Output written to: {OUT_PATH}")


if __name__ == "__main__":
    prepare()
