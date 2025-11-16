import pandas as pd
import spacy
from tqdm import tqdm

CLEAN_PATH = "data/processed/medquad_clean.csv"
OUT_PATH   = "data/processed/entity_mappings.csv"

def clean_for_spacy(text):
    """Ensure the text going into spaCy is always a valid string."""
    if pd.isna(text):
        return ""
    return str(text).strip()

def main():
    df = pd.read_csv(CLEAN_PATH)

    # Ensure absolutely no NaN survives in question/answer
    df["question"] = df["question"].apply(clean_for_spacy)
    df["answer"]   = df["answer"].apply(clean_for_spacy)

    # Combine text safely
    texts = (df["question"] + " " + df["answer"]).apply(clean_for_spacy).tolist()
    q_ids = df.index.tolist()

    # Load biomedical NER model
    nlp = spacy.load("en_ner_bc5cdr_md", disable=["parser", "tagger", "lemmatizer"])

    rows = []

    print("Running NER...")
    for qid, doc in tqdm(zip(q_ids, nlp.pipe(texts, batch_size=32)), total=len(texts)):
        for ent in doc.ents:
            rows.append({
                "question_id": qid,
                "entity_text": ent.text,
                "entity_type": ent.label_
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print("Saved:", OUT_PATH)

if __name__ == "__main__":
    main()
