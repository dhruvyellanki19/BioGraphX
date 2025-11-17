#!/usr/bin/env python3
"""
Extract biomedical named entities (diseases and chemicals) from MedQuAD
using the BC5CDR NER model. Produces two outputs:

1. A JSONL file containing entity lists for each question-answer.
2. A cleaned CSV file with original data plus extracted entities.

This script is part of Sprint 2 of the BioGraphX project.
"""

import os
import json
import pandas as pd
import spacy

def load_medquad(path):
    """Load the MedQuAD CSV dataset."""
    print("Loading MedQuAD dataset...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows.")
    return df

def extract_entities_from_text(nlp, text):
    """Run NER on a text string and return diseases and chemicals."""
    if not isinstance(text, str) or not text.strip():
        return {"diseases": [], "chemicals": []}

    doc = nlp(text)

    diseases = []
    chemicals = []

    for ent in doc.ents:
        if ent.label_ == "DISEASE":
            diseases.append(ent.text)
        elif ent.label_ == "CHEMICAL":
            chemicals.append(ent.text)

    return {
        "diseases": list(set(diseases)),
        "chemicals": list(set(chemicals))
    }

def main():
    input_path = "data/raw/medquad.csv"  # Updated for your dataset
    output_jsonl = "data/processed/medquad_entities.jsonl"
    output_csv = "data/processed/medquad_with_entities.csv"

    print("Loading BC5CDR NER model...")
    nlp = spacy.load("en_ner_bc5cdr_md")
    print("Model loaded.")

    df = load_medquad(input_path)

    print("Extracting entities...")
    all_entities = []

    for idx, row in df.iterrows():
        answer_text = row.get("answer", "")
        entities = extract_entities_from_text(nlp, answer_text)
        all_entities.append(entities)

    # Add entities into DataFrame
    df["entities"] = all_entities

    # Save JSONL
    print("Saving JSONL entity file...")
    os.makedirs("data/processed", exist_ok=True)
    with open(output_jsonl, "w") as f:
        for item in all_entities:
            f.write(json.dumps(item) + "\n")

    # Save CSV with entities embedded
    print("Saving cleaned MedQuAD with entities...")
    df.to_csv(output_csv, index=False)

    print("Finished.")
    print(f"Entities saved to: {output_jsonl}")
    print(f"Enhanced dataset saved to: {output_csv}")

if __name__ == "__main__":
    main()
