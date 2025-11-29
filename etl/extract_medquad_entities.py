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


def load_medquad(path: str) -> pd.DataFrame:
    """Load the MedQuAD CSV dataset."""
    print("Loading MedQuAD dataset...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows.")
    return df


def ensure_question_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the dataset has a stable question_id column.
    If it does not exist, create it as a simple sequential integer starting from 1.
    """
    if "question_id" not in df.columns:
        print("Column 'question_id' not found. Creating sequential question_id...")
        df = df.copy()
        df.insert(0, "question_id", range(1, len(df) + 1))
    else:
        print("Column 'question_id' already present. Keeping existing IDs.")
    return df


def extract_entities_from_text(nlp, text: str) -> dict:
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

    # Deduplicate within each text
    return {
        "diseases": sorted(set(diseases)),
        "chemicals": sorted(set(chemicals)),
    }


def main():
    input_path = "data/raw/medquad.csv"
    output_jsonl = "data/processed/medquad_entities.jsonl"
    output_csv = "data/processed/medquad_with_entities.csv"

    print("Loading BC5CDR NER model (en_ner_bc5cdr_md)...")
    nlp = spacy.load("en_ner_bc5cdr_md")
    print("Model loaded.")

    # Load original MedQuAD CSV
    df = load_medquad(input_path)

    # Ensure we have a stable ID that flows through the pipeline and into Neo4j
    df = ensure_question_id(df)

    print("Extracting entities from question + answer...")
    all_entities = []

    for idx, row in df.iterrows():
        question_text = str(row.get("question", "") or "")
        answer_text = str(row.get("answer", "") or "")

        # Combine question + answer so that we don't miss entities present only in the question
        combined = (question_text + " " + answer_text).strip()
        entities = extract_entities_from_text(nlp, combined)
        all_entities.append(entities)

    # Add entities into DataFrame
    df["entities"] = all_entities

    # Ensure output directory exists
    os.makedirs("data/processed", exist_ok=True)

    # Save JSONL – one dict per line
    print("Saving JSONL entity file...")
    with open(output_jsonl, "w") as f:
        for item in all_entities:
            f.write(json.dumps(item) + "\n")

    # Save CSV with entities embedded
    print("Saving MedQuAD with entities column...")
    df.to_csv(output_csv, index=False)

    print("Finished.")
    print(f"Entities saved to: {output_jsonl}")
    print(f"Enhanced dataset saved to: {output_csv}")


if __name__ == "__main__":
    main()