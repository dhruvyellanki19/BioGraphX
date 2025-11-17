#!/usr/bin/env python3
"""
Test script for verifying installation of the BC5CDR biomedical NER model.
Run this after environment setup to confirm that spaCy + the model load correctly.
"""

import spacy

def test_bc5cdr_model():
    print("=" * 70)
    print("   BioGraphX — BC5CDR Biomedical NER Model Test")
    print("=" * 70)

    try:
        print("\nLoading model: en_ner_bc5cdr_md ...")
        nlp = spacy.load("en_ner_bc5cdr_md")
        print("✔ Model loaded successfully!\n")

        text = "Aspirin is commonly used to treat fever and inflammation."
        print(f"Testing on sample text:\n  \"{text}\"\n")

        doc = nlp(text)

        if not doc.ents:
            print("⚠ Model loaded but no entities found. Something is off.")
        else:
            print("Entities detected:")
            for ent in doc.ents:
                print(f"  - {ent.text:20s} → {ent.label_}")

        print("\n✔ BC5CDR model test completed.")

    except Exception as e:
        print(" Failed to load BC5CDR model!")
        print("Error:")
        print(e)
        print("\nMake sure:")
        print(" - The model installed correctly")
        print(" - spaCy version is compatible (should be 3.4.x)")
        print(" - You are inside the correct virtual environment")

if __name__ == "__main__":
    test_bc5cdr_model()
