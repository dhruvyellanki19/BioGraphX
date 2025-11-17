#!/usr/bin/env python3
"""
Automated installer script for the BC5CDR biomedical NER model.
Downloads the .tar.gz file and installs it using pip.
"""

import os
import sys
import subprocess
import urllib.request

MODEL_URL = "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz"
MODEL_NAME = "en_ner_bc5cdr_md-0.5.1.tar.gz"
DOWNLOAD_DIR = "models"

def download_model():
    """Download the BC5CDR model file if not already present."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    model_path = os.path.join(DOWNLOAD_DIR, MODEL_NAME)

    if os.path.exists(model_path):
        print(f"Model already downloaded: {model_path}")
        return model_path

    print("Downloading BC5CDR biomedical NER model...")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print(f"Download complete: {model_path}")

    return model_path

def install_model(model_path):
    """Install the downloaded model using pip."""
    print("Installing model using pip...")
    subprocess.run([sys.executable, "-m", "pip", "install", model_path], check=True)
    print("Model installation complete.")

def test_installation():
    """Verify that the model loads successfully."""
    print("Testing model installation...")
    try:
        import spacy
        nlp = spacy.load("en_ner_bc5cdr_md")
        print("Model loaded successfully.")
        return True
    except Exception as e:
        print("Model failed to load.")
        print("Error details:")
        print(e)
        return False

def main():
    print("BioGraphX — BC5CDR Model Installer")
    print("-" * 60)

    model_path = download_model()
    install_model(model_path)

    if test_installation():
        print("BC5CDR model installation verified.")
    else:
        print("Installation completed but model did not load. Check spaCy version.")

if __name__ == "__main__":
    main()
