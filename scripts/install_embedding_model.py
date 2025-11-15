#!/usr/bin/env python3
"""
Script to install the biomedical embedding model separately after requirements installation
"""

import subprocess
import sys

def install_embedding_model():
    """Install the SciSpacy biomedical embedding model"""
    
    print("Installing biomedical embedding model...")
    print("This may take a few minutes as it downloads ~120MB...")
    
    model_url = "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz"
    
    try:
        # Install with --no-deps to avoid version conflicts
        subprocess.run([
            sys.executable, "-m", "pip", "install", model_url, "--no-deps"
        ], check=True)
        
        print("Biomedical embedding model installed successfully!")
        
        # Test the model
        print("\nTesting the model...")
        import spacy
        
        nlp = spacy.load("en_ner_bc5cdr_md")
        doc = nlp("Aspirin is used to treat headache and inflammation.")
        
        print(f"Embedding model test successful! Found {len(doc.ents)} entities:")
        for ent in doc.ents:
            print(f"   - '{ent.text}' -> {ent.label_}")
            
    except subprocess.CalledProcessError as e:
        print(f"Failed to install biomedical embedding model: {e}")
        print("You can try installing it manually with:")
        print(f"pip install {model_url}")
        return False
        
    except Exception as e:
        print(f"Error testing the model: {e}")
        print("Model might be installed but not working properly.")
        return False
    
    return True

if __name__ == "__main__":
    print("BioGraphX Biomedical Embedding Model Installer")
    print("=" * 50)
    
    success = install_embedding_model()
    
    if success:
        print("\n Installation complete!")
        print("\nThe model can recognize:")
        print("  - CHEMICAL: drugs, compounds, molecules")
        print("  - DISEASE: conditions, symptoms, disorders")
    else:
        print("\nInstallation failed. Check error messages above.")
        sys.exit(1)