# BioGraphX Environment Setup

## Quick Setup Instructions

### 1. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

### 2. Install all requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install biomedical embedding model (optional)
```bash
python scripts/install_embedding_model.py
```

## Manual Model Installation (if script fails)
If the automatic installation fails, you can install the biomedical embedding model manually:

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz --no-deps
```

## Verification
Test your installation:

```python
import spacy
import scispacy
import langchain
import streamlit

# Test biomedical embedding model
nlp = spacy.load("en_ner_bc5cdr_md")
doc = nlp("Aspirin treats headaches")
print([(ent.text, ent.label_) for ent in doc.ents])
```

## Package Overview
- **Core ML**: torch, transformers, sentence-transformers
- **Biomedical NLP**: spacy, scispacy, biomedical embedding model
- **Vector Stores**: chromadb, faiss-cpu
- **Knowledge Graphs**: neo4j, py2neo, networkx
- **LLM Framework**: langchain, langgraph
- **UI**: streamlit, pyvis
- **Testing**: pytest, pytest-cov