# BioGraphX – Biomedical QA System

BioGraphX is a multi-agent biomedical question-answering system that combines NER (BC5CDR), Neo4j graph lookups, MedQuAD + PubMed retrieval (Chroma), and a LoRA-adapted Llama model for grounded answers.

---

## Quickstart (local)

1) Create env and install deps  
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# BC5CDR NER model
python -m spacy download en_ner_bc5cdr_md
```

2) Build vector indexes (after placing datasets in `data/raw/`)  
```bash
python etl/build_chroma_index.py                         # MedQuAD Chroma
PUBMED_MAX_SENTENCES=120000 python etl/build_pubmed_sentence_index.py  # PubMed sentences
```

3) Run inference  
```bash
python inference.py "what are the complications of appendicitis?"
```

4) Run the UI  
```bash
streamlit run app/app.py
```

Notes  
- Large artifacts (`data/vectorstore*`, `data/processed/*.parquet`, models, adapters) are ignored; keep them local.  
- Set `USE_LLM_REWRITE=true` if you want the LoRA model to rewrite the deterministic answer; default is deterministic for reliability.

---

## Deployment guidance
- Keep raw/processed data, vectorstores, and model checkpoints out of git (see `.gitignore`).  
- For a lightweight deploy: build indexes once, then ship only code + requirements to the target; mount data/vectorstores separately or rebuild them there.  
- To push: check status, add only code (e.g., `agents/`, `etl/`, `app/`, `graph/`, `rag/`, `inference.py`, `README.md`, `.gitignore`), commit, and `git push origin <branch>`.

---

## Legacy sprint log (for reference)

### Sprint 1
- Repo scaffolding, env setup, MedQuAD EDA

### Sprint 2 — Biomedical NER and Entity Normalization
- Installed BC5CDR (`en_ner_bc5cdr_md-0.5.1`)
- Extracted and normalized entities from MedQuAD
- Generated vocabularies and Neo4j CSVs (nodes/relationships)

### Current pipeline highlight
- LangGraph agents: query routing, BC5CDR NER, Neo4j graph lookup, dual vector retrieval (MedQuAD + PubMed), evidence selection, fusion, and deterministic answer generator (optional LoRA rewrite).
