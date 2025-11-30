# BioGraphX – Biomedical QA System

BioGraphX is a multi-agent biomedical question-answering system that combines NER (BC5CDR), Neo4j graph lookups, MedQuAD + PubMed retrieval (Chroma), and a LoRA-adapted Llama model for grounded answers.

---

## What’s inside
- **Agents (LangGraph):** query router, BC5CDR NER, Neo4j neighbor lookup, dual vector retrieval (MedQuAD + PubMed), evidence selector, fusion, deterministic answer generator (optional LoRA rewrite).
- **ETL:** build MedQuAD embeddings, Chroma indexes, PubMed sentence index.
- **Models:** sentence-transformers `NeuML/pubmedbert-base-embeddings`, BC5CDR spaCy NER, and a local LoRA adapter for Llama 3.2-1B.
- **UI:** Streamlit app (`app/app.py`) that calls `inference.py` and renders evidence.

---

## Quickstart (local dev)

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

## Deploying to Streamlit Cloud
Streamlit Cloud has strict space/time limits, so don’t ship large indexes or model weights in git.

### Minimal steps
1) **Repository hygiene**
   - Ensure `.gitignore` is respected (`data/`, `vectorstore`, `models/`, `llm/outputs/` stay local).
   - Push only code: `agents/`, `app/`, `etl/`, `graph/`, `rag/`, `inference.py`, `requirements.txt`, `.gitignore`, `README.md`.
2) **Data & indexes**
   - Option A (recommended): host `data/processed/*.parquet` and `data/vectorstore*` on object storage (S3/GCS/etc.). In `app.py` startup, download them to `/tmp` or `./data/` before serving.  
   - Option B: rebuild indexes on boot with a smaller subset to stay within memory/time limits (e.g., lower `PUBMED_MAX_SENTENCES`).
3) **Models**
   - Sentence embedder (`NeuML/pubmedbert-base-embeddings`) and BC5CDR NER (`en_ner_bc5cdr_md`) will download on first run; include these in `requirements.txt` and use a startup script to pre-download/cache if possible.
   - LoRA adapter: host in remote storage and download on boot to `./llm/outputs/`; or disable rewrite (`USE_LLM_REWRITE=false`) to skip LoRA entirely.
4) **Secrets & config**
   - Add Streamlit secrets or env vars for anything sensitive (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`). If you skip Neo4j, set a fallback in `graph_agent` or mock it.
   - Export `USE_LLM_REWRITE` as desired.
5) **Run command on Streamlit Cloud**
   - Set the app entrypoint to `streamlit run app/app.py`.
   - In `app.py`, ensure any download/bootstrap happens before rendering.

### Lightweight bootstrap snippet (example)
```python
# In app/app.py (top-level)
import os, subprocess

# Example: fetch vectorstore if missing
if not os.path.exists("data/vectorstore"):
    os.makedirs("data", exist_ok=True)
    # replace with your storage URL
    subprocess.run(["curl", "-L", "https://storage.example.com/vectorstore.tar.gz", "-o", "/tmp/vs.tar.gz"], check=True)
    subprocess.run(["tar", "-xzf", "/tmp/vs.tar.gz", "-C", "data"], check=True)
```

If you prefer to rebuild each time, call the ETL scripts with reduced sizes (e.g., `PUBMED_MAX_SENTENCES=20000 python etl/build_pubmed_sentence_index.py`) in a one-time bootstrap block.

---

## Git push checklist (avoid large files)
- `git status -s`
- `git add README.md .gitignore agents/ app/ etl/ graph/ rag/ inference.py requirements.txt`
- `git commit -m "Add deploy notes and update ignores"`
- `git push origin <branch>`

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
