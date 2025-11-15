
# 📘 **BioGraphX — Sprint 1 Documentation**

## 🧬 Project Overview

**BioGraphX** is an end-to-end *Graph-Augmented, Agentic Biomedical Question-Answering System*.
It integrates:

* A Neo4j biomedical knowledge graph
* SciSpaCy-based biomedical entity extraction
* BioBERT/SciBERT embedding-based retrieval
* A LangGraph multi-agent reasoning pipeline
* A Streamlit interface for interpretable QA

This README documents **Sprint 1 deliverables** as defined in the project plan.

---

# 🟦 **Sprint 1 — Project Bootstrapping & Data Download**

Sprint 1 focuses entirely on **setting up the environment**, **downloading datasets**, and **performing initial exploration (EDA)**.
No modeling, extraction, or graph construction happens yet.

---

## ✅ **1. Repository & Environment Setup**

### ✔ Create project structure

A clean folder layout was initialized:

```
BioGraphX/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── etl/
├── graph/
├── rag/
├── agents/
├── training/
├── models/
├── app/
└── configs/
```

### ✔ Create & activate virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# or
venv\Scripts\activate           # Windows
```

### ✔ Install all dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:

* PyTorch, Transformers
* spaCy, SciSpaCy
* Neo4j + Py2Neo
* ChromaDB + FAISS
* LangChain + LangGraph
* Streamlit
* Evaluation tools

---

## ✅ **2. Raw Dataset Download**

All raw datasets from Kaggle were downloaded manually and placed under:

```
data/raw/
```

### Included datasets:

* `medquad.csv` – Main biomedical Q/A dataset
* `pubmed_abstracts.csv` – Corpus for evidence retrieval
* `pubmed_qa_pga_labeled.parquet` – Evaluation dataset
* `pubmed_qa_pga_artificial.parquet` – Supplemental PubMed QA data

These datasets will be cleaned, processed, embedded, and graph-linked in later sprints.

---

## ✅ **3. Biomedical Model Installation & Verification**

The SciSpaCy biomedical NER model was installed using:

```bash
python scripts/install_embedding_model.py
```

or manual fallback:

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s-scip...
```

### ✔ Successful model load test

```python
import spacy
nlp = spacy.load("en_ner_bc5cdr_md")
doc = nlp("Acetaminophen reduces fever and pain.")
print([(ent.text, ent.label_) for ent in doc.ents])
```

**Result:**
Entities such as CHEMICAL and DISEASE were correctly detected.

This confirms that the biomedical NLP stack is ready for Sprint 2.

---

## ✅ **4. EDA Notebook Created**

The required EDA notebook has been created at:

```
notebooks/01_eda_medquad.ipynb
```

### Notebook contents include:

* Loading `medquad.csv`
* `df.head()`, structural inspection
* Random sample Q/A pairs
* Missing value analysis
* Question/answer length distributions
* Optional: Test biomedical NER model inside the notebook

This validates dataset integrity and prepares for cleaning + entity extraction in Sprint 2.

---

# 🟩 **Sprint 1 Summary**

Sprint 1 goals were fully achieved:

| Deliverable                           | Status |
| ------------------------------------- | ------ |
| Project structure initialized         | ✔      |
| Virtual environment created           | ✔      |
| Requirements installed                | ✔      |
| Kaggle datasets downloaded            | ✔      |
| EDA notebook created                  | ✔      |
| SciSpaCy NER model installed & tested | ✔      |

---

# 🚀 **Next Step: Sprint 2**

Sprint 2 will include:

* Data cleaning
* Biomedical entity extraction
* Neo4j schema definition
* Preparing graph node/edge CSVs

