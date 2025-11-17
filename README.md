# BioGraphX – Biomedical QA System

BioGraphX is a biomedical question-answering system that integrates NLP, knowledge graphs, retrieval-augmented generation (RAG), and multi-agent reasoning. The system uses three major biomedical datasets—MedQuAD, PubMed Abstracts, and PubMedQA—to build an end-to-end pipeline capable of retrieving, reasoning, and answering domain-specific questions.

---

## Completed Work (Sprint 1)

### 1. Project Structure Setup
A clean, modular repository structure has been created:

```
BioGraphX/
│
├── data/
│   ├── raw/           # manually downloaded Kaggle datasets
│   └── processed/     # cleaned data will go here in Sprint 2
│
├── etl/               # cleaning, NER, graph-prep scripts (to be added)
├── graph/             # Neo4j schema + node/edge generators (Sprint 2)
├── rag/               # embedding + retrieval utils (Sprint 3)
├── agents/            # LangGraph pipeline (Sprint 5)
├── training/          # QA model fine-tuning (Sprint 4)
│
├── notebooks/         
│   └── 01_eda_medquad.ipynb   # first EDA notebook (completed)
│
├── requirements.txt   # full environment dependencies
└── README.md
```


Many folders are empty for now—they will be filled during their respective sprints.  
This structure mirrors an industry-grade NLP project layout.

---

### 2. Environment Setup
A dedicated Python virtual environment (`.venv`) was created.  
All required libraries were installed, including:

- pandas, numpy  
- transformers, datasets, sentence-transformers  
- scispacy, spacy  
- chromadb, faiss  
- neo4j, py2neo  
- torch, accelerate, peft  
- streamlit  
- evaluate, rouge-score  
- ipykernel (for Jupyter support)

---

### 3. Dataset Preparation
The following Kaggle datasets were manually downloaded and placed inside:
```
data/raw/
    medquad.csv
    pubmed_abstracts.csv
    pubmed_qa_pqa_labeled.parquet
    pubmed_qa_pqa_artificial.parquet
```

A `.gitignore` file was configured to ensure raw datasets and large files are never pushed to GitHub.

---

### 4. Exploratory Data Analysis (MedQuAD)
A complete EDA notebook was created:

**`notebooks/01_eda_medquad.ipynb`**

This notebook covers:

- Loading MedQuAD directly from `data/raw/`
- Schema inspection (`df.info()`)
- Sample Q/A preview
- Missing value & duplicate detection
- Question and answer length statistics
- Visualization of length distributions

The insights from this EDA will guide data cleaning, NER extraction, and knowledge graph design in Sprint 2.

---

### 5. Git & Branch Setup
- Repository connected to:  
  `https://github.com/dhruvyellanki19/BioGraphX`
- Working on branch: **anvesh_branch**
- Large file errors resolved by removing ZIP files and updating `.gitignore`
- Successful commit and push of Sprint 1 work

---
## Completed Work (Sprint 1)

## Sprint 2 — Biomedical NER and Entity Normalization

### Objectives
Sprint 2 focuses on extracting biomedical entities (diseases and chemicals) from the MedQuAD dataset and preparing structured data for graph construction. This includes NER model setup, entity extraction, normalization, vocabulary creation, and generating Neo4j-ready CSV files.

---

## 1. Biomedical NER Model Setup
A compatible biomedical NER model (BC5CDR) was manually downloaded and installed, since ScispaCy no longer provides public model URLs.

**Installed model:**  
`en_ner_bc5cdr_md-0.5.1`

**Verification script:**  
`Scripts/test_bc5cdr_model.py`

---

## 2. Entity Extraction from MedQuAD
**Script:**  
`Scripts/extract_medquad_entities.py`

**Outputs:**
- `data/processed/medquad_entities.jsonl`
- `data/processed/medquad_with_entities.csv`

Each row now contains:

{
"diseases": [...],
"chemicals": [...]
}


---

## 3. Entity Normalization
**Script:**  
`Scripts/normalize_medquad_entities.py`

Normalization steps applied:
- lowercase conversion  
- trimming whitespace  
- punctuation removal  
- deduplication  
- consistent formatting  

**Output:**  
`data/processed/medquad_with_entities_normalized.csv`

---

## 4. Global Vocabulary Extraction
**Script:**  
`Scripts/generate_unique_entities.py`

**Outputs:**
- `data/processed/unique_diseases.txt`
- `data/processed/unique_chemicals.txt`

These files contain the canonical biomedical vocabulary needed for graph node creation.

---

## 5. Neo4j Ingestion File Preparation
**Script:**  
`Scripts/prepare_neo4j_medquad.py`

This script converts normalized entity data into Neo4j-ready CSV files.

**Generated node files:**
- `nodes_diseases.csv`
- `nodes_chemicals.csv`
- `nodes_questions.csv`

**Generated relationship files:**
- `rels_question_disease.csv`
- `rels_question_chemical.csv`

All files comply with the Neo4j bulk import format (`:ID`, `:START_ID`, `:END_ID`, `:TYPE`).

---

## Sprint 2 Summary
- Biomedical NER pipeline built and verified  
- Entities extracted and normalized  
- Global vocabularies generated  
- Neo4j nodes and relationship CSVs produced  
- All scripts modular and ready for Sprint 3

