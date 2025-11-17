# BioGraphX – Biomedical QA System

BioGraphX is a biomedical question-answering system that integrates NLP, knowledge graphs, retrieval-augmented generation (RAG), and multi-agent reasoning. The system uses three major biomedical datasets—MedQuAD, PubMed Abstracts, and PubMedQA—to build an end-to-end pipeline capable of retrieving, reasoning, and answering domain-specific questions.

---

## Completed Work (Sprint 1)

### 1. Project Structure Setup
A clean, modular repository structure has been created:

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

data/raw/
    medquad.csv
    pubmed_abstracts.csv
    pubmed_qa_pqa_labeled.parquet
    pubmed_qa_pqa_artificial.parquet


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

## Next Steps (Sprint 2)
Sprint 2 will focus on:

1. Cleaning MedQuAD (`etl/clean_medquad.py`)
2. Running SciSpaCy NER over questions and answers
3. Designing the Neo4j knowledge graph schema
4. Generating node and edge CSVs for graph import

This will convert MedQuAD into a structured biomedical knowledge graph, forming the backbone for RAG and multi-agent reasoning.

---
