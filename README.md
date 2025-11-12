# BioGraphX
An Agentic AI Framework for Explainable Biomedical Knowledge Discovery


Let’s turn your **Graph-Augmented Agentic Biomedical QA** idea into a **structured implementation plan**, divided into realistic **sprints (agile style)**.

Each sprint is 1–2 weeks, with **clear goals, deliverables, tools, and verification**.
This plan assumes a total duration of about **8 weeks (4 sprints)** — easily adjustable.

---

## 🚀 **High-Level Roadmap**

| Sprint          | Theme                                | Core Goal                                                                            |
| --------------- | ------------------------------------ | ------------------------------------------------------------------------------------ |
| 🧩 **Sprint 1** | Data ingestion & entity mapping      | Collect and normalize Kaggle datasets → clean, link, and prepare for graph ingestion |
| 🕸 **Sprint 2** | Graph construction & Neo4j setup     | Build biomedical knowledge graph with nodes & edges                                  |
| 🧠 **Sprint 3** | Retrieval & Agentic reasoning layer  | Build vector store, retrieval pipeline, multi-agent reasoning                        |
| 💬 **Sprint 4** | Explainable UI + evaluation + polish | Create Streamlit interface, test explainability & evaluation                         |

---

## 🧩 **Sprint 1 — Data Ingestion & Entity Mapping**

### 🎯 **Goal**

Prepare all biomedical data (from Kaggle) and unify entities with common IDs so later modules can join seamlessly.

### 🧱 **Tasks**

1. **Set up environment**

   * Install: `pandas`, `numpy`, `scispacy`, `neo4j`, `py2neo`, `sentence-transformers`, `chromadb`, `langchain`, `langgraph`, `streamlit`.
   * Initialize Git + Docker setup for reproducibility.

2. **Download Kaggle datasets**

   * `MedQuAD`
   * `PubMed Abstracts`
   * `PubMedQA`
   * `MedMCQA`
   * `Drug–Disease Interactions`

3. **Data cleaning & formatting**

   * Remove nulls, duplicates, HTML artifacts.
   * Normalize column names (`drug`, `disease`, `relation`, etc.).
   * Save cleaned CSVs in `data/processed/`.

4. **Entity extraction (NER)**

   * Use **SciSpaCy** with UMLS model:
     `en_core_sci_lg` + `UmlsEntityLinker`
   * Extract mentions of **Drugs**, **Diseases**, **Genes** from MedQuAD and PubMed text.

5. **Entity mapping**

   * Create mapping tables:

     * `drug_name → DrugBank_ID`
     * `disease_name → UMLS_CUI`
     * `gene_symbol → Entrez_ID`
   * Deduplicate synonyms.

6. **Validation**

   * Check overlap of entities between datasets (MedQuAD ↔ PubMed ↔ Drug–Disease).
   * Expect > 80 % coverage for major entities.

### 📦 **Deliverables**

* Cleaned datasets (`.csv`)
* Mapping tables for drugs/diseases/genes
* Notebook showing extraction + coverage summary

### 🧰 **Key Tools**

`pandas`, `SciSpaCy`, `UMLS`, `DrugBank`, `Entrez`, `Python`

---

## 🕸 **Sprint 2 — Graph Construction & Neo4j Setup**

### 🎯 **Goal**

Create a **Biomedical Knowledge Graph (BKG)** integrating the cleaned data.

### 🧱 **Tasks**

1. **Set up Neo4j**

   * Launch via Docker or Neo4j Desktop.
   * Create DB: `biomed_graph`.

2. **Design schema**

   * **Nodes:** `Drug`, `Disease`, `Gene`, `Publication`, `Question`
   * **Relationships:**

     ```
     (Drug)-[:TREATS]->(Disease)
     (Disease)-[:ASSOCIATED_WITH]->(Gene)
     (Drug)-[:MENTIONED_IN]->(Publication)
     (Question)-[:REFERS_TO]->(Disease|Drug)
     ```
   * Properties: `id`, `name`, `source`, `confidence`, `pmid`.

3. **Load nodes & edges**

   * Write `src/06_load_neo4j.py` loader using `py2neo` or `cypher-shell`.
   * Load:

     * `nodes_drug.csv`
     * `nodes_disease.csv`
     * `rel_drug_treats_disease.csv`
   * Add uniqueness constraints.

4. **Cross-verify**

   * Run sample Cypher queries:

     ```cypher
     MATCH (d:Drug)-[:TREATS]->(s:Disease)
     RETURN d.name, s.name LIMIT 10;
     ```

5. **Integrate publications**

   * Link `Drug` and `Disease` to `Publication` using PMIDs from PubMed abstracts.

6. **Visualize subgraphs**

   * Use `pyvis` or Neo4j Bloom to explore structure.

### 📦 **Deliverables**

* Functional Neo4j instance with connected nodes
* Example Cypher queries
* Screenshot or graph visualization for report

### 🧰 **Key Tools**

`Neo4j`, `py2neo`, `pandas`, `pyvis`, `Docker`

---

## 🧠 **Sprint 3 — Retrieval & Agentic Reasoning Layer**

### 🎯 **Goal**

Implement **Retrieval-Augmented Generation (RAG)** + multi-agent reasoning pipeline.

### 🧱 **Tasks**

1. **Vector database**

   * Split PubMed abstracts into sentences.
   * Encode using `BioBERT` / `SciBERT` (via `sentence-transformers`).
   * Store embeddings in **ChromaDB**.

2. **Retriever testing**

   * Query similarity:

     ```python
     collection.query(query_texts=["metformin alzheimer mechanism"], n_results=5)
     ```
   * Validate top results manually.

3. **Agent design (LangGraph)**

   * **Parser Agent:** Extract entities from question.
   * **Graph Query Agent:** Query Neo4j for relationships.
   * **Evidence Agent:** Retrieve relevant abstracts from Chroma.
   * **Synthesis Agent:** Combine graph + text to craft an answer.
   * **Explanation Agent:** Format with citations + graph paths.

4. **Pipeline orchestration**

   * Connect all agents in LangGraph DAG:

     ```
     Parser → GraphQuery → Evidence → Synthesis → Explanation
     ```
   * Use **BioGPT** or **GPT-4** for the Synthesis node.

5. **Logging + traceability**

   * Save reasoning steps as JSON (for explainability).

### 📦 **Deliverables**

* Chroma vector index
* Working LangGraph multi-agent pipeline
* JSON responses containing `answer`, `graph_paths`, `evidence (PMIDs)`

### 🧰 **Key Tools**

`BioBERT`, `ChromaDB`, `LangChain`, `LangGraph`, `BioGPT`, `Neo4j`

---

## 💬 **Sprint 4 — Explainable UI + Evaluation**

### 🎯 **Goal**

Deliver an interactive, explainable biomedical QA system with metrics.

### 🧱 **Tasks**

1. **Build Streamlit interface**

   * Input: user question
   * Output:

     * Final answer (with PMIDs)
     * Evidence table (sentences, scores)
     * Graph visualization (pyvis/NetworkX)

2. **Evaluation**

   * Quantitative:

     * **Answer accuracy:** EM & F1 on PubMedQA
     * **Retrieval quality:** Precision@k
   * Qualitative:

     * 20 human-rated explanations (relevance & clarity)

3. **Ethical review**

   * Add disclaimer (“not for clinical use”).
   * Discuss transparency & bias mitigation.

4. **Packaging**

   * Dockerize: `Neo4j`, `ChromaDB`, `Streamlit` containers.
   * Set up GitHub Actions for CI/CD (test → build → deploy).

5. **Presentation assets**

   * Demo video (Streamlit walkthrough).
   * Report diagrams (architecture + workflow).

### 📦 **Deliverables**

* Streamlit explainable QA app
* Evaluation metrics table + graphs
* Dockerized system
* Final report & slides

### 🧰 **Key Tools**

`Streamlit`, `networkx`, `matplotlib`, `Docker`, `GitHub Actions`

---

## 📊 **Sprint-wise Deliverables Summary**

| Sprint | Key Output                   | Verification                                           |
| ------ | ---------------------------- | ------------------------------------------------------ |
| 1      | Clean data + entity mappings | Coverage ≥ 80 %, correct canonical IDs                 |
| 2      | Neo4j graph                  | Valid `Drug–Disease–Gene` edges, graph queries succeed |
| 3      | RAG + Agents                 | JSON answers include citations & graph paths           |
| 4      | UI + Evaluation              | App usable, metrics reported, Docker build runs        |

---

## 🧩 **Timeline at a Glance**

```
Weeks 1–2 → Sprint 1  (Data & Mapping)
Weeks 3–4 → Sprint 2  (Graph DB)
Weeks 5–6 → Sprint 3  (Agents & Retrieval)
Weeks 7–8 → Sprint 4  (UI & Evaluation)
```

---

## 🧠 **Tips for Success**

* **Work incrementally:** test each module standalone before chaining.
* **Track provenance:** store dataset name & PMID in every edge.
* **Document decisions:** keep `README.md` for every script.
* **Automate small samples first** — scale later.
* **End each sprint with a short demo** (Jupyter notebook or UI prototype).


Chatgpt Chat- https://chatgpt.com/share/6913ede5-d3dc-8012-b77e-0c34f4e5568a

---

Would you like me to generate a **visual Gantt-style sprint timeline** (as a diagram or PDF) showing all four sprints, their dependencies, and deliverables? It’s great for inclusion in a final report or presentation.
