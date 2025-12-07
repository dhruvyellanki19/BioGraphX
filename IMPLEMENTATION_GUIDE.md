# BioGraphX Implementation Guide

**A comprehensive guide to understanding, setting up, and running the BioGraphX biomedical question answering system from scratch.**

---

## Table of Contents

1. [Introduction](#introduction)
2. [System Architecture Deep Dive](#system-architecture-deep-dive)
3. [Directory Structure Explained](#directory-structure-explained)
4. [File-by-File Explanation](#file-by-file-explanation)
5. [Step-by-Step Implementation](#step-by-step-implementation)
6. [Configuration Guide](#configuration-guide)
7. [Usage Examples](#usage-examples)
8. [Troubleshooting](#troubleshooting)
9. [Development Guide](#development-guide)

---

## Introduction

### What This Guide Covers

This implementation guide provides:
- **Detailed explanations** of every major file in the project
- **Step-by-step instructions** to set up the system from scratch
- **Data pipeline walkthrough** from raw data to production
- **Agent system architecture** and interaction patterns
- **Model training process** for fine-tuning Qwen
- **Docker deployment** best practices
- **Troubleshooting** common issues

### Prerequisites Knowledge

To fully understand this guide, you should have:
- Basic Python programming (functions, classes, imports)
- Familiarity with machine learning concepts
- Understanding of REST APIs
- Basic Docker knowledge (helpful but not required)

### Expected Outcomes

After following this guide, you will be able to:
1. Set up the complete BioGraphX system from scratch
2. Understand how each component works
3. Modify and extend the system
4. Deploy the application using Docker
5. Troubleshoot common issues

---

## System Architecture Deep Dive

### Data Flow Diagram

```
Raw Data (MedQuAD, PubMed)
         |
         v
    [ETL Pipeline]
    - Clean text
    - Extract entities
    - Process sentences
         |
         v
    [Vector Database]
    - Embed sentences (all-mpnet-base-v2)
    - Store in ChromaDB (298K documents)
         |
         v
    [Agent Pipeline]
         |
    User Question
         |
         v
    [QuestionAgent] --> Extract entities (SciSpaCy)
         |
         v
    [NormalizeAgent] --> Fuzzy match entities
         |
         v
    [WikipediaAgent] --> Retrieve general context
         |
         v
    [RetrieverAgent] --> Search ChromaDB
         |
         v
    [QAModelAgent] --> Generate answer (Qwen)
         |
         v
    [EvidenceAgent] --> Format citations
         |
         v
    [ExplanationAgent] --> Compile response
         |
         v
    Final Answer + Evidence
```

### Agent Interaction Patterns

Each agent follows a consistent pattern:

```python
class Agent:
    def __init__(self):
        # Load models/data
        pass
    
    def run(self, state):
        # Process state
        # Update state with results
        return state
```

**State Object**: A dictionary passed between agents containing:
- `question`: Original user question
- `entities`: Extracted biomedical entities
- `normalized_entities`: Canonical entity names
- `wikipedia`: General medical context
- `evidence`: PubMed research findings
- `answer`: Generated response
- `explanation`: Reasoning process

### Model Pipeline

```
Input: "What are the symptoms of diabetes?"
         |
         v
    [SciSpaCy NER]
    Entities: ["diabetes"]
         |
         v
    [Entity Normalization]
    Canonical: ["diabetes mellitus"]
         |
         v
    [Embedding]
    Vector: [0.123, -0.456, ...] (768-dim)
         |
         v
    [ChromaDB Search]
    Top-5 similar documents
         |
         v
    [Context Assembly]
    Wikipedia + PubMed evidence
         |
         v
    [Qwen Model]
    Prompt: "Based on evidence: ..."
         |
         v
    Output: Evidence-based answer
```

---

## Directory Structure Explained

### agents/ - Multi-Agent System

**Purpose**: Contains the 7 specialized agents that form the core question-answering pipeline.

**Key Files**:
- `agent_graph_orchestrator.py`: Coordinates agent execution
- `question_agent.py`: Entity extraction
- `normalize_agent.py`: Entity normalization
- `wikipedia_agent.py`: Wikipedia retrieval
- `retriever_agent.py`: Vector database search
- `qa_model_agent.py`: Answer generation
- `evidence_agent.py`: Evidence formatting
- `explanation_agent.py`: Response compilation

**Dependencies**: SciSpaCy, ChromaDB, Transformers, Wikipedia API

### app/ - Web Application

**Purpose**: Flask-based web interface for interacting with the system.

**Key Files**:
- `main.py`: API endpoints and Flask app
- `pipeline_wrapper.py`: Integration with agent pipeline
- `run.py`: Application launcher
- `templates/`: HTML templates for UI
- `static/`: CSS and JavaScript files

**Port**: 8989 (local), 5000 (Docker internal), 1919 (Docker external)

### data/ - Data Storage

**Purpose**: Stores raw datasets, processed data, and vector database.

**Structure**:
```
data/
├── raw/                    # Original datasets
│   ├── medquad.csv        # 16,412 Q&A pairs
│   └── pubmed_abstracts/  # PubMed research papers
├── processed/             # Cleaned and processed data
│   ├── medquad_clean.csv  # Cleaned Q&A
│   └── entity_mappings.csv # 15,723 canonical entities
└── chroma/                # ChromaDB vector database
    └── chroma.sqlite3     # 298,152 indexed sentences
```

### etl/ - Data Processing

**Purpose**: Scripts for cleaning and processing raw data.

**Key Files**:
- `prepare_pubmed_sentences.py`: Extract sentences from PubMed abstracts
- `clean_pubmedqa.py`: Clean and normalize PubMedQA dataset

**Process**: Raw CSV -> Clean text -> Sentence splitting -> Entity extraction

### rag/ - Retrieval-Augmented Generation

**Purpose**: Vector database creation and semantic search.

**Key Files**:
- `embed_pubmed.py`: Create embeddings and build ChromaDB index
- `query_pipeline.py`: Complete RAG pipeline
- `build_index.py`: Index construction utilities
- `test_retriever.py`: Testing and validation

**Embedding Model**: sentence-transformers/all-mpnet-base-v2 (768-dimensional)

### training/ - Model Fine-Tuning

**Purpose**: Scripts for preparing datasets and fine-tuning the Qwen model.

**Key Files**:
- `prepare_medquad_dataset.py`: Clean MedQuAD for training
- `prepare_hf_medquad_dataset.py`: Convert to HuggingFace format
- `finetune_qa_model.py`: LoRA fine-tuning script
- `merge_lora.py`: Merge LoRA adapters with base model
- `eval_pubmedqa.py`: Evaluate model performance

**Training Data**: 14,724 biomedical Q&A pairs

### models/ - Trained Models

**Purpose**: Storage for fine-tuned models.

**Structure**:
```
models/
└── fine_tuned/
    └── qwen25_1_5b_medquad_merged/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        └── ...
```

### scripts/ - Utility Scripts

**Purpose**: Helper scripts for installation and testing.

**Key Files**:
- `run_pipeline.py`: Command-line interface for testing
- `install_embedding_model.py`: Download SciSpaCy model

---

## File-by-File Explanation

### Core Agent Files

#### agents/agent_graph_orchestrator.py

**Purpose**: Main pipeline coordinator that executes agents in sequence.

**Key Components**:
```python
class AgentGraphPipeline:
    def __init__(self):
        # Initialize all 7 agents
        self.question_agent = QuestionAgent()
        self.normalize_agent = NormalizeAgent()
        self.wikipedia_agent = WikipediaAgent()
        self.retriever_agent = RetrieverAgent()
        self.qa_model_agent = QAModelAgent()
        self.evidence_agent = EvidenceAgent()
        self.explanation_agent = ExplanationAgent()
    
    def run(self, question):
        # Execute agents sequentially
        state = {"question": question}
        state = self.question_agent.run(state)
        state = self.normalize_agent.run(state)
        state = self.wikipedia_agent.run(state)
        state = self.retriever_agent.run(state)
        state = self.qa_model_agent.run(state)
        state = self.evidence_agent.run(state)
        state = self.explanation_agent.run(state)
        return state
```

**Flow**: Question -> Entity Extraction -> Normalization -> Wikipedia -> Retrieval -> Generation -> Evidence -> Explanation

#### agents/question_agent.py

**Purpose**: Extract biomedical entities from questions using SciSpaCy.

**Model**: `en_ner_bc5cdr_md` (BC5CDR corpus trained)

**Entity Types**:
- DISEASE: Medical conditions (e.g., "diabetes", "asthma")
- CHEMICAL: Drugs and compounds (e.g., "metformin", "aspirin")

**Example**:
```python
Input: "What are the side effects of metformin?"
Output: entities = ["metformin"]
```

**Implementation**:
```python
import spacy

class QuestionAgent:
    def __init__(self):
        self.nlp = spacy.load("en_ner_bc5cdr_md")
    
    def run(self, state):
        doc = self.nlp(state["question"])
        entities = [ent.text.lower() for ent in doc.ents]
        state["entities"] = entities
        return state
```

#### agents/normalize_agent.py

**Purpose**: Normalize extracted entities to canonical medical terms using fuzzy matching.

**Method**: RapidFuzz with 70% similarity threshold

**Data Source**: `entity_mappings.csv` (15,723 canonical terms)

**Example**:
```python
Input: entities = ["diabetis"]  # Misspelled
Output: normalized = ["diabetes"]  # Corrected
```

**Algorithm**:
1. Clean entity text (lowercase, remove punctuation)
2. Fuzzy match against canonical vocabulary
3. Accept matches above 70% similarity
4. Return normalized entity or original if no match

#### agents/wikipedia_agent.py

**Purpose**: Retrieve general medical knowledge from Wikipedia.

**API**: Wikipedia Python library

**Process**:
1. Search Wikipedia for each normalized entity
2. Extract article summary (first few paragraphs)
3. Combine summaries for context

**Example**:
```python
Input: normalized_entities = ["diabetes"]
Output: wikipedia = "Diabetes mellitus is a group of metabolic disorders..."
```

#### agents/retriever_agent.py

**Purpose**: Semantic search over PubMed abstracts using ChromaDB.

**Components**:
- **Embedding Model**: all-mpnet-base-v2 (768-dimensional vectors)
- **Vector Database**: ChromaDB with 298,152 documents
- **Search**: Top-5 semantic similarity

**Process**:
1. Embed query using sentence-transformers
2. Search ChromaDB for similar documents
3. Return top-K results with PMID citations

**Example**:
```python
Input: normalized_entities = ["diabetes"]
Output: evidence = [
    {"pmid": "12345", "text": "Diabetes is characterized by..."},
    {"pmid": "67890", "text": "Treatment includes..."}
]
```

#### agents/qa_model_agent.py

**Purpose**: Generate evidence-based answers using fine-tuned Qwen2.5-1.5B.

**Model**: `models/fine_tuned/qwen25_1_5b_medquad_merged/`

**Fallback**: Qwen/Qwen2.5-1.5B-Instruct (if local model not found)

**Prompt Template**:
```
You are a biomedical expert with 25+ years of experience.

Question: {question}

Wikipedia Context: {wikipedia}

PubMed Evidence: {evidence}

Provide a comprehensive, evidence-based answer.
```

**Generation Parameters**:
- Temperature: 0.7
- Max tokens: 512
- Top-p: 0.9

#### agents/evidence_agent.py

**Purpose**: Format evidence with proper citations and structure.

**Output Format**:
```python
{
    "evidence_list": [
        {
            "pmid": "12345",
            "text": "Research finding...",
            "relevance": "high"
        }
    ],
    "citation_count": 5
}
```

#### agents/explanation_agent.py

**Purpose**: Compile final response with reasoning explanation.

**Output Sections**:
1. **Final Answer**: Evidence-based response
2. **Wikipedia Knowledge**: General medical context
3. **Supporting Evidence**: PubMed citations
4. **Reasoning Process**: Step-by-step explanation

**Format**: Markdown with proper headings and structure

---

### Web Application Files

#### app/main.py

**Purpose**: Flask application with API endpoints.

**Endpoints**:

1. **GET /**
   - Serves main chat interface
   - Returns: HTML template

2. **POST /ask**
   - Processes biomedical questions
   - Input: `{"question": "..."}`
   - Output: Complete agent analysis with answer

3. **GET /health**
   - Health check endpoint
   - Returns: `{"status": "healthy", "pipeline_ready": true}`

**Implementation**:
```python
from flask import Flask, request, jsonify
from pipeline_wrapper import PipelineWrapper

app = Flask(__name__)
pipeline = PipelineWrapper()

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')
    result = pipeline.run_pipeline(question)
    return jsonify(result)
```

#### app/pipeline_wrapper.py

**Purpose**: Integration layer between Flask and agent pipeline.

**Key Methods**:

1. **init_pipeline()**: Initialize agent system
2. **run_pipeline(question)**: Execute full pipeline
3. **process_pipeline_output(result)**: Format for web display
4. **generate_graph_html(state)**: Create visualization

**Caching**: Lazy initialization of pipeline (loaded on first request)

#### app/run.py

**Purpose**: Application launcher script.

**Configuration**:
- Debug mode: Controlled by environment variable
- Host: 127.0.0.1 (local) or 0.0.0.0 (Docker)
- Port: 8989 (default)

**Usage**:
```bash
python run.py
```

---

### Data Processing Files

#### etl/prepare_pubmed_sentences.py

**Purpose**: Extract and process sentences from PubMed abstracts.

**Process**:
1. Load raw PubMed CSV files
2. Extract abstracts
3. Split into sentences using NLTK
4. Clean and normalize text
5. Save to processed directory

**Output**: Individual sentences ready for embedding

#### etl/clean_pubmedqa.py

**Purpose**: Clean PubMedQA dataset for evaluation.

**Cleaning Steps**:
1. Remove HTML tags
2. Normalize whitespace
3. Handle missing values
4. Remove duplicates
5. Validate Q&A pairs

---

### RAG Pipeline Files

#### rag/embed_pubmed.py

**Purpose**: Create vector embeddings and build ChromaDB index.

**Process**:
1. Load processed PubMed sentences
2. Generate embeddings using sentence-transformers
3. Batch insert into ChromaDB (1000 sentences per batch)
4. Create persistent database

**Embedding Model**: all-mpnet-base-v2
- Dimensions: 768
- Context window: 384 tokens
- Performance: ~1000 sentences/minute

**Usage**:
```bash
python rag/embed_pubmed.py
```

**Output**: `data/chroma/chroma.sqlite3` (603MB with 298K documents)

#### rag/query_pipeline.py

**Purpose**: Complete RAG pipeline for testing.

**Features**:
- Question processing
- Entity extraction
- Vector search
- Answer generation

**Usage**:
```bash
python rag/query_pipeline.py "What is diabetes?"
```

---

### Training Files

#### training/prepare_medquad_dataset.py

**Purpose**: Clean MedQuAD dataset for model training.

**Cleaning Steps**:
1. Load raw MedQuAD CSV
2. Remove HTML tags: `<.*?>` regex
3. Normalize whitespace: `\s+` -> single space
4. Handle missing values
5. Remove empty Q&A pairs
6. Remove duplicates

**Output**: `data/processed/medquad_clean.csv`

**Statistics**:
- Input: 16,412 Q&A pairs
- Output: ~16,360 clean pairs (after deduplication)

#### training/prepare_hf_medquad_dataset.py

**Purpose**: Convert cleaned data to HuggingFace dataset format.

**Format**:
```json
{
    "instruction": "Answer the following biomedical question:",
    "input": "What is diabetes?",
    "output": "Diabetes is a metabolic disorder..."
}
```

**Split**:
- Training: 14,724 pairs (90%)
- Validation: 1,635 pairs (10%)

**Output Files**:
- `data/processed/evaluation/medquad_train.json`
- `data/processed/evaluation/medquad_val.json`

#### training/finetune_qa_model.py

**Purpose**: Fine-tune Qwen2.5-1.5B using LoRA.

**Configuration**:
```python
# LoRA Config
lora_config = LoraConfig(
    r=8,                          # LoRA rank
    lora_alpha=16,                # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training Config
training_args = TrainingArguments(
    output_dir="models/fine_tuned/qwen25_1_5b_medquad",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=True,                    # Mixed precision
    logging_steps=50,
    save_steps=500
)
```

**Training Process**:
1. Load base Qwen2.5-1.5B-Instruct model
2. Apply LoRA adapters (3.6M trainable parameters)
3. Train on MedQuAD dataset
4. Save LoRA weights

**Training Time**: 8-12 hours on Mac Apple Silicon (MPS)

**Output**: `models/fine_tuned/qwen25_1_5b_medquad/` (LoRA adapters)

#### training/merge_lora.py

**Purpose**: Merge LoRA adapters with base model for deployment.

**Process**:
1. Load base Qwen model
2. Load LoRA adapters
3. Merge weights
4. Save complete model

**Output**: `models/fine_tuned/qwen25_1_5b_medquad_merged/` (full model)

**Advantage**: Faster inference (no adapter overhead)

---

## Step-by-Step Implementation

### Step 1: Environment Setup

**1.1 Clone Repository**
```bash
git clone <repository-url>
cd BioGraphX
```

**1.2 Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**1.3 Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected Time**: 10-15 minutes

---

### Step 2: Install Biomedical Models

**2.1 Download SciSpaCy Model**
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

**2.2 Download NLTK Data**
```bash
python -m nltk.downloader punkt punkt_tab stopwords
```

**2.3 Verify Installation**
```python
import spacy
nlp = spacy.load("en_ner_bc5cdr_md")
doc = nlp("Metformin treats diabetes")
print([(ent.text, ent.label_) for ent in doc.ents])
# Expected: [('Metformin', 'CHEMICAL'), ('diabetes', 'DISEASE')]
```

**Expected Time**: 5 minutes

---

### Step 3: Data Preparation (Optional)

**Note**: Pre-processed data is included in the repository. This step is only needed if you want to rebuild from scratch.

**3.1 Download Raw Datasets**
- MedQuAD: Place in `data/raw/medquad.csv`
- PubMed: Place in `data/raw/pubmed_abstracts/`

**3.2 Clean MedQuAD**
```bash
python training/prepare_medquad_dataset.py
```

**3.3 Process PubMed Sentences**
```bash
python etl/prepare_pubmed_sentences.py
```

**Expected Time**: 30-60 minutes

---

### Step 4: Build Vector Database (Optional)

**Note**: Pre-built ChromaDB is included. Rebuild only if needed.

**4.1 Create Embeddings**
```bash
python rag/embed_pubmed.py
```

**Process**:
- Loads processed PubMed sentences
- Generates 768-dimensional embeddings
- Inserts into ChromaDB in batches
- Creates persistent database

**Expected Time**: 2-4 hours (for 298K documents)

**Output**: `data/chroma/chroma.sqlite3` (603MB)

**4.2 Verify Database**
```python
import chromadb
client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_collection("pubmed_index")
print(f"Documents: {collection.count()}")
# Expected: 298152
```

---

### Step 5: Model Setup

**5.1 Option A: Use Pre-trained Model (Recommended)**

The fine-tuned model should be in `models/fine_tuned/qwen25_1_5b_medquad_merged/`

Verify:
```bash
ls models/fine_tuned/qwen25_1_5b_medquad_merged/
# Should show: config.json, model.safetensors, tokenizer files
```

**5.2 Option B: Fine-tune Your Own Model**

**Prepare Dataset**:
```bash
python training/prepare_hf_medquad_dataset.py
```

**Fine-tune**:
```bash
python training/finetune_qa_model.py
```

**Merge LoRA**:
```bash
python training/merge_lora.py
```

**Expected Time**: 8-12 hours for training + 30 minutes for merging

---

### Step 6: Test Agent Pipeline

**6.1 Run Command-Line Interface**
```bash
python scripts/run_pipeline.py
```

**6.2 Enter Test Question**
```
Enter biomedical question: What are the symptoms of diabetes?
```

**Expected Output**:
```
[Pipeline] Initializing agents...
[QuestionAgent] Loading SciSpaCy biomedical NER...
[NormalizeAgent] Loading canonical entity mappings...
[WikipediaAgent] Initialized for medical article retrieval
[RetrieverAgent] Loading ChromaDB & embedder...
[RetrieverAgent] Found existing pubmed_index collection
[QAModelAgent] Loading fine-tuned Qwen...
[QAModelAgent] Model loaded successfully!
Pipeline loaded successfully!

[QuestionAgent] Extracted entities: ['diabetes']
[NormalizeAgent] Fuzzy-normalizing entities...
[WikipediaAgent] Retrieved 1 Wikipedia articles
[RetrieverAgent] Retrieved 9 evidence sentences.

-----------------------------------------
 FINAL ANSWER
-----------------------------------------
Diabetes is a chronic metabolic disease characterized by elevated
levels of blood glucose... [Evidence: PMID 12345, PMID 67890]
-----------------------------------------
```

**Troubleshooting**:
- If ChromaDB not found: Run Step 4
- If model not found: Uses fallback Qwen model (slower first load)
- If NLTK error: Run `python -m nltk.downloader punkt punkt_tab`

---

### Step 7: Run Web Application

**7.1 Start Flask Server**
```bash
cd app
python run.py
```

**7.2 Access Web Interface**
```
http://localhost:8989
```

**7.3 Test Question**
- Enter: "How does metformin work?"
- Wait 5-8 seconds for processing
- View answer with evidence and citations

**Expected Behavior**:
- Question appears in chat
- Loading indicator shows
- Answer displays with:
  - Wikipedia context
  - PubMed evidence (collapsible)
  - Reasoning explanation

---

### Step 8: Docker Deployment

**8.1 Build Docker Image**
```bash
docker-compose up --build -d
```

**8.2 Verify Container**
```bash
docker-compose ps
```

**Expected Output**:
```
NAME            STATUS                   PORTS
biographx-app   Up (healthy)            0.0.0.0:1919->5000/tcp
```

**8.3 Check Health**
```bash
curl http://localhost:1919/health
```

**Expected Response**:
```json
{
  "status": "healthy",
  "pipeline_ready": true,
  "timestamp": "2025-12-06T20:00:00"
}
```

**8.4 Access Dockerized App**
```
http://localhost:1919
```

**8.5 View Logs**
```bash
docker-compose logs -f
```

**Expected Logs**:
```
Loading BioGraphX Agent Pipeline...
[QuestionAgent] Loading SciSpaCy biomedical NER...
[RetrieverAgent] Found existing pubmed_index collection
[QAModelAgent] Model loaded successfully!
Pipeline loaded successfully!
Pipeline ready for requests
* Running on http://0.0.0.0:5000
```

---

## Configuration Guide

### Environment Variables

Create `.env` file:
```bash
# Flask Configuration
FLASK_DEBUG=false
PORT=5000
HOST=0.0.0.0

# Model Configuration
MODEL_PATH=models/fine_tuned/qwen25_1_5b_medquad_merged

# Database Configuration
CHROMA_PATH=data/chroma

# Application Settings
MAX_EVIDENCE_DOCS=5
SIMILARITY_THRESHOLD=0.7
```

### Model Paths

Edit `agents/qa_model_agent.py`:
```python
MODEL_DIR = "models/fine_tuned/qwen25_1_5b_medquad_merged"
FALLBACK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
```

### ChromaDB Configuration

Edit `agents/retriever_agent.py`:
```python
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "pubmed_index"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOP_K = 5  # Number of documents to retrieve
```

### Port Configuration

**Local Development** (`app/run.py`):
```python
port = int(os.getenv('PORT', 8989))
```

**Docker** (`docker-compose.yaml`):
```yaml
ports:
  - "1919:5000"  # host:container
```

---

## Usage Examples

### Example 1: Simple Medical Question

**Question**: "What is asthma?"

**Expected Response**:
```
Final Answer:
Asthma is a chronic respiratory disease characterized by inflammation
and narrowing of the airways, leading to difficulty breathing, wheezing,
and coughing. [Evidence: PMID 12345, PMID 67890]

Wikipedia Knowledge:
Asthma is a common long-term inflammatory disease of the airways...

Supporting Evidence:
1. PMID 12345: "Asthma is characterized by airway hyperresponsiveness..."
2. PMID 67890: "Treatment includes inhaled corticosteroids..."

Reasoning Process:
1. Extracted entity: "asthma"
2. Retrieved 5 PubMed articles
3. Generated evidence-based answer
```

### Example 2: Drug Mechanism Question

**Question**: "How does metformin work?"

**Entities Extracted**: ["metformin"]

**Evidence Retrieved**: 9 PubMed articles about metformin mechanism

**Answer**: Detailed explanation of metformin's mechanism of action with citations

### Example 3: Treatment Question

**Question**: "What are the treatments for type 2 diabetes?"

**Entities**: ["type 2 diabetes"]

**Wikipedia**: General overview of diabetes

**Evidence**: Research on various treatment approaches

**Answer**: Comprehensive treatment options with evidence

---

## Troubleshooting

### Common Issues

#### 1. "Pipeline initialization failed"

**Cause**: Missing models or data

**Solution**:
```bash
# Check SciSpaCy model
python -c "import spacy; spacy.load('en_ner_bc5cdr_md')"

# Check ChromaDB
python -c "import chromadb; client = chromadb.PersistentClient(path='data/chroma'); print(client.list_collections())"

# Check NLTK data
python -m nltk.downloader punkt punkt_tab
```

#### 2. "ChromaDB collection not found"

**Cause**: Vector database not built

**Solution**:
```bash
python rag/embed_pubmed.py
```

#### 3. "Out of memory" during training

**Cause**: Insufficient RAM

**Solution**:
- Reduce batch size in `finetune_qa_model.py`:
  ```python
  per_device_train_batch_size=1
  gradient_accumulation_steps=16  # Increase this
  ```
- Use gradient checkpointing (already enabled)

#### 4. "Port 1919 already in use"

**Cause**: Another process using the port

**Solution**:
```bash
# Find process
lsof -i :1919

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yaml
ports:
  - "8080:5000"  # Use different port
```

#### 5. "Model not found, using fallback"

**Cause**: Fine-tuned model not in expected location

**Solution**:
```bash
# Check model exists
ls models/fine_tuned/qwen25_1_5b_medquad_merged/

# If missing, either:
# 1. Fine-tune your own (Step 5.2)
# 2. Use fallback (automatic, but slower first load)
```

#### 6. Docker container exits immediately

**Cause**: Build error or missing dependencies

**Solution**:
```bash
# Check logs
docker-compose logs biographx-app

# Rebuild with no cache
docker-compose build --no-cache
docker-compose up -d
```

---

---

## Interactive Notebooks

The `notebooks/` directory contains 5 Jupyter notebooks for learning, experimentation, and analysis.

### notebooks/eda_medquad.ipynb

**Purpose**: Exploratory Data Analysis of the MedQuAD dataset.

**What You'll Learn**:
- Dataset structure and statistics
- Question and answer length distributions
- Entity type distributions
- Data quality assessment

**Key Sections**:
1. **Data Loading**: Load and inspect MedQuAD CSV
2. **Statistical Analysis**: Question/answer lengths, missing values
3. **Visualization**: Histograms, distribution plots
4. **Sample Exploration**: Random Q&A pairs

**Usage**:
```bash
jupyter notebook notebooks/eda_medquad.ipynb
```

**Expected Insights**:
- 16,412 biomedical Q&A pairs
- Average question length: 50-100 characters
- Average answer length: 200-500 characters
- Common medical topics: diabetes, cancer, heart disease

---

### notebooks/entity_processing_demo.ipynb

**Purpose**: Interactive demonstration of entity extraction and normalization.

**What You'll Learn**:
- How SciSpaCy extracts biomedical entities
- How fuzzy matching corrects typos
- Entity normalization process
- Complete extraction pipeline

**Key Sections**:

1. **Part 1: Entity Extraction with SciSpaCy**
   - Load SciSpaCy model
   - Extract entities from questions
   - Identify DISEASE and CHEMICAL types

2. **Part 2: Entity Normalization**
   - Fuzzy matching demonstration
   - Typo correction examples
   - Canonical entity mapping

3. **Part 3: Complete Pipeline**
   - End-to-end entity processing
   - Question → Entities → Normalized

**Usage**:
```bash
jupyter notebook notebooks/entity_processing_demo.ipynb
```

**Example Outputs**:
```python
Input: "What treats diabete?"  # Typo
Extracted: ["diabete"]
Normalized: ["diabetes"]  # Corrected!
```

**Note**: If you encounter `IndexError: list index out of range`, it means no entities were extracted. Use questions with recognized medical terms like "diabetes", "asthma", "metformin".

---

### notebooks/vector_search_demo.ipynb

**Purpose**: Demonstrate ChromaDB vector search and semantic similarity.

**What You'll Learn**:
- How embeddings work
- Semantic search vs keyword search
- ChromaDB query interface
- Similarity scoring

**Key Sections**:

1. **Setup ChromaDB**
   - Connect to vector database
   - Load embedding model
   - Inspect collection

2. **Semantic Search Examples**
   - Query with medical terms
   - View top-K results
   - Examine similarity scores

3. **Comparison: Semantic vs Keyword**
   - Same question, different methods
   - Quality comparison
   - Why semantic search wins

**Usage**:
```bash
jupyter notebook notebooks/vector_search_demo.ipynb
```

**Example**:
```python
Query: "diabetes treatment"
Top Results:
1. "Metformin is first-line therapy..." (score: 0.85)
2. "Insulin therapy for type 1..." (score: 0.82)
3. "Lifestyle modifications include..." (score: 0.79)
```

---

### notebooks/rag_pipeline_demo.ipynb

**Purpose**: Complete RAG (Retrieval-Augmented Generation) pipeline walkthrough.

**What You'll Learn**:
- How RAG combines retrieval + generation
- Context assembly from multiple sources
- Prompt engineering for biomedical QA
- Answer quality evaluation

**Key Sections**:

1. **RAG Components**
   - Vector retrieval
   - Context assembly
   - LLM generation

2. **Step-by-Step Pipeline**
   - Question processing
   - Evidence retrieval
   - Answer generation
   - Citation formatting

3. **Quality Analysis**
   - With vs without RAG
   - Evidence grounding
   - Citation accuracy

**Usage**:
```bash
jupyter notebook notebooks/rag_pipeline_demo.ipynb
```

**Example Flow**:
```
Question: "What is diabetes?"
    ↓
[Retrieve] 5 PubMed articles
    ↓
[Assemble] Context with evidence
    ↓
[Generate] Evidence-based answer
    ↓
Output: "Diabetes is a metabolic disorder... [PMID: 12345]"
```

---

### notebooks/model_evaluation.ipynb

**Purpose**: Evaluate fine-tuned Qwen model performance.

**What You'll Learn**:
- Model evaluation metrics
- Comparison: Base vs Fine-tuned
- Performance on biomedical questions
- Error analysis

**Key Sections**:

1. **Load Models**
   - Base Qwen2.5-1.5B
   - Fine-tuned version
   - Evaluation dataset

2. **Quantitative Metrics**
   - ROUGE scores
   - BLEU scores
   - Exact match
   - F1 scores

3. **Qualitative Analysis**
   - Side-by-side comparisons
   - Medical accuracy
   - Citation quality

4. **Error Analysis**
   - Common failure modes
   - Improvement areas

**Usage**:
```bash
jupyter notebook notebooks/model_evaluation.ipynb
```


### Running Notebooks

**Prerequisites**:
```bash
# Install Jupyter
pip install jupyter notebook

# Or use JupyterLab
pip install jupyterlab
```

**Start Jupyter**:
```bash
# From project root
jupyter notebook notebooks/

# Or JupyterLab
jupyter lab notebooks/
```

**Kernel Setup**:
1. Ensure your virtual environment is activated
2. Install ipykernel: `pip install ipykernel`
3. Register kernel: `python -m ipykernel install --user --name=biographx`
4. Select "biographx" kernel in Jupyter

**Troubleshooting Notebooks**:

1. **"Module not found" errors**
   - Ensure virtual environment is activated
   - Install missing packages: `pip install -r requirements.txt`
   - Restart kernel

2. **"ChromaDB not found"**
   - Build vector database: `python rag/embed_pubmed.py`
   - Or use smaller test collection

3. **"Model not found"**
   - Download SciSpaCy: See Step 2
   - Fine-tuned model: See Step 5

4. **Memory errors**
   - Reduce batch sizes in code
   - Close other applications
   - Use smaller data samples

---

### Learning Path

**Recommended Order**:

1. **Start**: `eda_medquad.ipynb`
   - Understand the data
   - See what questions look like

2. **Next**: `entity_processing_demo.ipynb`
   - Learn entity extraction
   - Understand normalization

3. **Then**: `vector_search_demo.ipynb`
   - See how retrieval works
   - Understand embeddings

4. **After**: `rag_pipeline_demo.ipynb`
   - Complete pipeline
   - See everything together

5. **Finally**: `model_evaluation.ipynb`
   - Evaluate performance
   - Understand improvements

**Time Required**: 2-3 hours to work through all notebooks

---

## Development Guide


### Adding New Agents

**1. Create Agent File**

Create `agents/my_new_agent.py`:
```python
class MyNewAgent:
    def __init__(self):
        # Initialize resources
        pass
    
    def run(self, state):
        # Process state
        # Add new information to state
        state["my_new_data"] = "processed data"
        return state
```

**2. Register in Orchestrator**

Edit `agents/agent_graph_orchestrator.py`:
```python
from agents.my_new_agent import MyNewAgent

class AgentGraphPipeline:
    def __init__(self):
        # ... existing agents ...
        self.my_new_agent = MyNewAgent()
    
    def run(self, question):
        state = {"question": question}
        # ... existing agents ...
        state = self.my_new_agent.run(state)
        # ... remaining agents ...
        return state
```

**3. Test Agent**
```bash
python scripts/run_pipeline.py
```

### Modifying the Pipeline

**Change Agent Order**:
Edit execution sequence in `agent_graph_orchestrator.py`

**Add Parallel Processing**:
```python
import concurrent.futures

def run(self, question):
    state = {"question": question}
    
    # Sequential
    state = self.question_agent.run(state)
    
    # Parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        wiki_future = executor.submit(self.wikipedia_agent.run, state)
        retriever_future = executor.submit(self.retriever_agent.run, state)
        
        state = wiki_future.result()
        state.update(retriever_future.result())
    
    # Continue sequential
    state = self.qa_model_agent.run(state)
    return state
```

### Custom Model Training

**Use Different Base Model**:

Edit `training/finetune_qa_model.py`:
```python
BASE_MODEL = "meta-llama/Llama-2-7b-hf"  # Instead of Qwen
```

**Use Custom Dataset**:

1. Prepare data in HuggingFace format
2. Update `finetune_qa_model.py`:
   ```python
   train_dataset = load_dataset("json", data_files="my_custom_train.json")
   ```

**Adjust Training Parameters**:
```python
training_args = TrainingArguments(
    num_train_epochs=3,           # More epochs
    learning_rate=1e-4,           # Lower learning rate
    per_device_train_batch_size=2 # Larger batch
)
```

### Testing

**Unit Tests**:
```bash
pytest tests/
```

**Integration Tests**:
```bash
python scripts/run_pipeline.py
# Test with various questions
```

**Performance Testing**:
```python
import time

questions = [
    "What is diabetes?",
    "How does metformin work?",
    # ... more questions
]

for q in questions:
    start = time.time()
    result = pipeline.run(q)
    elapsed = time.time() - start
    print(f"Question: {q}")
    print(f"Time: {elapsed:.2f}s")
```

---

## Conclusion

This implementation guide provides a complete walkthrough of the BioGraphX system. You should now be able to:

- Understand the architecture and data flow
- Set up the system from scratch
- Deploy using Docker
- Modify and extend the system
- Troubleshoot common issues

For additional help, refer to the main [README.md](README.md) or open an issue in the repository.

---

**Happy Building!**
