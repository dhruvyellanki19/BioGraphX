# BioGraphX - Biomedical AI Question Answering System

**An intelligent multi-agent RAG system for biomedical question answering powered by retrieval-augmented generation, knowledge graphs, and fine-tuned language models.**

**Authors:** Sai Dhruv Yellanki Hanmanthrao, Anvesh Chitturi, Shruthi Raj Gangapuri, Venkata SatySai Maruti Kameshwar Modali

**Team Name: Team Kanya Raashi**

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Documentation](#documentation)

---

## Problem Statement

### The Challenge

Biomedical question answering presents unique challenges:

1. **Complex Terminology**: Medical terms require precise entity extraction and normalization
2. **Multiple Knowledge Sources**: Answers need integration of research literature, knowledge graphs, and general medical knowledge
3. **Evidence Requirements**: Medical answers must be grounded in scientific evidence with proper citations
4. **Domain Expertise**: General-purpose language models lack specialized biomedical knowledge
5. **Entity Relationships**: Understanding connections between diseases, drugs, and treatments requires structured knowledge

### The Gap

Existing biomedical QA systems typically:
- Rely on single knowledge sources (either literature OR knowledge graphs, not both)
- Use general-purpose models without domain-specific fine-tuning
- Lack sophisticated entity extraction and normalization
- Don't provide transparent reasoning or evidence trails
- Are difficult to deploy and scale

---

## Solution Overview

**BioGraphX** addresses these challenges through a sophisticated multi-agent architecture that combines:

- **7 Specialized Agents** working in concert to process questions, extract entities, retrieve evidence, and generate answers
- **Retrieval-Augmented Generation (RAG)** with 298,000+ indexed PubMed documents
- **Fine-tuned Qwen2.5-1.5B Model** trained on 14,724 biomedical Q&A pairs
- **Entity Normalization** using fuzzy matching against 15,723 canonical biomedical terms
- **Wikipedia Integration** for general medical context
- **Interactive Web Interface** for easy access
- **Docker Deployment** for reproducible, scalable deployment

### What Makes BioGraphX Unique

1. **Multi-Agent Orchestration**: Each agent specializes in a specific task (entity extraction, normalization, retrieval, generation)
2. **Dual Knowledge Integration**: Combines vector database retrieval with Wikipedia knowledge
3. **Domain-Specific Fine-Tuning**: Custom-trained model on biomedical literature
4. **Evidence-Based Answers**: All responses include PubMed citations and reasoning explanations
5. **Production-Ready**: Fully containerized with Docker for easy deployment

---

## System Architecture

### High-Level Architecture

```
User Question
     |
     v
[QuestionAgent] --> Extract biomedical entities (SciSpaCy)
     |
     v
[NormalizeAgent] --> Normalize entities (fuzzy matching)
     |
     v
[WikipediaAgent] --> Retrieve general medical knowledge
     |
     v
[RetrieverAgent] --> Search 298K PubMed documents (ChromaDB)
     |
     v
[QAModelAgent] --> Generate answer (Fine-tuned Qwen2.5-1.5B)
     |
     v
[EvidenceAgent] --> Format evidence with citations
     |
     v
[ExplanationAgent] --> Compile final response
     |
     v
Final Answer with Evidence
```

### Core Components

#### 1. Multi-Agent Pipeline (agents/)
- **QuestionAgent**: Biomedical NER using SciSpaCy's `en_ner_bc5cdr_md` model
- **NormalizeAgent**: Entity normalization via RapidFuzz (70% similarity threshold)
- **WikipediaAgent**: Medical article retrieval for general context
- **RetrieverAgent**: Semantic search over PubMed abstracts (ChromaDB + all-mpnet-base-v2)
- **QAModelAgent**: Answer generation using fine-tuned Qwen2.5-1.5B-Instruct
- **EvidenceAgent**: Evidence formatting and citation management
- **ExplanationAgent**: Final response compilation with reasoning

#### 2. RAG Pipeline (rag/)
- **Vector Database**: ChromaDB with 298,152 PubMed sentences
- **Embeddings**: Sentence-Transformers (all-mpnet-base-v2, 768-dimensional)
- **Retrieval**: Top-K semantic similarity search
- **Evidence Integration**: Context-aware answer generation

#### 3. Fine-Tuned Model (training/)
- **Base Model**: Qwen/Qwen2.5-1.5B-Instruct
- **Training Data**: 14,724 MedQuAD biomedical Q&A pairs
- **Method**: LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- **Performance**: Domain-specific biomedical understanding

#### 4. Web Application (app/)
- **Framework**: Flask with CORS support
- **Interface**: ChatGPT-style dark theme UI
- **Features**: Real-time processing, collapsible evidence panels, interactive display
- **API**: RESTful endpoints for question answering

#### 5. Docker Deployment
- **Containerization**: Complete application in Docker
- **Port**: 1919 (host) -> 5000 (container)
- **Data Persistence**: Volume mounts for ChromaDB and models
- **Health Checks**: Automated monitoring

---

## Key Features

### Intelligent Question Processing
- Automatic biomedical entity extraction from natural language questions
- Entity normalization against canonical medical vocabulary
- Context-aware query expansion

### Multi-Source Knowledge Integration
- **PubMed Literature**: 298,152 indexed research abstracts
- **Wikipedia**: General medical knowledge and context
- **Fine-tuned Model**: Domain-specific biomedical understanding

### Evidence-Based Answers
- All answers grounded in retrieved scientific literature
- PMID citations for research papers
- Transparent reasoning explanations
- Source attribution

### Advanced NLP Pipeline
- **SciSpaCy**: Biomedical named entity recognition
- **Sentence Transformers**: Semantic embeddings
- **Qwen2.5-1.5B**: State-of-the-art language model
- **LoRA Fine-Tuning**: Efficient domain adaptation

### Production-Ready Deployment
- Docker containerization for reproducibility
- Automated health checks
- Data persistence
- Scalable architecture
- Easy deployment across platforms

### Interactive Web Interface
- Modern, responsive UI
- Real-time question processing
- Collapsible evidence panels
- Agent analysis visualization
- Wikipedia context display

---

## Technology Stack

### Core Technologies
- **Python 3.10**: Primary programming language
- **PyTorch**: Deep learning framework
- **Transformers**: HuggingFace library for LLMs

### Biomedical NLP
- **SciSpaCy 0.6.2**: Biomedical NER
- **en_ner_bc5cdr_md**: Disease and chemical entity recognition
- **RapidFuzz**: Fuzzy string matching for entity normalization

### Vector Database & Retrieval
- **ChromaDB 1.3.5**: Vector database
- **Sentence-Transformers**: Embedding generation
- **FAISS**: Efficient similarity search

### Language Models
- **Qwen2.5-1.5B-Instruct**: Base model
- **PEFT**: Parameter-efficient fine-tuning
- **LoRA**: Low-rank adaptation

### Web Framework
- **Flask 2.3+**: Web application framework
- **Flask-CORS**: Cross-origin resource sharing

### Data Processing
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **NLTK**: Natural language processing

### Deployment
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration

---

## Quick Start

### Prerequisites

- **Docker** (20.10+) and **Docker Compose** (2.0+)
- **8GB+ RAM** recommended
- **10GB+ disk space** for models and data

### Running with Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd BioGraphX
   ```

2. **Start the application**
   ```bash
   docker-compose up -d
   ```

3. **Access the web interface**
   ```
   http://localhost:1919
   ```

4. **Check health status**
   ```bash
   curl http://localhost:1919/health
   ```

### Running Locally (Development)

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download SciSpaCy model**
   ```bash
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
   ```

4. **Download NLTK data**
   ```bash
   python -m nltk.downloader punkt punkt_tab stopwords
   ```

5. **Run the application**
   ```bash
   cd app
   python run.py
   ```

6. **Access at** `http://localhost:8989`

---

## Project Structure

```
BioGraphX/
├── agents/                          # Multi-agent pipeline
│   ├── __init__.py                 # Agent package initialization
│   ├── agent_graph_orchestrator.py # Main pipeline coordinator
│   ├── question_agent.py           # Entity extraction (SciSpaCy)
│   ├── normalize_agent.py          # Entity normalization
│   ├── wikipedia_agent.py          # Wikipedia retrieval
│   ├── retriever_agent.py          # ChromaDB vector search
│   ├── qa_model_agent.py           # Answer generation (Qwen)
│   ├── evidence_agent.py           # Evidence formatting
│   └── explanation_agent.py        # Final response compilation
│
├── app/                            # Flask web application
│   ├── main.py                     # API endpoints
│   ├── pipeline_wrapper.py         # Pipeline integration
│   ├── run.py                      # Application launcher
│   ├── templates/                  # HTML templates
│   └── static/                     # CSS and JavaScript
│
├── data/                           # Data storage
│   ├── raw/                        # Raw datasets
│   ├── processed/                  # Processed data
│   └── chroma/                     # ChromaDB vector database
│
├── etl/                            # Data processing scripts
│   ├── prepare_pubmed_sentences.py # PubMed data processing
│   └── clean_pubmedqa.py           # Dataset cleaning
│
├── rag/                            # RAG pipeline
│   ├── embed_pubmed.py             # Vector database creation
│   ├── query_pipeline.py           # Retrieval system
│   ├── build_index.py              # Index construction
│   └── test_retriever.py           # Testing utilities
│
├── training/                       # Model training
│   ├── prepare_medquad_dataset.py  # Dataset preparation
│   ├── finetune_qa_model.py        # Qwen fine-tuning
│   ├── merge_lora.py               # LoRA model merging
│   └── eval_pubmedqa.py            # Model evaluation
│
├── models/                         # Trained models
│   └── fine_tuned/                 # Fine-tuned Qwen model
│
├── scripts/                        # Utility scripts
│   ├── run_pipeline.py             # CLI interface
│   └── install_embedding_model.py  # Model installation
│
├── notebooks/                      # Jupyter notebooks
│   └── 01_eda_medquad.ipynb       # Data exploration
│
├── Dockerfile                      # Docker image definition
├── docker-compose.yaml             # Docker orchestration
├── requirements.txt                # Python dependencies
├── start-docker.sh                 # Docker startup script
└── README.md                       # This file
```

---

## Usage

### Web Interface

1. Navigate to `http://localhost:1919`
2. Enter your biomedical question in the chat interface
3. View the AI-generated answer with:
   - Wikipedia context
   - PubMed evidence with citations
   - Reasoning explanation

### Command-Line Interface

```bash
python scripts/run_pipeline.py
```

Then enter your questions interactively.

### API Endpoints

**Health Check**
```bash
GET /health
```

**Ask Question**
```bash
POST /ask
Content-Type: application/json

{
  "question": "What are the symptoms of diabetes?"
}
```

**Response Format**
```json
{
  "answer": "Evidence-based answer...",
  "entities": ["diabetes"],
  "evidence": [
    {
      "pmid": "12345",
      "text": "Research finding..."
    }
  ],
  "wikipedia": "General medical context...",
  "explanation": "Reasoning process...",
  "status": "success"
}
```

---

## Documentation

- **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)**: Detailed file-by-file explanation and step-by-step setup guide
- **[Dockerfile](Dockerfile)**: Docker image configuration
- **[docker-compose.yaml](docker-compose.yaml)**: Container orchestration

---

## Performance

- **Response Time**: 5-8 seconds for complex biomedical questions
- **Vector Search**: Sub-second retrieval from 298K+ documents
- **Model Inference**: 3-4 seconds for answer generation
- **Memory Usage**: 4-6GB RAM (with models loaded)

---

## Data Sources

- **MedQuAD**: 16,412 biomedical Q&A pairs for training
- **PubMed**: 298,152 research abstracts for evidence retrieval
- **Wikipedia**: General medical knowledge articles
- **Entity Mappings**: 15,723 canonical biomedical terms

---

### Results: Baseline vs Fine-Tuned

| Metric | Baseline (Gemma-2B-IT) | Fine-Tuned (Qwen2.5-1.5B) | Improvement |
|--------|------------------------|---------------------------|-------------|
| **ROUGE-1** | 0.2099 | 0.3877 | +84.7% |
| **ROUGE-L** | 0.1050 | 0.2061 | +96.3% |
| **METEOR** | 0.1634 | 0.2834 | +73.4% |
| **Token F1** | 0.2251 | 0.3009 | +33.7% |
| **BERTScore F1** | 0.7598 | 0.8178 | +7.6% |

**Note**: Qwen outperforms Gemma on 10 out of 11 metrics

### Why Qwen2.5-1.5B for BioGraphX?

After comprehensive evaluation, **Qwen2.5-1.5B-Instruct** was selected as the production model for the RAG system over Gemma-2B-IT due to:

**1. Superior Performance**
- 84.7% improvement in ROUGE-1 score
- 96.3% improvement in ROUGE-L score  
- 73.4% improvement in METEOR score
- Better medical accuracy and terminology usage

**2. Better Biomedical Understanding**
- More accurate interpretation of medical concepts
- Improved handling of disease-drug relationships
- Better structured responses with clinical detail

**3. Efficient Fine-Tuning**
- Successfully fine-tuned on 14,724 MedQuAD Q&A pairs
- LoRA adaptation preserved general knowledge while adding domain expertise
- Faster convergence during training

**4. Production Readiness**
- Consistent, high-quality responses
- Better integration with RAG pipeline
- Reliable performance across diverse biomedical questions

**Current System**: The BioGraphX RAG pipeline uses the fine-tuned Qwen2.5-1.5B model for all answer generation, providing evidence-based responses with superior accuracy.

## Model Information

### Fine-Tuned Qwen2.5-1.5B

**Base Model**: Qwen/Qwen2.5-1.5B-Instruct

**Training Configuration**:
- **Method**: LoRA (Low-Rank Adaptation) parameter-efficient fine-tuning
- **Training Data**: 14,724 MedQuAD biomedical Q&A pairs (90% split)
- **Validation Data**: 1,635 pairs (10% split)
- **Total Parameters**: 1.5B
- **Trainable Parameters**: 3.6M (0.12% of full model)

**Training Hyperparameters**:
- **Epochs**: 1
- **Batch Size**: 1 per device
- **Gradient Accumulation Steps**: 8 (effective batch size: 8)
- **Learning Rate**: 2×10⁻⁴
- **Precision**: bfloat16
- **Sequence Length**: 384 tokens
- **Optimizer**: AdamW

**LoRA Configuration**:
- **Rank (r)**: 16
- **Alpha**: 32
- **Dropout**: 0.05
- **Target Modules**: q_proj, k_proj, v_proj, o_proj (attention layers)
- **Bias**: none

**Model Location**: `models/fine_tuned/qwen25_1_5b_medquad_merged/`

### SciSpaCy NER Model
- **Model**: en_ner_bc5cdr_md v0.5.4
- **Entities**: DISEASE, CHEMICAL
- **Framework**: spaCy 3.7.5

---

## Docker Deployment

### Container Details
- **Image**: biographx-biographx-app
- **Port**: 1919 (host) -> 5000 (container)
- **Volumes**:
  - `./data:/app/data` - Data persistence
  - `./models:/app/models` - Model storage
  - `./app:/app/app` - Live code updates

### Management Commands

```bash
# Start application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop application
docker-compose down

# Rebuild image
docker-compose up --build -d

# Check status
docker-compose ps
```

---

## Development

### Adding New Agents

1. Create new agent file in `agents/`
2. Inherit from base agent pattern
3. Implement `run(state)` method
4. Add to `agent_graph_orchestrator.py`

### Modifying the Pipeline

Edit `agents/agent_graph_orchestrator.py` to change agent execution order or add new processing steps.

### Custom Model Training

See `training/finetune_qa_model.py` for fine-tuning your own model on custom datasets.

---

## Troubleshooting

**Container won't start**
- Check Docker is running
- Verify port 1919 is available
- Check logs: `docker-compose logs`

**Out of memory**
- Increase Docker memory limit to 8GB+
- Reduce batch size in model configuration

**No evidence retrieved**
- Verify ChromaDB data exists in `data/chroma/`
- Check collection has documents: See IMPLEMENTATION_GUIDE.md

**Pipeline initialization fails**
- Ensure all models are downloaded
- Check NLTK data is installed
- Verify SciSpaCy model is present

---

## License

This project is for educational and research purposes.

---

## Acknowledgments

- **SciSpaCy**: Biomedical NLP models
- **Qwen**: Base language model
- **ChromaDB**: Vector database
- **HuggingFace**: Transformers library
- **MedQuAD**: Training dataset

---

## Contact

For questions and support, please open an issue in the repository.

---

**Built with advanced AI techniques for biomedical question answering**
