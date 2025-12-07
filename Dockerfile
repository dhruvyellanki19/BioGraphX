# BioGraphX Dockerfile
# Multi-stage build for optimized Python Flask application

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (minimal requirements with correct versions)
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install SciSpaCy biomedical NER model (version 0.5.4 compatible with spacy 3.7.5)
RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

# Download NLTK data
RUN python -m nltk.downloader punkt punkt_tab stopwords

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/chroma data/processed models app/static/graph

# Expose Flask port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the Flask application
CMD ["python", "app/main.py"]
