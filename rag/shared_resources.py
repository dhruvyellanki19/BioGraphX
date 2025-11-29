#!/usr/bin/env python3
"""
Shared resources for RAG pipeline:
- MedQuAD DataFrame
- PubMedBERT embedder
- Chroma client / collection
- Neo4j driver
- BC5CDR NER model
"""

import os
import ast

import dotenv
import pandas as pd
import chromadb
from chromadb.config import Settings
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import spacy

dotenv.load_dotenv()

# ------------------------------------------------------------------
# CONSTANTS (central config for RAG)
# ------------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

EMBEDDER_NAME = "NeuML/pubmedbert-base-embeddings"
PARQUET_PATH = "data/processed/medquad_with_embeddings.parquet"

CHROMA_DIR = "data/vectorstore"
COLLECTION_NAME = "medquad_combined"

TOP_K = 5

# ------------------------------------------------------------------
# Internal singletons
# ------------------------------------------------------------------
_df = None
_embedder = None
_chroma_client = None
_chroma_collection = None
_neo4j_driver = None
_nlp_bc5 = None


# ------------------------------------------------------------------
# Small helper for the dataframe
# ------------------------------------------------------------------
def _ensure_list(value):
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        if "," in value:
            return [v.strip() for v in value.split(",")]
        return [value]
    return [str(value)]


# ------------------------------------------------------------------
# Public getters
# ------------------------------------------------------------------
def get_medquad_df() -> pd.DataFrame:
    global _df
    if _df is not None:
        return _df

    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"Missing parquet: {PARQUET_PATH}")

    df = pd.read_parquet(PARQUET_PATH)
    df["question_id"] = df["question_id"].astype(str)

    if "diseases" in df.columns:
        df["diseases"] = df["diseases"].apply(_ensure_list)
    else:
        df["diseases"] = [[] for _ in range(len(df))]

    _df = df
    return _df


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print("Loading PubMedBERT sentence embedder...")
        _embedder = SentenceTransformer(EMBEDDER_NAME)
    return _embedder


def get_chroma_client() -> chromadb.Client:
    global _chroma_client
    if _chroma_client is None:
        print("Connecting to Chroma...")
        _chroma_client = chromadb.Client(
            Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR)
        )
    return _chroma_client


def get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = get_chroma_client()
        _chroma_collection = client.get_collection(COLLECTION_NAME)
    return _chroma_collection


def get_neo4j_driver():
    global _neo4j_driver
    if _neo4j_driver is not None:
        return _neo4j_driver

    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        raise ValueError("Missing Neo4j credentials in .env")

    print("Opening Neo4j driver...")
    _neo4j_driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
    )
    return _neo4j_driver


def get_bc5_model():
    global _nlp_bc5
    if _nlp_bc5 is None:
        print("Loading BC5CDR NER model (en_ner_bc5cdr_md)...")
        _nlp_bc5 = spacy.load("en_ner_bc5cdr_md")
    return _nlp_bc5 