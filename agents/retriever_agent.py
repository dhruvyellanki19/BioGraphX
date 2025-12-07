import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import nltk
import re
import heapq
import hashlib


class RetrieverAgent:
    def __init__(self):
        print("[RetrieverAgent] Loading ChromaDB & embedder...")

        # Exact model used for building embeddings → ensures PERFECT match
        self.embedder = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2"
        )

        # Persistent ChromaDB index
        import os
        # Get project root (assuming agents/retriever_agent.py is one level deep)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        chroma_path = os.path.join(base_dir, "data", "chroma")
        print(f"[RetrieverAgent] Connecting to ChromaDB at: {chroma_path}")

        self.client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Try to get existing collection, create if doesn't exist
        try:
            self.collection = self.client.get_collection("pubmed_index")
            print("[RetrieverAgent] Found existing pubmed_index collection")
        except chromadb.errors.NotFoundError:
            print("[RetrieverAgent] pubmed_index collection not found, creating empty one...")
            self.collection = self.client.create_collection(
                name="pubmed_index",
                metadata={"description": "PubMed abstracts for biomedical QA"}
            )
            print("[RetrieverAgent] Created empty pubmed_index collection")
        except Exception as e:
            print(f"[RetrieverAgent] ChromaDB error: {e}")
            # Fallback to in-memory collection for testing
            print("[RetrieverAgent] Falling back to in-memory collection...")
            self.client = chromadb.Client()
            self.collection = self.client.create_collection(
                name="pubmed_index",
                metadata={"description": "Temporary in-memory collection"}
            )

        # Download tokenizer once
        nltk.download("punkt", quiet=True)

        # A tiny cache: avoids re-running same entity query
        self.cache = {}

    # ---------------------------------------------------------------
    # Text cleanup
    # ---------------------------------------------------------------
    def clean_text(self, text):
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ---------------------------------------------------------------
    # Fast sentence relevance scoring
    # ---------------------------------------------------------------
    def score_sentence(self, sentence, entities):
        s = sentence.lower()
        return sum(e.lower() in s for e in entities)

    # ---------------------------------------------------------------
    # Main Retrieval Logic
    # ---------------------------------------------------------------
    def run(self, state):

        ents = state.get("normalized_entities", [])
        if not ents:
            state["evidence"] = []
            return state

        # -----------------------------------------------------------
        # CACHE CHECK (massive speed boost during evaluation)
        # -----------------------------------------------------------
        cache_key = hashlib.md5(",".join(sorted(ents)).encode()).hexdigest()
        if cache_key in self.cache:
            state["evidence"] = self.cache[cache_key]
            print(f"[RetrieverAgent] (cache hit) Retrieved {len(self.cache[cache_key])} evidence.\n")
            return state

        # -----------------------------------------------------------
        # Step 1: Embed the entity-based query
        # -----------------------------------------------------------
        query_str = ", ".join(sorted(ents))
        q_emb = self.embedder.encode([query_str], normalize_embeddings=True)[0].tolist()

        # -----------------------------------------------------------
        # Step 2: Retrieve candidate chunks from Chroma
        # -----------------------------------------------------------
        try:
            results = self.collection.query(
                query_embeddings=[q_emb],
                n_results=20                # reduced for speed, filtered later
            )

            docs = results["documents"][0]
            metas = results["metadatas"][0]
            
            # Check if we got any results
            if not docs:
                print("[RetrieverAgent]-  No documents found in collection")
                state["evidence"] = self._create_fallback_evidence(ents)
                return state
                
        except Exception as e:
            print(f"[RetrieverAgent] -ChromaDB query error: {e}")
            state["evidence"] = self._create_fallback_evidence(ents)
            return state

        candidate = []

        # -----------------------------------------------------------
        # Step 3: Clean → sentence split → filter → score
        # -----------------------------------------------------------
        for doc, meta in zip(docs, metas):

            clean = self.clean_text(doc)
            sentences = nltk.sent_tokenize(clean)

            # Keep 1–2 short, meaningful sentences
            for s in sentences[:3]:
                if 50 <= len(s) <= 240:
                    score = self.score_sentence(s, ents)
                    candidate.append((score, meta.get("pmid"), s))

        # -----------------------------------------------------------
        # Step 4: Take 10–12 best-scored sentences
        # -----------------------------------------------------------
        best = heapq.nlargest(12, candidate, key=lambda x: x[0])

        # Deduplicate
        evidence = []
        seen = set()

        for score, pmid, sent in best:
            key = (pmid, sent)
            if key not in seen:
                evidence.append({"pmid": pmid, "sentence": sent})
                seen.add(key)

        final = evidence[:10]     # soft limit

        print(f"[RetrieverAgent] Retrieved {len(final)} evidence sentences.\n")

        # Save in cache
        self.cache[cache_key] = final
        state["evidence"] = final
        return state

    def _create_fallback_evidence(self, entities):
        """Create fallback evidence when ChromaDB is empty or unavailable"""
        print("[RetrieverAgent]  Creating fallback evidence...")
        
        # Simple fallback evidence based on entities
        fallback_evidence = []
        for entity in entities[:3]:  # Take first 3 entities
            fallback_evidence.append({
                "pmid": "N/A",
                "sentence": f"General medical information about {entity} may be relevant to this biomedical question."
            })
        
        if not fallback_evidence:
            fallback_evidence = [{
                "pmid": "N/A", 
                "sentence": "No specific evidence available. This is a general biomedical query that may require domain expertise."
            }]
        
        print(f"[RetrieverAgent] Created {len(fallback_evidence)} fallback evidence items")
        return fallback_evidence
