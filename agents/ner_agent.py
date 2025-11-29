#!/usr/bin/env python3
# agents/ner_agent.py

_BC5_NLP = None
_BC5_LOAD_ERROR = None


def _load_bc5():
    global _BC5_NLP
    global _BC5_LOAD_ERROR
    if _BC5_NLP is not None:
        return _BC5_NLP
    if _BC5_LOAD_ERROR is not None:
        return None

    try:
        import spacy
    except ImportError as exc:
        _BC5_LOAD_ERROR = exc
        print("BC5CDR model unavailable: spaCy not installed.")
        return None

    try:
        print("Loading BC5CDR NER model (en_ner_bc5cdr_md)...")
        _BC5_NLP = spacy.load("en_ner_bc5cdr_md")
        print("BC5CDR model loaded.")
    except Exception as exc:
        _BC5_LOAD_ERROR = exc
        print(f"BC5CDR model unavailable: {exc}")
        _BC5_NLP = None
    return _BC5_NLP


def ner_agent(state: dict) -> dict:
    """
    Extract DISEASE entities from the user query using BC5CDR.
    """
    nlp = _load_bc5()
    text = state.get("query", "")
    if not text.strip():
        print("[NER_AGENT] Empty query; no diseases extracted.")
        return {"diseases": []}

    if nlp is None:
        print("[NER_AGENT] BC5CDR unavailable; skipping disease extraction.")
        return {"diseases": []}

    doc = nlp(text)
    diseases = sorted({
        ent.text.strip().lower()
        for ent in doc.ents
        if ent.label_ == "DISEASE" and ent.text.strip()
    })

    print(f"[NER_AGENT] Diseases in query: {diseases}")
    return {"diseases": diseases}
