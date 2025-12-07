class ExplanationAgent:
    """
    Creates a clean, human-readable explanation block combining:
    - Final model answer
    - Wikipedia general knowledge
    - Retrieved PubMed literature evidence
    This does NOT modify the answer — only enriches metadata for UI/evaluation.
    """

    def run(self, state):

        answer = state.get("answer", "").strip()
        wikipedia_evidence = state.get("wikipedia_evidence", [])
        evidence = state.get("evidence", [])

        # --------------------------------------------------------------
        # Wikipedia Summary
        # --------------------------------------------------------------
        if wikipedia_evidence:
            wiki_lines = [
                f"- **[{wiki['title']}]({wiki['url']})** — {wiki['summary']}"
                for wiki in wikipedia_evidence
            ]
            wiki_summary = "\n".join(wiki_lines)
        else:
            wiki_summary = "_No Wikipedia articles retrieved._"

        # --------------------------------------------------------------
        # Evidence Summary
        # --------------------------------------------------------------
        if evidence:
            ev_lines = [
                f"- **PMID {ev.get('pmid')}** — {ev.get('sentence')}"
                for ev in evidence
            ]
            evidence_summary = "\n".join(ev_lines)
        else:
            evidence_summary = "_No relevant PubMed evidence retrieved._"

        # --------------------------------------------------------------
        # Explanation Metadata (Markdown)
        # --------------------------------------------------------------
        explanation_md = f"""
###  Final Answer
{answer}

---

###  Wikipedia Knowledge (General Medical Information)
{wiki_summary}

---

###  Supporting Evidence (PubMed Research Literature)
{evidence_summary}

---

###  Reasoning Process Summary
The system produced this answer by:
1. **Extracting biomedical entities** using SciSpaCy  
2. **Normalizing entities** using fuzzy matching against canonical vocabulary  
3. **Retrieving Wikipedia articles** for general medical knowledge
4. **Retrieving PubMed evidence** using ChromaDB (MPNet embeddings)  
5. **Combining all evidence** inside the Qwen-based reasoning model  
6. **Generating a grounded biomedical explanation** with citations

This explanation block is included for transparency, debugging, and evaluation.
"""

        state["explanation"] = explanation_md
        return state
