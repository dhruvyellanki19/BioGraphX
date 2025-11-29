#!/usr/bin/env python3
# agents/evidence_agent.py

def evidence_agent(state: dict) -> dict:
    """
    Select and normalize evidence snippets from vector_results.
    Filters to snippets mentioning the detected diseases when possible.
    Keeps a small mix of MedQuAD (QID) and PubMed (PMID) evidence.
    """
    diseases = [d.lower() for d in (state.get("diseases") or []) if isinstance(d, str)]
    medquad_hits = state.get("vector_results_medquad") or []
    pubmed_hits = state.get("vector_results_pubmed") or []

    comp_terms = [
        "complication",
        "complications",
        "perforat",
        "abscess",
        "peritonitis",
        "sepsis",
        "death",
        "mortality",
        "obstruction",
        "adhesion",
    ]

    def mentions_disease(hit):
        text = (hit.get("combined") or hit.get("question") or "").lower()
        return any(d in text for d in diseases) if diseases else False

    def has_complication_language(hit):
        text = (hit.get("combined") or hit.get("question") or "").lower()
        return any(term in text for term in comp_terms)

    def pick_hits(hits, limit):
        if not hits:
            return []
        matched = [h for h in hits if mentions_disease(h)]
        if not matched:
            matched = hits
        # Prefer entries that mention complications explicitly
        matched_sorted = sorted(
            matched,
            key=lambda h: (has_complication_language(h), h.get("score", 0.0)),
            reverse=True,
        )
        return matched_sorted[:limit]

    selected_medquad = pick_hits(medquad_hits, 5)
    selected_pubmed = pick_hits(pubmed_hits, 5)

    evidence = []
    seen_snippets = set()
    for r in selected_medquad + selected_pubmed:
        snippet = (r.get("combined") or r.get("text") or "").strip().replace("\n", " ")
        if not snippet:
            continue
        if snippet in seen_snippets:
            continue
        seen_snippets.add(snippet)

        source = r.get("source", "medquad")
        qid = str(r.get("qid", "")).strip()
        pubid = str(r.get("pubid", "")).strip()
        citation = f"Q{qid}" if source == "medquad" and qid else (f"PMID{pubid}" if pubid else "UNK")

        evidence.append(
            {
                "source": source,
                "citation": citation,
                "qid": qid,
                "pubid": pubid,
                "snippet": snippet,
            }
        )

    return {"evidence": evidence}
