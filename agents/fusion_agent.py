#!/usr/bin/env python3
# agents/fusion_agent.py

def fusion_agent(state: dict) -> dict:
    """
    Fuse the user question, detected diseases, graph neighbors, and curated evidence
    into a concise prompt for the LLM. Evidence is cited using QIDs (MedQuAD) or
    PMIDs (PubMed sentences).
    """
    query = state.get("query", "").strip()
    diseases = state.get("diseases") or []
    graph_results = state.get("graph_results") or []
    evidence = state.get("evidence") or []

    lines = []
    lines.append("You are a biomedical assistant. Use only the evidence and graph cues to answer.")
    lines.append("Write one paragraph, answer the user question directly, and cite sources inline as [QID]/[PMID].")
    lines.append("Focus on complications/outcomes, not definitions or diagnostics. Avoid lists; write flowing text.")
    lines.append("If evidence is thin, say that briefly and give the most supported complications without speculation.")
    lines.append("")
    lines.append(f"User question: {query}")
    if diseases:
        lines.append(f"Detected diseases: {', '.join(diseases)}")

    if graph_results:
        top_graph = ", ".join(f"{d} (w={w})" for d, w in graph_results[:6])
        lines.append(f"Graph neighbors: {top_graph}")

    if evidence:
        lines.append("")
        lines.append("Evidence:")
        for ev in evidence:
            snippet = ev.get("snippet", "").replace("\n", " ").strip()
            citation = ev.get("citation", "UNK")
            src = ev.get("source", "")
            lines.append(f"- [{citation}] ({src}) {snippet}")
    else:
        lines.append("")
        lines.append("Evidence: none retrieved. Use general clinical knowledge with caution.")

    lines.append("")
    lines.append("Answer:")

    fused_context = "\n".join(lines)
    print("[FUSION_AGENT] Built fused context for LLM.")
    return {"fused_context": fused_context}
