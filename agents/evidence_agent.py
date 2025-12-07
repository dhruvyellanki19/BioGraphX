class EvidenceAgent:
    """
    Converts retrieved evidence sentences into structured summaries
    grouped by PMID. This improves grounding and allows QAModelAgent
    to use clearer, more organized biomedical evidence.
    """

    def run(self, state):

        evidence = state.get("evidence", [])

        if not evidence:
            state["evidence_summary"] = "No evidence retrieved."
            state["evidence_grouped"] = []
            return state

        # ---------------------------------------------------------
        # GROUP SENTENCES BY PMID
        # ---------------------------------------------------------
        grouped = {}
        for ev in evidence:
            pmid = ev.get("pmid")
            sent = ev.get("sentence", "").strip()

            if not pmid or not sent:
                continue

            grouped.setdefault(pmid, [])
            grouped[pmid].append(sent)

        # ---------------------------------------------------------
        # BUILD CLEAN SUMMARIES FOR EACH PMID
        # ---------------------------------------------------------
        summary_blocks = []
        grouped_list = []

        for pmid, sentences in grouped.items():
            top_sentences = sentences[:2]     # keep 1–2 per PMID for clarity
            merged = " ".join(top_sentences)

            block = f"- Evidence from PMID {pmid}: {merged}"
            summary_blocks.append(block)

            grouped_list.append({
                "pmid": pmid,
                "sentences": top_sentences,
                "combined": merged
            })

        # Final Markdown summary (for UI + QAModelAgent)
        final_summary = "\n".join(summary_blocks)

        # Attach to shared state
        state["evidence_summary"] = final_summary
        state["evidence_grouped"] = grouped_list

        return state
