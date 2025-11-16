REACT_QA_TEMPLATE = """
You are a biomedical reasoning assistant. 
Use explicit step-by-step thinking following the ReAct (Reason+Act) framework.

## OBJECTIVE
Answer biomedical questions clearly, factually, and concisely.

## CONTEXT
GRAPH CONTEXT:
{graph_context}

EVIDENCE SENTENCES:
{evidence_block}

## USER QUESTION
{question}

---

## REASONING (WRITE THIS FIRST)
Think step-by-step:
1. Identify relevant biomedical entities.
2. Summarize evidence sentences.
3. Map evidence to question intent.
4. Decide best answer supported by citations.

Write your reasoning here:
<reasoning>
"""

REACT_FINAL_ANSWER = """
</reasoning>

## FINAL ANSWER (DO NOT REPEAT REASONING)
Give a short answer supported directly by the evidence.
Cite PMIDs inline.

Answer:
"""

def build_prompt(question, graph_context, evidence):
    evidence_block = "\n".join(
        [f"- PMID {e['pmid']}: {e['sentence']}" for e in evidence]
    )

    return REACT_QA_TEMPLATE.format(
        graph_context=graph_context,
        evidence_block=evidence_block,
        question=question
    ) + REACT_FINAL_ANSWER
