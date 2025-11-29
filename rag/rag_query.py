#!/usr/bin/env python3
"""
Multi-Agent RAG Query Script (Neo4j + Chroma + Llama LoRA)
----------------------------------------------------------

Pipeline:

User Query
 -> Query Agent (logging / future routing)
 -> NER Agent (BC5CDR DISEASE extraction)
 -> Graph Agent (Neo4j co-occurrence over Disease graph)
 -> Vector Agent (Chroma + PubMedBERT semantic search)
 -> Fusion Agent (combine graph + vector into context)
 -> Answer Agent (Llama-3.2-1B + MedQuAD LoRA)

This script is the CLI entrypoint. It uses LangGraph to orchestrate
the six agents defined in the `agents/` folder.
"""

import sys
from langgraph.graph import StateGraph, END

from agents.query_router_agent import query_router_agent
from agents.ner_agent import ner_agent
from agents.graph_agent import graph_agent
from agents.vector_agent import vector_agent
from agents.fusion_agent import fusion_agent
from agents.answer_agent import answer_agent


def build_graph():
    """
    Build a LangGraph state machine where `state` is just a dict.
    Each node returns a partial update, which is merged into `state`.
    """
    builder = StateGraph(dict)

    builder.add_node("query_agent", query_router_agent)
    builder.add_node("ner_agent", ner_agent)
    builder.add_node("graph_agent", graph_agent)
    builder.add_node("vector_agent", vector_agent)
    builder.add_node("fusion_agent", fusion_agent)
    builder.add_node("answer_agent", answer_agent)

    builder.set_entry_point("query_agent")

    builder.add_edge("query_agent", "ner_agent")
    builder.add_edge("ner_agent", "graph_agent")
    builder.add_edge("graph_agent", "vector_agent")
    builder.add_edge("vector_agent", "fusion_agent")
    builder.add_edge("fusion_agent", "answer_agent")
    builder.add_edge("answer_agent", END)

    graph = builder.compile()
    return graph


def run_rag_multi_agent(query: str) -> dict:
    graph = build_graph()
    initial_state = {"query": query}
    final_state = graph.invoke(initial_state)
    return final_state


def main():
    if len(sys.argv) < 2:
        print("Usage: python rag_query.py \"your medical question\"")
        sys.exit(1)

    user_query = sys.argv[1].strip()
    print(f"\nUSER QUERY: {user_query}\n")

    final_state = run_rag_multi_agent(user_query)

    # For logging / debugging:
    graph_results = final_state.get("graph_results") or []
    vector_results = final_state.get("vector_results") or []
    answer = final_state.get("answer") or ""

    print("======= RETRIEVAL SUMMARY =======")
    if graph_results:
        print("Graph diseases:")
        for d, w in graph_results[:10]:
            print(f"- {d} (weight {w})")
    else:
        print("Graph diseases: none")

    if vector_results:
        print("\nVector top matches (question IDs):")
        print([r["qid"] for r in vector_results])
    else:
        print("\nVector top matches: none")

    print("\n======= FINAL ANSWER =======\n")
    print(answer)
    print("\nDone.")


if __name__ == "__main__":
    main()