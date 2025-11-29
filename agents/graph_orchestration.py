#!/usr/bin/env python3
# agents/graph_orchestration.py

from langgraph.graph import StateGraph, END

from agents.query_router_agent import query_router_agent
from agents.ner_agent import ner_agent
from agents.graph_agent import graph_agent
from agents.vector_agent import vector_agent
from agents.evidence_agent import evidence_agent
from agents.fusion_agent import fusion_agent
from agents.answer_agent import answer_agent


def preserve_state(fn):
    def wrapper(state):
        updates = fn(state) or {}
        return {**state, **updates}
    return wrapper


def build_graph():
    workflow = StateGraph(dict)

    workflow.add_node("query_router", preserve_state(query_router_agent))
    workflow.add_node("ner_agent", preserve_state(ner_agent))
    workflow.add_node("graph_agent", preserve_state(graph_agent))
    workflow.add_node("vector_agent", preserve_state(vector_agent))
    workflow.add_node("evidence_agent", preserve_state(evidence_agent))
    workflow.add_node("fusion_agent", preserve_state(fusion_agent))
    workflow.add_node("answer_agent", preserve_state(answer_agent))

    workflow.add_edge("query_router", "ner_agent")
    workflow.add_edge("ner_agent", "graph_agent")
    workflow.add_edge("graph_agent", "vector_agent")
    workflow.add_edge("vector_agent", "evidence_agent")
    workflow.add_edge("evidence_agent", "fusion_agent")
    workflow.add_edge("fusion_agent", "answer_agent")
    workflow.add_edge("answer_agent", END)

    workflow.set_entry_point("query_router")

    return workflow.compile()
