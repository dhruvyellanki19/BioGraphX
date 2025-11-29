#!/usr/bin/env python3
# agents/query_router_agent.py

def query_router_agent(state: dict) -> dict:
    """
    Very simple first agent:
    - Logs the incoming query
    - Optionally could route to different flows later (general / specialist)
    """
    query = (state.get("query") or "").strip()
    print(f"[QUERY_AGENT] Received query: {query}")
    return {"query": query}