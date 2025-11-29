#!/usr/bin/env python3

import sys
import json
from agents.graph_orchestration import build_graph

graph = build_graph()

def run(query):
    state = {"query": query}
    return graph.invoke(state)

if __name__ == "__main__":
    # Accept the query only via command-line args
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No query provided"}))
        sys.exit(1)

    q = sys.argv[1]

    result = run(q)

    # Some pipelines return "answer", others "final_answer".
    # Make it robust to both.
    answer = result.get("final_answer") or result.get("answer") or ""

    print(json.dumps({
        "answer": answer,
        "raw_state": result
    }))