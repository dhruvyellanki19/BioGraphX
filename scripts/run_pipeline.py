# scripts/run_pipeline.py
# --------------------------------------------------------
# Runs the full BioGraphX Agentic Pipeline end-to-end.
# Automatically fixes Python path so "agents/" package
# can be imported from anywhere.
# --------------------------------------------------------

import sys
import os

# --- FIX PYTHON PATH (Critical) ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from agents.agent_graph_orchestrator import AgentGraphPipeline


def main():
    print("=========================================")
    print(" BioGraphX QA Pipeline")
    print("=========================================\n")

    pipeline = AgentGraphPipeline()

    while True:
        q = input("\nEnter biomedical question (or 'exit'): ").strip()

        if q.lower() == "exit":
            print("\nExiting BioGraphX pipeline.\n")
            break

        if not q:
            print("Please enter a valid question.")
            continue

        print("\n[Pipeline] Processing...\n")

        try:
            state = pipeline.run(q)
            answer = state.get("answer", "").strip()

            print("-----------------------------------------")
            print("               FINAL ANSWER")
            print("-----------------------------------------")
            print(answer if answer else "No answer produced.")
            print("-----------------------------------------\n")

        except Exception as e:
            print("\n ERROR during pipeline execution:")
            print(e)
            print("-----------------------------------------\n")


if __name__ == "__main__":
    main()
