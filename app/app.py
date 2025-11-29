#!/usr/bin/env python3
import streamlit as st
import subprocess
import sys
import json

st.set_page_config(page_title="BiographX – Medical QA", layout="wide")

st.title("BiographX – Biomedical Question Answering")

st.write(
    "Ask any medical question. The system will run NER, graph search, vector "
    "retrieval, and the LoRA-fine-tuned model to generate an answer."
)

query = st.text_input("Enter your question:")

run_button = st.button("Get Answer")

if run_button and query.strip():
    with st.spinner("Running full BiographX pipeline…"):
        try:
            process = subprocess.Popen(
                [sys.executable, "inference.py", query],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate()

            parsed = None
            for line in reversed(stdout.strip().splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                    break
                except Exception:
                    continue

            if parsed and isinstance(parsed, dict):
                answer_text = parsed.get("answer") or ""
                raw_state = parsed.get("raw_state") or {}

                st.subheader("Answer")
                st.markdown(answer_text)

                evidence = raw_state.get("evidence") or []
                if evidence:
                    st.subheader("Supporting evidence")
                    for ev in evidence:
                        snippet = (ev.get("snippet") or "").strip()
                        source = ev.get("source", "")
                        st.markdown(f"- {snippet} ({source})")
            else:
                st.subheader("Raw Output")
                st.code(stdout)

            if process.returncode != 0 and stderr.strip():
                st.subheader("Errors")
                st.error(stderr)

        except Exception as e:
            st.error(f"Unexpected error: {e}")
