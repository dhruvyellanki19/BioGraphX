"""
Ultra-Optimized PubMedQA Evaluation Script for BioGraphX
--------------------------------------------------------
Optimizations:
- Pipeline loaded once (SciSpaCy, Chroma, Qwen, Neo4j)
- Metrics loaded once
- Minimal per-sample overhead
- Deterministic generation (do_sample=False)
- Automatic memory cleanup every 20 iterations
- Clean logging + structured failures
"""

import sys
import os
import gc
import pandas as pd
from tqdm import tqdm
import numpy as np

# ---------------------------------------------------------
# FIX PYTHON PATH (Critical)
# ---------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from agents.agent_graph_orchestrator import AgentGraphPipeline
from evaluate import load as load_metric


# ---------------------------------------------------------
# Load metrics once (fast)
# ---------------------------------------------------------
rouge_metric = load_metric("rouge")
meteor_metric = load_metric("meteor")
bertscore_metric = load_metric("bertscore")


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def token_metrics(pred, gold):
    """Token-level precision/recall/F1."""
    p_tokens = pred.lower().split()
    g_tokens = gold.lower().split()

    if not p_tokens or not g_tokens:
        return 0.0, 0.0, 0.0

    p_set, g_set = set(p_tokens), set(g_tokens)

    tp = len(p_set & g_set)
    fp = len(p_set - g_set)
    fn = len(g_set - p_set)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)

    return precision, recall, f1


def exact_match(pred, gold):
    return int(pred.strip().lower() == gold.strip().lower())


# ---------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------

def main():
    print("\n=== BioGraphX — Optimized Biomedical QA Evaluation ===\n")

    df = pd.read_csv("data/processed/pubmedqa_clean.csv").head(200)

    print("\n[Pipeline] Initializing full pipeline ONCE...\n")
    pipeline = AgentGraphPipeline()
    print("[Pipeline] Ready.\n")

    preds, refs = [], []
    EMs, Ps, Rs, F1s = [], [], [], []

    print("Evaluating model...\n")

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        q = row["question"]
        gold = row["long_answer"]

        try:
            out = pipeline.run(q)
            pred = out.get("answer", "")

        except Exception as e:
            print(f"\n[ERROR @sample {i}]: {e}")
            pred = ""

        preds.append(pred)
        refs.append(gold)

        # Token metrics
        EMs.append(exact_match(pred, gold))
        p, r, f1 = token_metrics(pred, gold)
        Ps.append(p)
        Rs.append(r)
        F1s.append(f1)

        # Memory cleanup every 20 samples (critical on MPS)
        if i % 20 == 0:
            gc.collect()

    # ---------------------------------------------------------
    # HuggingFace Metrics (computed once)
    # ---------------------------------------------------------
    rouge_scores = rouge_metric.compute(predictions=preds, references=refs)
    meteor_scores = meteor_metric.compute(predictions=preds, references=refs)
    bert_scores = bertscore_metric.compute(predictions=preds, references=refs, lang="en")

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------
    print("\n=== FINAL METRICS ===\n")
    print(f"Exact Match:         {np.mean(EMs):.4f}")
    print(f"Token Precision:     {np.mean(Ps):.4f}")
    print(f"Token Recall:        {np.mean(Rs):.4f}")
    print(f"Token F1:            {np.mean(F1s):.4f}\n")

    print(f"ROUGE-1:             {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-L:             {rouge_scores['rougeL']:.4f}\n")
    print(f"METEOR:              {meteor_scores['meteor']:.4f}\n")
    print(f"BERTScore F1:        {np.mean(bert_scores['f1']):.4f}")

    print("\nEvaluation Complete.\n")


if __name__ == "__main__":
    main()
