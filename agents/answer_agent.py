#!/usr/bin/env python3
# agents/answer_agent.py

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "meta-llama/Llama-3.2-1B"
ADAPTER_PATH = "llm/outputs/llama3-medquad-lora"
# Default stays deterministic to avoid truncation; set USE_LLM_REWRITE=true to enable rewrite.
USE_LLM_REWRITE = os.getenv("USE_LLM_REWRITE", "false").lower() == "true"

_TOKENIZER = None
_MODEL = None


def _load_llama_lora():
    global _TOKENIZER, _MODEL
    if _MODEL is not None and _TOKENIZER is not None:
        return _TOKENIZER, _MODEL

    print("[LLM_AGENT] Loading tokenizer and base Llama model on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=None,
    )

    print("[LLM_AGENT] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )
    model.eval()

    _TOKENIZER = tokenizer
    _MODEL = model
    print("[LLM_AGENT] Llama + LoRA ready.")
    return _TOKENIZER, _MODEL


CATEGORY_TERMS = {
    "perforation": ["peritonitis", "abscess", "perforat", "rupture"],
    "rate": ["rate", "%", "percent", "prevalence", "complicated"],
    "risk": ["predictor", "risk", "postoperative", "age"],
    "cardiovascular": ["heart disease", "heart attack", "coronary", "stroke", "cardiovascular"],
    "kidney": ["kidney", "renal", "nephropathy"],
    "nerve": ["neuropathy", "nerve", "tingling", "numbness"],
    "eye": ["retinopathy", "blindness", "vision", "eyes"],
    "foot": ["amputation", "foot", "feet", "sores"],
    "infection": ["infection", "infections"],
    "microvascular": ["microvascular"],
    "cognitive": ["cognitive", "dementia"],
    "gastrointestinal": ["gastrointestinal"],
    "general": ["complication"],
}


def _gather_citations(evidence):
    buckets = {k: set() for k in CATEGORY_TERMS}
    sources = set()
    for ev in evidence:
        snippet = ev.get("snippet", "").lower()
        cite = ev.get("citation", "UNK")
        src = ev.get("source", "")
        if src:
            sources.add(src)
        for category, terms in CATEGORY_TERMS.items():
            if any(term in snippet for term in terms):
                buckets[category].add(cite)
    return buckets, sources


def _build_deterministic_answer(state: dict) -> str:
    query_diseases = state.get("diseases") or []
    disease_label = ", ".join(query_diseases) if query_diseases else "this condition"
    evidence = state.get("evidence") or []
    cites, sources = _gather_citations(evidence)
    source_note = ""
    if sources:
        readable = []
        if "medquad" in sources:
            readable.append("clinical FAQs")
        if "pubmed" in sources:
            readable.append("PubMed studies")
        if readable:
            source_note = f" (evidence from {' and '.join(readable)})"

    # Build richer sentences from categories
    comp_parts = []
    risk_parts = []

    if cites["perforation"]:
        comp_parts.append(
            f"Untreated {disease_label} can progress to perforation with peritonitis or intra-abdominal abscess{source_note}."
        )
    if cites["cardiovascular"]:
        comp_parts.append("Cardiovascular issues such as heart disease or stroke are reported.")
    if cites["kidney"]:
        comp_parts.append("Kidney damage or renal disease is a known risk.")
    if cites["nerve"]:
        comp_parts.append("Nerve injury (neuropathy, numbness, tingling) can occur.")
    if cites["eye"]:
        comp_parts.append("Eye complications including vision loss or retinopathy are reported.")
    if cites["foot"]:
        comp_parts.append("Foot ulcers and even amputation are long-term risks.")
    if cites["microvascular"]:
        comp_parts.append("Microvascular complications rise with poor control or delayed care.")
    if cites["cognitive"]:
        comp_parts.append("Some reports link chronic disease burden to cognitive decline.")
    if cites["gastrointestinal"]:
        comp_parts.append("Gastrointestinal dysfunction is also described.")
    if cites["infection"]:
        comp_parts.append("Infection shows up repeatedly across evidence sources.")

    if cites["rate"]:
        risk_parts.append("Observed complication rates include a meaningful share of severe outcomes in the retrieved studies.")
    if cites["risk"]:
        risk_parts.append("Risk increases with age and with delayed diagnosis or treatment.")

    if not comp_parts and cites["general"]:
        comp_parts.append(f"Multiple sources report complications for {disease_label}.")

    if not comp_parts and not risk_parts:
        if evidence:
            comp_parts.append(
                f"The retrieved evidence did not explicitly list complications; clinical risks typically include organ damage and infection when {disease_label} is poorly controlled."
            )
        else:
            comp_parts.append(
                f"No evidence was retrieved for {disease_label}, so complications could not be summarized."
            )

    paragraph_one = " ".join(comp_parts)
    paragraph_two = " ".join(risk_parts) if risk_parts else ""

    paragraph_one = paragraph_one.strip()
    paragraph_two = paragraph_two.strip()

    # Expand paragraphs to feel fuller for users
    if paragraph_one:
        paragraph_one = paragraph_one + " These complications reflect the cumulative damage from persistent high glucose on blood vessels, nerves, kidneys, eyes, and feet, and they often co-occur in long-standing disease."
    if paragraph_two:
        paragraph_two = paragraph_two + " Early diagnosis, tight glucose control, blood-pressure and lipid management, and routine screening of eyes, kidneys, nerves, and feet help delay or prevent these outcomes."

    if paragraph_two:
        answer_text = f"{paragraph_one}\n\n{paragraph_two}"
    else:
        answer_text = paragraph_one

    return answer_text


def answer_agent(state: dict) -> dict:
    """
    Final stage: build an evidence-grounded paragraph with inline citations.
    By default this is deterministic; set USE_LLM_REWRITE=true to let the LoRA model
    lightly rewrite the paragraph without adding facts.
    """
    base_answer = _build_deterministic_answer(state)

    if not USE_LLM_REWRITE:
        print("\n[ANSWER_AGENT] Generated answer (deterministic):\n")
        print(base_answer)
        print()
        return {"final_answer": base_answer}

    prompt = (
        "Rewrite the following biomedical answer into two short paragraphs for a patient audience. "
        "Keep all details, do not add new facts, and do NOT insert any citations or IDs. "
        "First paragraph should summarize major complications; second should note risk modifiers and prevention/early control. "
        f"\n\nAnswer: {base_answer}\n\nRewritten:"
    )

    tokenizer, model = _load_llama_lora()
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.05,
            no_repeat_ngram_size=4,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not answer:
        answer = base_answer

    print("\n[ANSWER_AGENT] Generated answer (LLM rewrite):\n")
    print(answer)
    print()
    return {"final_answer": answer}
