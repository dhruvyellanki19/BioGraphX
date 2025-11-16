#!/usr/bin/env python3
"""
FINAL — BioGraphX LoRA Fine-Tuning Script
Model: Qwen2.5-3B-Instruct
Status: CLEAN + STABLE + FULLY DEBUGGED
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model


# ================================
#  DEVICE & TORCH SETTINGS
# ================================
torch.backends.mps.allow_fp16_reduced_precision_reduction = True
device = "mps" if torch.backends.mps.is_available() else "cpu"

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUT_DIR = "models/fine_tuned/qwen25_medquad/"


# ================================
#  FORMAT: MedQuAD → Qwen ChatML
# ================================
def format_chat(example):
    """Convert Q/A pair into ChatML format required by Qwen."""
    return (
        "<|im_start|>user\n"
        + example["input"]
        + "\n<|im_end|>\n"
        "<|im_start|>assistant\n"
        + example["output"]
        + "\n<|im_end|>"
    )


# ================================
#  MAIN TRAINING PIPELINE
# ================================
def main():

    print(f"\n🧬 Loading Qwen2.5-3B-Instruct...\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,   # Best for MPS stability
        device_map="auto",
    )

    model.config.use_cache = False  # Required for training

    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ============================
    #       LORA CONFIG
    # ============================
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen correct
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)

    print("\n📊 Trainable Parameters:")
    model.print_trainable_parameters()


    # ============================
    #      LOAD DATASETS
    # ============================
    print("\n📁 Loading MedQuAD dataset...")

    train_ds = load_dataset(
        "json",
        data_files="data/processed/evaluation/medquad_train.json"
    )["train"]

    val_ds = load_dataset(
        "json",
        data_files="data/processed/evaluation/medquad_val.json"
    )["train"]

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")


    # Convert to ChatML
    train_ds = train_ds.map(lambda x: {"text": format_chat(x)})
    val_ds   = val_ds.map(lambda x: {"text": format_chat(x)})


    # ============================
    #      TOKENIZATION
    # ============================
    def tokenize(batch):
        tok = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        tok["labels"] = tok["input_ids"]
        return tok


    # REMOVE all non-tensor columns to avoid padding errors
    cols_to_remove = ["text", "input", "output", "id", "url", "title"]

    train_ds = train_ds.map(
        tokenize,
        batched=True,
        remove_columns=[c for c in cols_to_remove if c in train_ds.column_names],
    )

    val_ds = val_ds.map(
        tokenize,
        batched=True,
        remove_columns=[c for c in cols_to_remove if c in val_ds.column_names],
    )


    # ============================
    #      DATA COLLATOR
    # ============================
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=512,
        return_tensors="pt",
    )


    # ============================
    #    TRAINING CONFIG (MPS)
    # ============================
    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,         # Keep small for Mac
        learning_rate=2e-4,
        logging_steps=50,
        save_strategy="epoch",
        optim="adamw_torch",
        gradient_checkpointing=False,
        fp16=False,                 # MUST be False for MPS
        bf16=False,
        remove_unused_columns=False,
        report_to=None,
        max_grad_norm=0.3,
    )


    # ============================
    #         TRAINER
    # ============================
    print("\n🚀 Starting Qwen2.5-3B LoRA fine-tuning...\n")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()


    # ============================
    #         SAVE MODEL
    # ============================
    print("\n💾 Saving fine-tuned model...")
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    print(f"✅ Done! Model saved at: {OUT_DIR}\n")


# ================================
#           ENTRYPOINT
# ================================
if __name__ == "__main__":
    main()