#!/usr/bin/env python3

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model

torch.backends.mps.allow_fp16_reduced_precision_reduction = True
device = "mps" if torch.backends.mps.is_available() else "cpu"

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUT_DIR = "models/fine_tuned/qwen25_1_5b_medquad"

def format_chat(example):
    return (
        "<|im_start|>user\n"
        + example["input"]
        + "\n<|im_end|>\n"
        "<|im_start|>assistant\n"
        + example["output"]
        + "\n<|im_end|>"
    )

def main():
    print("Loading Qwen2.5-1.5B-Instruct...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    model.config.use_cache = False

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("Loading dataset...")

    train_ds = load_dataset(
        "json",
        data_files="data/processed/evaluation/medquad_train.json"
    )["train"]

    val_ds = load_dataset(
        "json",
        data_files="data/processed/evaluation/medquad_val.json"
    )["train"]

    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))

    train_ds = train_ds.map(lambda x: {"text": format_chat(x)})
    val_ds = val_ds.map(lambda x: {"text": format_chat(x)})

    def tokenize(batch):
        tok = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=384     # Optimized for speed (2× faster)
        )
        tok["labels"] = tok["input_ids"]
        return tok

    cols_to_remove = ["text", "input", "output", "id", "url", "title"]

    train_ds = train_ds.map(
        tokenize,
        batched=True,
        remove_columns=[c for c in cols_to_remove if c in train_ds.column_names]
    )

    val_ds = val_ds.map(
        tokenize,
        batched=True,
        remove_columns=[c for c in cols_to_remove if c in val_ds.column_names]
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=384,
        return_tensors="pt"
    )

    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=120,
        save_strategy="steps",
        save_steps=120,
        load_best_model_at_end=True,

        optim="adamw_torch",
        gradient_checkpointing=False,
        fp16=False,
        remove_unused_columns=False,
        report_to=None,
        max_grad_norm=0.3
    )

    print("Starting fine-tuning...")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    print("Saving model...")
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    print("Done. Model saved to:", OUT_DIR)

if __name__ == "__main__":
    main()