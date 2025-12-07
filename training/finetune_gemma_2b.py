#!/usr/bin/env python3
"""
Gemma-2B-IT Fine-Tuning Script for MedQuAD Biomedical QA
Optimized for Mac (MPS) with LoRA for efficient training
Expected ROUGE-1: 0.48-0.52 (vs Qwen 1.5B: 0.42-0.45)
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
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dotenv import load_dotenv

# Load environment variables (for HF token)
load_dotenv()

# Mac MPS optimization
torch.backends.mps.allow_fp16_reduced_precision_reduction = True
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Model configuration
MODEL_NAME = "google/gemma-2b-it"
OUT_DIR = "../models/fine_tuned/gemma_2b_medquad"

# Get HF token from environment
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

def format_chat_gemma(example):
    """
    Format data for Gemma's chat template
    Gemma uses: <start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>
    """
    return (
        "<start_of_turn>user\n"
        + example["input"]
        + "<end_of_turn>\n"
        "<start_of_turn>model\n"
        + example["output"]
        + "<end_of_turn>"
    )

def main():
    print("="*60)
    print("Gemma-2B-IT Fine-Tuning for MedQuAD")
    print("="*60)
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {OUT_DIR}\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    
    # Gemma-specific: Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    print("Loading Gemma-2B-IT model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for Mac MPS
        device_map="auto",
        trust_remote_code=True,  # Required for Gemma
        token=HF_TOKEN
    )
    
    # Disable cache for training
    model.config.use_cache = False
    
    # OPTIMIZED LoRA configuration for speed
    print("\nConfiguring OPTIMIZED LoRA...")
    lora_cfg = LoraConfig(
        r=16,  # Reduced from 32 for faster training
        lora_alpha=32,  # 2x rank for optimal scaling
        lora_dropout=0.05,
        target_modules=[
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj"
            # Removed gate_proj, up_proj, down_proj for speed
        ],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    
    # Load datasets
    print("\nLoading MedQuAD datasets...")
    train_ds = load_dataset(
        "json",
        data_files="../data/processed/evaluation/medquad_train.json"
    )["train"]
    
    val_ds = load_dataset(
        "json",
        data_files="../data/processed/evaluation/medquad_val.json"
    )["train"]
    
    print(f"Train samples: {len(train_ds)} (100% of data)")
    print(f"Val samples: {len(val_ds)}")
    
    # Format datasets for Gemma
    print("\nFormatting datasets for Gemma chat template...")
    train_ds = train_ds.map(lambda x: {"text": format_chat_gemma(x)})
    val_ds = val_ds.map(lambda x: {"text": format_chat_gemma(x)})
    
    # Tokenization function
    def tokenize(batch):
        tok = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=384  # Optimized for speed (vs 512)
        )
        tok["labels"] = tok["input_ids"]
        return tok
    
    # Tokenize datasets
    print("Tokenizing datasets...")
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
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=384,
        return_tensors="pt"
    )
    
    # Training arguments - optimized for Mac MPS
    print("\nConfiguring training arguments...")
    args = TrainingArguments(
        output_dir=OUT_DIR,
        
        # Batch size (optimized for Mac memory)
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Effective batch size = 8
        
        # Learning rate (slightly lower for 2B model)
        learning_rate=2e-4,  # Slightly higher for faster convergence
        
        # Training duration - OPTIMIZED TO 1 EPOCH
        num_train_epochs=1,  # Reduced from 2 for speed
        
        # Logging and evaluation
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,  # Less frequent for speed
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # Optimization
        optim="adamw_torch",
        warmup_steps=50,  # Reduced warmup
        weight_decay=0.01,
        max_grad_norm=0.3,
        
        # Memory optimization
        gradient_checkpointing=False,  # Disable for MPS compatibility
        fp16=False,  # Use bfloat16 instead (set in model loading)
        
        # Misc
        remove_unused_columns=False,
        report_to=None,
        save_total_limit=2  # Keep only best 2 checkpoints
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Calculate expected time
    total_steps = len(train_ds) // (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    print("\n" + "="*60)
    print("OPTIMIZED TRAINING - 100% DATA")
    print("="*60)
    print(f"Total steps: {total_steps}")
    print(f"Expected time: ~6 hours")
    print("\nOptimizations:")
    print("  ✓ LoRA rank: 16 (vs 32)")
    print("  ✓ Sequence: 384 (vs 512)")
    print("  ✓ Epochs: 1 (fast convergence)")
    print("  ✓ Modules: 4 (core attention)")
    print("  ✓ Data: 100% (14,724 samples)")
    print("\nTarget ROUGE-1: ≥0.50")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Save model
    print("\n" + "="*60)
    print("Saving fine-tuned model...")
    print("="*60)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Save LoRA adapter
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    
    # Merge LoRA weights with base model for easier inference
    print("\nMerging LoRA weights with base model...")
    merged_dir = f"{OUT_DIR}_merged"
    os.makedirs(merged_dir, exist_ok=True)
    
    # Merge and save
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    
    print("\n" + "="*60)
    print("✓ Fine-tuning complete!")
    print("="*60)
    print(f"LoRA adapter saved to: {OUT_DIR}")
    print(f"Merged model saved to: {merged_dir}")
    print("\nNext steps:")
    print("1. Run evaluation script to measure ROUGE scores")
    print("2. Expected ROUGE-1: 0.48-0.52 (vs Qwen: 0.42-0.45)")
    print("="*60)

if __name__ == "__main__":
    main()
