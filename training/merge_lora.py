import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
LORA = "models/fine_tuned/qwen25_1_5b_medquad"
OUT  = "models/fine_tuned/qwen25_1_5b_medquad_merged"

print("Loading base model…")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float32
)

print("Loading LoRA adapter…")
model = PeftModel.from_pretrained(base_model, LORA)

print("Merging LoRA weights…")
merged = model.merge_and_unload()

print("Saving merged model…")
merged.save_pretrained(OUT)
AutoTokenizer.from_pretrained(BASE).save_pretrained(OUT)

print("\nMerged model saved to:", OUT)