import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


class QAModelAgent:
    def __init__(self):
        print("[QAModelAgent] Loading fine-tuned Qwen...")

        MODEL_DIR = "models/fine_tuned/qwen25_1_5b_medquad_merged"
        
        # Check if local model exists, otherwise use fallback
        if os.path.exists(MODEL_DIR):
            self.model_name = MODEL_DIR
            print(f"[QAModelAgent] Using local model: {MODEL_DIR}")
        else:
            # Fallback to a public model
            self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
            print(f"[QAModelAgent] Local model not found, using fallback: {self.model_name}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Qwen2 best on float16 (MPS supports this)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            # Best precision for stability
            torch.set_float32_matmul_precision("high")
            print("[QAModelAgent] Model loaded successfully!")
            
        except Exception as e:
            print(f"[QAModelAgent] Error loading model: {e}")
            print("[QAModelAgent] Falling back to text-only responses...")
            self.model = None
            self.tokenizer = None

    # ------------------------------------------------------------
    # Main run() method
    # ------------------------------------------------------------
    def run(self, state):
        question = state["question"]
        triples = state.get("graph_triples", [])
        evidence = state.get("evidence", [])

        # If model failed to load, provide a fallback response
        if self.model is None or self.tokenizer is None:
            print("[QAModelAgent] Using fallback text response (model not available)")
            answer = self._create_fallback_answer(question, triples, evidence)
            state["answer"] = answer
            return state

        prompt = self.build_prompt(question, triples, evidence)

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.model.device)

            # IMPORTANT:
            # No top_p, top_k, temperature → Qwen ignores unsupported arguments.
            # Greedy decoding = max accuracy for evaluation.
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=220,
                    do_sample=False,            # deterministic & stable
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Extract only the generated answer (skip the input prompt)
            generated_tokens = output[0][inputs['input_ids'].shape[1]:]
            answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            state["answer"] = answer.strip()
            return state
            
        except Exception as e:
            print(f"[QAModelAgent] Error during generation: {e}")
            answer = self._create_fallback_answer(question, triples, evidence)
            state["answer"] = answer
            return state

    # ------------------------------------------------------------
    # Optimized prompt format
    # ------------------------------------------------------------
    def build_prompt(self, question, triples, evidence):

        # Evidence (PMID sentences)
        if evidence:
            evidence_text = "\n".join(
                f"- {ev['sentence']} (PMID: {ev['pmid']})"
                for ev in evidence
            )
        else:
            evidence_text = "None."

        # Final Qwen-friendly prompt - focused and direct
        return f"""
You are a biomedical expert AI system. Use only the provided evidence sentences to generate a medically accurate, concise answer. Base your response strictly on the given evidence.

### Evidence Sentences:
{evidence_text}

### User Question:
{question}

### Answer:
"""

    def _create_fallback_answer(self, question, triples, evidence):
        """Create a fallback answer when model is not available"""
        print("[QAModelAgent] Generating fallback answer...")
        
        # Extract key information from triples and evidence
        key_entities = set()
        if triples:
            for triple in triples[:3]:  # Use first few triples
                if isinstance(triple, dict):
                    key_entities.add(triple.get('subject', ''))
                    key_entities.add(triple.get('object', ''))
                elif isinstance(triple, (list, tuple)) and len(triple) >= 3:
                    key_entities.add(str(triple[0]))
                    key_entities.add(str(triple[2]))
        
        # Clean up entities
        key_entities = [e.strip() for e in key_entities if e and e.strip()]
        
        # Create basic response
        if key_entities:
            entities_text = ", ".join(key_entities[:5])  # Limit to 5 entities
            answer = f"Based on the available biomedical knowledge graph, this question relates to {entities_text}. "
        else:
            answer = "This is a biomedical question that requires domain expertise. "
        
        if evidence:
            answer += f"There are {len(evidence)} relevant evidence sources available for further analysis. "
        
        answer += "For the most accurate and up-to-date information, please consult current biomedical literature and healthcare professionals."
        
        return answer
