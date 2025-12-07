import spacy
import re


class QuestionAgent:
    def __init__(self):
        print("[QuestionAgent] Loading SciSpaCy biomedical NER...")

        # Disable unnecessary spaCy pipeline components → huge speed boost
        self.nlp = spacy.load(
            "en_ner_bc5cdr_md",
            disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]
        )

        # Pre-compile cleanup regex
        self.clean_re = re.compile(r"[^a-zA-Z0-9\s\-]+")

    # ---------------------------------------------------------------
    # Fast text cleaner
    # ---------------------------------------------------------------
    def clean(self, text: str) -> str:
        """
        Removes noisy punctuation that confuses NER slightly and
        makes SciSpaCy run faster.
        """
        text = self.clean_re.sub(" ", text)
        return re.sub(r"\s+", " ", text).strip()

    # ---------------------------------------------------------------
    # Main execution
    # ---------------------------------------------------------------
    def run(self, state):
        q = state["question"]

        # 1) Clean for faster NER pass
        clean_q = self.clean(q)

        # 2) Run SciSpaCy NER on original question
        doc = self.nlp(clean_q)
        
        # 3) Extract entities from original NER result
        ents = [ent.text.lower().strip() for ent in doc.ents if ent.label_ in ['DISEASE', 'CHEMICAL']]
        
        # 4) If no medical entities found, try pattern matching for question formats
        if not ents:
            import re
            patterns = [
                r"what\s+is\s+([a-zA-Z][a-zA-Z\s]*?)[\?\s]*$",
                r"what\s+are\s+.*?([a-zA-Z][a-zA-Z\s]{2,}?)[\?\s]*$", 
                r"tell\s+me\s+about\s+([a-zA-Z][a-zA-Z\s]*?)[\?\s]*$",
                r"explain\s+([a-zA-Z][a-zA-Z\s]*?)[\?\s]*$",
                r"describe\s+([a-zA-Z][a-zA-Z\s]*?)[\?\s]*$"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, clean_q.lower())
                if match:
                    potential_entity = match.group(1).strip()
                    # Clean up common non-medical words
                    potential_entity = re.sub(r'^(the|of|symptoms?|causes?|treatment|info|information)\s+', '', potential_entity)
                    potential_entity = re.sub(r'\s+(symptoms?|causes?|treatment|info|information)$', '', potential_entity)
                    
                    if len(potential_entity) > 2 and potential_entity not in ['the', 'of', 'a', 'an', 'is', 'are']:
                        # Test if it's a medical term by running NER on declarative sentence
                        test_sentence = f"{potential_entity} is a medical condition"
                        test_doc = self.nlp(test_sentence)
                        medical_ents = [e.text.lower().strip() for e in test_doc.ents 
                                      if e.label_ in ['DISEASE', 'CHEMICAL'] and e.text.lower().strip() != 'medical condition']
                        
                        if medical_ents:
                            ents.extend(medical_ents)
                        else:
                            # Fallback: add the extracted term anyway if it looks medical-ish
                            medical_keywords = ['asthma', 'diabetes', 'cancer', 'pneumonia', 'covid', 'hypertension', 
                                              'arthritis', 'influenza', 'tuberculosis', 'malaria', 'obesity']
                            if any(keyword in potential_entity.lower() for keyword in medical_keywords):
                                ents.append(potential_entity)
                    break

        # 5) Deduplicate
        ents = list(set(ents))

        print(f"[QuestionAgent] Extracted raw entities: {ents}")

        state["entities"] = ents
        return state
