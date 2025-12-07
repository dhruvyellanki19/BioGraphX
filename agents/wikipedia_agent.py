import wikipedia
import re


class WikipediaAgent:
    """
    Retrieves medical information from Wikipedia to complement PubMed evidence.
    Provides general medical knowledge and definitions.
    """

    def __init__(self, max_results=2, max_sentences=3):
        """
        Args:
            max_results: Maximum number of Wikipedia articles to retrieve
            max_sentences: Maximum sentences to extract from each article
        """
        self.max_results = max_results
        self.max_sentences = max_sentences
        
        # Set Wikipedia language to English
        wikipedia.set_lang("en")
        
        print("[WikipediaAgent] Initialized for medical article retrieval\n")

    def _clean_text(self, text):
        """Clean Wikipedia text by removing references and extra whitespace"""
        # Remove reference markers like [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def _extract_sentences(self, text, n=3):
        """Extract first n sentences from text"""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return sentences[:n]

    def run(self, state):
        """
        Retrieve Wikipedia articles based on normalized entities.
        
        Input state:
            - normalized_entities: List of canonical entity names
        
        Output state:
            - wikipedia_evidence: List of dicts with {title, summary, url}
        """
        entities = state.get("normalized_entities", [])
        
        if not entities:
            print("[WikipediaAgent] No entities to search")
            state["wikipedia_evidence"] = []
            return state

        wikipedia_results = []
        
        print(f"[WikipediaAgent] Searching Wikipedia for: {entities}")

        for entity in entities[:self.max_results]:  # Limit to max_results entities
            try:
                # Search Wikipedia
                search_results = wikipedia.search(entity, results=1)
                
                if not search_results:
                    print(f"[WikipediaAgent] No results for: {entity}")
                    continue

                # Get the first result
                page_title = search_results[0]
                
                # Fetch the page
                page = wikipedia.page(page_title, auto_suggest=False)
                
                # Clean and extract summary
                summary = self._clean_text(page.summary)
                sentences = self._extract_sentences(summary, self.max_sentences)
                summary_text = ' '.join(sentences)

                wikipedia_results.append({
                    "entity": entity,
                    "title": page.title,
                    "summary": summary_text,
                    "url": page.url
                })
                
                print(f"[WikipediaAgent] ✓ Found: {page.title}")

            except wikipedia.exceptions.DisambiguationError as e:
                # Multiple meanings - try to pick medical one
                try:
                    medical_options = [opt for opt in e.options if any(
                        term in opt.lower() for term in ['disease', 'medicine', 'medical', 'syndrome', 'disorder']
                    )]
                    
                    if medical_options:
                        page = wikipedia.page(medical_options[0], auto_suggest=False)
                        summary = self._clean_text(page.summary)
                        sentences = self._extract_sentences(summary, self.max_sentences)
                        summary_text = ' '.join(sentences)
                        
                        wikipedia_results.append({
                            "entity": entity,
                            "title": page.title,
                            "summary": summary_text,
                            "url": page.url
                        })
                        print(f"[WikipediaAgent] ✓ Found (disambiguated): {page.title}")
                    else:
                        print(f"[WikipediaAgent] Disambiguation for '{entity}', no medical match")
                except Exception as inner_e:
                    print(f"[WikipediaAgent] Error disambiguating '{entity}': {inner_e}")

            except wikipedia.exceptions.PageError:
                print(f"[WikipediaAgent] Page not found for: {entity}")
            
            except Exception as e:
                print(f"[WikipediaAgent] Error fetching '{entity}': {e}")

        print(f"[WikipediaAgent] Retrieved {len(wikipedia_results)} Wikipedia articles\n")
        
        state["wikipedia_evidence"] = wikipedia_results
        return state
