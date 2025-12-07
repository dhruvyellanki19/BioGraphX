import pandas as pd
from rapidfuzz import process, fuzz
from pathlib import Path
import re


class NormalizeAgent:
    def __init__(self, path=None):
        print("[NormalizeAgent] Loading canonical entity mappings...")

        # -------------------------------
        # Load canonical entity file
        # -------------------------------
        if path is None:
            root = Path(__file__).resolve().parents[1]
            path = root / "data" / "processed" / "entity_mappings.csv"

        df = pd.read_csv(path)

        # Pre-clean and store canonical terms (FAST)
        self.canonical = self._prepare_canonical(df)

        print(f"[NormalizeAgent] Loaded {len(self.canonical)} canonical biomedical entities.\n")

    # ---------------------------------------------------------------
    # Prepare canonical vocabulary (massively optimized)
    # ---------------------------------------------------------------
    def _prepare_canonical(self, df):
        raw_terms = (
            df["entity_text"]
            .dropna()
            .astype(str)
            .str.lower()
            .tolist()
        )

        cleaned = []
        for term in raw_terms:
            t = self._clean(term)
            if t:
                cleaned.append(t)

        # Sorting helps RapidFuzz's caching
        return sorted(set(cleaned))

    # ---------------------------------------------------------------
    # Fast cleaner (single regex, pre-compiled)
    # ---------------------------------------------------------------
    _re_strip = re.compile(r"[^a-z0-9\s\-]+")

    def _clean(self, text: str):
        if not isinstance(text, str):
            return None

        t = text.lower().strip()
        t = self._re_strip.sub(" ", t)
        t = re.sub(r"\s+", " ", t).strip()

        if len(t) < 3:
            return None

        if len(t.split()) > 5:
            return None

        return t


    # ---------------------------------------------------------------
    # Main Execution — highly optimized (RapidFuzz shared memory)
    # ---------------------------------------------------------------
    def run(self, state):
        ents = state.get("entities", [])
        normalized = []

        print("[NormalizeAgent] Fuzzy-normalizing entities:")
        print("   Raw extracted entities:", ents)

        # Pre-lowercase entities → reduces match operations
        ents = [e.lower().strip() for e in ents]

        for e in ents:

            # RapidFuzz is fastest with extractOne + WRatio
            match, score, _ = process.extractOne(
                e,
                self.canonical,
                scorer=fuzz.WRatio,
                score_cutoff=72   # slightly stricter → fewer wrong matches
            ) or (e, None, None)

            print(f"   → '{e}' → '{match}' (score={score})")

            normalized.append(match)

        state["normalized_entities"] = normalized
        return state
