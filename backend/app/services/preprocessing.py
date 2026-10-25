"""
Text Preprocessing Service — PLACEHOLDER

Phase 2 Implementation Plan:
─────────────────────────────
This module will handle all text preprocessing for the IR pipeline:

  1. Tokenisation       — split raw article text into tokens
  2. Stop-word removal  — filter common English stop-words
  3. Stemming/Lemmatisation — reduce words to their root form
  4. Punctuation cleaning   — strip non-alphanumeric characters
  5. Lowercasing            — normalise casing

Dependencies (to be added in Phase 2):
  - nltk
  - spacy  (optional)

Public API (planned):
  preprocess(text: str) -> List[str]
  preprocess_batch(texts: List[str]) -> List[List[str]]
"""


def preprocess(text: str) -> list[str]:
    """
    [PLACEHOLDER] Tokenise and clean a single document.

    Returns a list of tokens. Full implementation in Phase 2.
    """
    raise NotImplementedError("Text preprocessing will be implemented in Phase 2.")


def preprocess_batch(texts: list[str]) -> list[list[str]]:
    """
    [PLACEHOLDER] Preprocess a batch of documents.

    Full implementation in Phase 2.
    """
    raise NotImplementedError("Batch preprocessing will be implemented in Phase 2.")
