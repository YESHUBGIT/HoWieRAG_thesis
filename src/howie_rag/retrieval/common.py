import re
from typing import List


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "do",
    "in",
    "is",
    "the",
    "to",
    "we",
    "what",
}


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def validate_top_k(top_k: int) -> None:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
