import math
from collections import Counter
from typing import List

from howie_rag.core.schemas import Chunk
from howie_rag.retrieval.base import BaseRetriever, RetrievalMatch
from howie_rag.retrieval.common import tokenize, validate_top_k


class BM25Retriever(BaseRetriever):
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def retrieve(self, query: str, chunks: List[Chunk], top_k: int = 3) -> List[RetrievalMatch]:
        validate_top_k(top_k)
        if not chunks:
            return []

        tokenized_chunks = [tokenize(chunk.text) for chunk in chunks]
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        average_document_length = sum(len(tokens) for tokens in tokenized_chunks) / len(tokenized_chunks)
        document_frequencies = Counter()
        for tokens in tokenized_chunks:
            for token in set(tokens):
                document_frequencies[token] += 1

        scores = [
            self._score_document(
                query_tokens=query_tokens,
                document_tokens=document_tokens,
                document_frequencies=document_frequencies,
                average_document_length=average_document_length,
                total_documents=len(tokenized_chunks),
            )
            for document_tokens in tokenized_chunks
        ]

        matches = [
            RetrievalMatch(chunk=chunk, score=float(score))
            for chunk, score in zip(chunks, scores)
            if score > 0
        ]
        matches.sort(key=lambda match: match.score, reverse=True)
        return matches[:top_k]

    def _score_document(
        self,
        query_tokens: List[str],
        document_tokens: List[str],
        document_frequencies: Counter,
        average_document_length: float,
        total_documents: int,
    ) -> float:
        document_term_frequencies = Counter(document_tokens)
        document_length = len(document_tokens)
        score = 0.0

        for token in query_tokens:
            if token not in document_term_frequencies:
                continue

            term_frequency = document_term_frequencies[token]
            inverse_document_frequency = math.log(
                1 + (total_documents - document_frequencies[token] + 0.5) / (document_frequencies[token] + 0.5)
            )
            numerator = term_frequency * (self.k1 + 1)
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * (document_length / average_document_length)
            )
            score += inverse_document_frequency * (numerator / denominator)

        return score
