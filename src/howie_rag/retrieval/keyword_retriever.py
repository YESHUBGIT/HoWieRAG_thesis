from typing import List, Optional, Tuple

from howie_rag.core.schemas import Chunk
from howie_rag.retrieval.base import BaseRetriever, RetrievalMatch
from howie_rag.retrieval.common import tokenize, validate_top_k


def _score_chunk(query: str, chunk: Chunk) -> int:
    query_tokens = set(tokenize(query))
    chunk_tokens = set(tokenize(chunk.text))
    return len(query_tokens & chunk_tokens)


class KeywordRetriever(BaseRetriever):
    def __init__(self) -> None:
        self._prepared_chunks_key: Optional[Tuple[int, int]] = None
        self._chunk_token_sets: List[set[str]] = []

    def _prepare_chunks(self, chunks: List[Chunk]) -> None:
        chunks_key = (id(chunks), len(chunks))
        if self._prepared_chunks_key == chunks_key:
            return

        self._chunk_token_sets = [set(tokenize(chunk.text)) for chunk in chunks]
        self._prepared_chunks_key = chunks_key

    def retrieve(self, query: str, chunks: List[Chunk], top_k: int = 3) -> List[RetrievalMatch]:
        validate_top_k(top_k)
        self._prepare_chunks(chunks)

        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []

        matches: List[RetrievalMatch] = []
        for chunk, chunk_tokens in zip(chunks, self._chunk_token_sets):
            score = len(query_tokens & chunk_tokens)
            if score > 0:
                matches.append(RetrievalMatch(chunk=chunk, score=float(score)))

        matches.sort(key=lambda match: match.score, reverse=True)
        return matches[:top_k]


def retrieve_chunks(query: str, chunks: List[Chunk], top_k: int = 3) -> List[RetrievalMatch]:
    return KeywordRetriever().retrieve(query, chunks, top_k=top_k)
