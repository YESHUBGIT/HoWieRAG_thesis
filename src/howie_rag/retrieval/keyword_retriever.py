from typing import List

from howie_rag.core.schemas import Chunk
from howie_rag.retrieval.base import BaseRetriever, RetrievalMatch
from howie_rag.retrieval.common import tokenize, validate_top_k


def _score_chunk(query: str, chunk: Chunk) -> int:
    query_tokens = set(tokenize(query))
    chunk_tokens = set(tokenize(chunk.text))
    return len(query_tokens & chunk_tokens)


class KeywordRetriever(BaseRetriever):
    def retrieve(self, query: str, chunks: List[Chunk], top_k: int = 3) -> List[RetrievalMatch]:
        validate_top_k(top_k)

        matches: List[RetrievalMatch] = []
        for chunk in chunks:
            score = _score_chunk(query, chunk)
            if score > 0:
                matches.append(RetrievalMatch(chunk=chunk, score=float(score)))

        matches.sort(key=lambda match: match.score, reverse=True)
        return matches[:top_k]


def retrieve_chunks(query: str, chunks: List[Chunk], top_k: int = 3) -> List[RetrievalMatch]:
    return KeywordRetriever().retrieve(query, chunks, top_k=top_k)
