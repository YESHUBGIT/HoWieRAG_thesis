import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

from howie_rag.core.schemas import Chunk
from howie_rag.retrieval.base import BaseRetriever, RetrievalMatch
from howie_rag.retrieval.common import tokenize, validate_top_k


FIELD_NAMES = ("main_text", "title", "table_text", "pre_text", "post_text", "metadata_text")


def _safe_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _field_texts(chunk: Chunk) -> Dict[str, str]:
    metadata = chunk.metadata
    metadata_parts = [
        _safe_text(metadata.get("source_file")),
        _safe_text(metadata.get("source_entity")),
        _safe_text(metadata.get("source_year")),
        _safe_text(metadata.get("source_page_number")),
        _safe_text(metadata.get("source_domain")),
        _safe_text(metadata.get("source_subset")),
        _safe_text(metadata.get("file_name")),
        _safe_text(metadata.get("company_name")),
        _safe_text(metadata.get("company_symbol")),
        _safe_text(metadata.get("report_year")),
        _safe_text(metadata.get("page_number")),
        _safe_text(metadata.get("company_sector")),
        _safe_text(metadata.get("company_industry")),
        _safe_text(metadata.get("domain")),
        _safe_text(metadata.get("subset")),
    ]
    return {
        "main_text": chunk.text,
        "title": _safe_text(metadata.get("source_title")) or _safe_text(metadata.get("title")),
        "table_text": _safe_text(metadata.get("table")),
        "pre_text": _safe_text(metadata.get("pre_text")),
        "post_text": _safe_text(metadata.get("post_text")),
        "metadata_text": " ".join(part for part in metadata_parts if part),
    }


def _has_numeric_cues(query: str) -> bool:
    lowered = query.lower()
    numeric_terms = (
        "percentage",
        "percent",
        "ratio",
        "difference",
        "change",
        "increase",
        "decrease",
        "total",
        "sum",
        "average",
        "revenue",
        "income",
        "expense",
        "margin",
        "balance",
        "debt",
        "shares",
    )
    return any(term in lowered for term in numeric_terms) or any(char.isdigit() for char in lowered)


def _has_source_cues(query: str) -> bool:
    lowered = query.lower()
    source_terms = ("where", "which page", "which section", "which source", "according to", "reported in")
    return any(term in lowered for term in source_terms)


def _has_narrative_cues(query: str) -> bool:
    lowered = query.lower()
    narrative_terms = ("summary", "summarize", "explain", "why", "how did", "interpret", "limitation", "method")
    return any(term in lowered for term in narrative_terms)


class FieldBM25Retriever(BaseRetriever):
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._prepared_chunks_key: Optional[Tuple[int, int]] = None
        self._field_tokens_by_name: Dict[str, List[List[str]]] = {}
        self._field_document_frequencies: Dict[str, Counter] = {}
        self._field_average_lengths: Dict[str, float] = {}

    def _prepare_chunks(self, chunks: List[Chunk]) -> None:
        chunks_key = (id(chunks), len(chunks))
        if self._prepared_chunks_key == chunks_key:
            return

        self._field_tokens_by_name = {field_name: [] for field_name in FIELD_NAMES}
        for chunk in chunks:
            field_texts = _field_texts(chunk)
            for field_name in FIELD_NAMES:
                self._field_tokens_by_name[field_name].append(tokenize(field_texts[field_name]))

        self._field_document_frequencies = {}
        self._field_average_lengths = {}
        for field_name in FIELD_NAMES:
            token_lists = self._field_tokens_by_name[field_name]
            document_frequencies = Counter()
            for tokens in token_lists:
                for token in set(tokens):
                    document_frequencies[token] += 1
            self._field_document_frequencies[field_name] = document_frequencies
            self._field_average_lengths[field_name] = (
                sum(len(tokens) for tokens in token_lists) / len(token_lists) if token_lists else 0.0
            )

        self._prepared_chunks_key = chunks_key

    def _score_document(
        self,
        query_tokens: List[str],
        document_tokens: List[str],
        document_frequencies: Counter,
        average_document_length: float,
        total_documents: int,
    ) -> float:
        if not document_tokens:
            return 0.0

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
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * (document_length / average_document_length)
            )
            score += inverse_document_frequency * (term_frequency * (self.k1 + 1) / denominator)

        return score

    def _field_weights(self, query: str, chunk: Chunk) -> Dict[str, float]:
        metadata = chunk.metadata
        chunk_type = metadata.get("chunk_type")
        has_table_like_content = bool(metadata.get("has_table_like_content"))

        weights = {
            "main_text": 1.0,
            "title": 0.35,
            "table_text": 0.6,
            "pre_text": 0.35,
            "post_text": 0.35,
            "metadata_text": 0.2,
        }

        if chunk_type == "table" or has_table_like_content:
            weights["table_text"] += 0.9
            weights["main_text"] += 0.15
        elif chunk_type == "narrative":
            weights["pre_text"] += 0.35
            weights["post_text"] += 0.35

        if _has_numeric_cues(query):
            weights["table_text"] += 0.5
            weights["main_text"] += 0.1

        if _has_source_cues(query):
            weights["title"] += 0.35
            weights["metadata_text"] += 0.35

        if _has_narrative_cues(query):
            weights["pre_text"] += 0.25
            weights["post_text"] += 0.25

        return weights

    def retrieve(self, query: str, chunks: List[Chunk], top_k: int = 3) -> List[RetrievalMatch]:
        validate_top_k(top_k)
        if not chunks:
            return []

        self._prepare_chunks(chunks)
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        total_documents = len(chunks)
        matches: List[RetrievalMatch] = []
        for index, chunk in enumerate(chunks):
            weights = self._field_weights(query, chunk)
            score = 0.0
            for field_name in FIELD_NAMES:
                field_weight = weights[field_name]
                if field_weight <= 0:
                    continue
                field_tokens = self._field_tokens_by_name[field_name][index]
                field_score = self._score_document(
                    query_tokens=query_tokens,
                    document_tokens=field_tokens,
                    document_frequencies=self._field_document_frequencies[field_name],
                    average_document_length=self._field_average_lengths[field_name],
                    total_documents=total_documents,
                )
                score += field_weight * field_score

            if score > 0:
                matches.append(RetrievalMatch(chunk=chunk, score=float(score)))

        matches.sort(key=lambda match: match.score, reverse=True)
        return matches[:top_k]
