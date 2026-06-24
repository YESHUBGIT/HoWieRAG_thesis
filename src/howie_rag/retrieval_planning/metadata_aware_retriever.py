from typing import List, Optional

from howie_rag.core.schemas import Chunk
from howie_rag.retrieval.base import BaseRetriever, RetrievalMatch
from howie_rag.retrieval_planning.retrieval_plan_schema import RetrievalPlan


def _boost_for_document_type(document_type: Optional[str], plan: RetrievalPlan) -> float:
    if not document_type:
        return 0.0

    boost = 0.0
    if document_type in plan.preferred_document_types:
        boost += 1.0

    if plan.retrieval_mode == "narrative_preferred":
        if document_type == "narrative":
            boost += 0.75
        elif document_type == "mixed":
            boost += 0.4
    elif plan.retrieval_mode == "statistical_preferred":
        if document_type == "statistical":
            boost += 0.75
        elif document_type == "mixed":
            boost += 0.4
    elif plan.retrieval_mode == "broad_hybrid":
        if document_type == "mixed":
            boost += 0.6
        elif document_type in {"narrative", "statistical"}:
            boost += 0.3
    return boost


def _boost_for_metadata(chunk: Chunk, plan: RetrievalPlan) -> float:
    metadata = chunk.metadata
    boost = 0.0

    if metadata.get("has_tables") and (
        plan.detected_intent == "TREND_PATTERN" or plan.retrieval_mode == "statistical_preferred"
    ):
        boost += 0.5

    if plan.retrieval_mode == "source_metadata_preferred":
        if metadata.get("original_source_file") or metadata.get("source_path"):
            boost += 0.2
        if metadata.get("title"):
            boost += 0.1

    return boost


def _boost_for_chunk_structure(chunk: Chunk, plan: RetrievalPlan) -> float:
    metadata = chunk.metadata
    chunk_type = metadata.get("chunk_type")
    table_ratio = float(metadata.get("table_line_ratio") or 0.0)
    has_table_like_content = bool(metadata.get("has_table_like_content"))
    detected_intent = plan.detected_intent

    boost = 0.0

    if plan.retrieval_mode == "statistical_preferred":
        if chunk_type == "table":
            boost += 2.0
        elif chunk_type == "mixed":
            boost += 0.9
        elif chunk_type == "narrative":
            boost -= 0.4
        boost += table_ratio * 1.2

    elif plan.retrieval_mode == "narrative_preferred":
        if chunk_type == "narrative":
            boost += 1.0
        elif chunk_type == "mixed":
            boost += 0.35
        elif chunk_type == "table":
            boost -= 0.5

    elif plan.retrieval_mode == "broad_hybrid":
        if chunk_type == "table":
            boost += 1.0
        elif chunk_type == "mixed":
            boost += 0.6
        else:
            boost += 0.2
        boost += table_ratio * 0.5

    elif plan.retrieval_mode == "source_metadata_preferred":
        if has_table_like_content:
            boost += 1.2
        elif chunk_type == "narrative":
            boost += 0.2

    if detected_intent in {"FACT", "COMPARISON", "TREND_PATTERN", "NAVIGATION"}:
        if has_table_like_content:
            boost += 1.1
        elif chunk_type == "table":
            boost += 0.7

    if detected_intent in {"SUMMARY", "EXPLANATION", "METHOD_CONTEXT", "LIMITATION", "INTERPRETATION"}:
        if chunk_type == "narrative":
            boost += 0.6
        elif chunk_type == "table":
            boost -= 0.3

    return boost


def rerank_matches_with_metadata(matches: List[RetrievalMatch], plan: RetrievalPlan) -> List[RetrievalMatch]:
    reranked: List[RetrievalMatch] = []

    for match in matches:
        document_type = match.chunk.metadata.get("document_type")
        boost = _boost_for_document_type(document_type, plan)
        boost += _boost_for_metadata(match.chunk, plan)
        boost += _boost_for_chunk_structure(match.chunk, plan)
        reranked.append(
            RetrievalMatch(
                chunk=match.chunk,
                score=match.score + boost,
                original_score=match.score,
                adjusted_score=match.score + boost,
            )
        )

    reranked.sort(key=lambda item: item.adjusted_score if item.adjusted_score is not None else item.score, reverse=True)
    return reranked


def retrieve_with_metadata_boost(
    retriever: BaseRetriever,
    query: str,
    chunks: List[Chunk],
    plan: RetrievalPlan,
) -> List[RetrievalMatch]:
    candidate_matches = retriever.retrieve(query, chunks, top_k=plan.candidate_pool_size)
    if not candidate_matches:
        return []

    reranked = rerank_matches_with_metadata(candidate_matches, plan)
    return reranked[: plan.top_k]
