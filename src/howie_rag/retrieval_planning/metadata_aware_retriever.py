import re
from typing import List, Optional

from howie_rag.core.schemas import Chunk
from howie_rag.retrieval.base import BaseRetriever, RetrievalMatch
from howie_rag.retrieval_planning.retrieval_plan_schema import RetrievalPlan


def _allowed_chunk_types(plan: RetrievalPlan) -> set[str]:
    if plan.detected_intent in {"SUMMARY", "EXPLANATION", "METHOD_CONTEXT", "LIMITATION", "INTERPRETATION"}:
        return {"narrative", "mixed"}

    if plan.detected_intent == "TREND_PATTERN":
        return {"table", "mixed"}

    if plan.preferred_chunk_types and plan.detected_intent in {"FOLLOWUP"}:
        allowed = set(plan.preferred_chunk_types)
        if "table" in allowed:
            allowed.add("mixed")
        if "narrative" in allowed:
            allowed.add("mixed")
        return allowed

    if plan.retrieval_mode == "statistical_preferred":
        return {"table", "mixed"} if plan.detected_intent == "TREND_PATTERN" else set()
    if plan.retrieval_mode == "narrative_preferred":
        return {"narrative", "mixed"}
    return set()


def _chunk_matches_plan(chunk: Chunk, plan: RetrievalPlan) -> bool:
    metadata = chunk.metadata
    chunk_type = metadata.get("chunk_type")
    document_type = metadata.get("document_type")
    allowed_chunk_types = _allowed_chunk_types(plan)

    if allowed_chunk_types and chunk_type not in allowed_chunk_types:
        return False

    if plan.preferred_document_types and document_type is not None:
        if document_type not in plan.preferred_document_types and document_type != "mixed":
            return False

    if plan.retrieval_mode == "source_metadata_preferred":
        source_marker = (
            metadata.get("source_file")
            or metadata.get("source_title")
            or metadata.get("original_source_file")
            or metadata.get("source_path")
            or metadata.get("file_name")
        )
        if not source_marker and chunk_type == "table":
            return False

    return True


def _filter_chunks_for_plan(chunks: List[Chunk], plan: RetrievalPlan) -> List[Chunk]:
    filtered_chunks = [chunk for chunk in chunks if _chunk_matches_plan(chunk, plan)]
    if len(filtered_chunks) >= plan.top_k:
        return filtered_chunks
    return chunks


def _query_years(query: str) -> set[str]:
    return set(re.findall(r"\b(?:19|20)\d{2}\b", query))


def _query_tokens(query: str) -> set[str]:
    return set(re.findall(r"\b[a-z0-9][a-z0-9&.'-]*\b", query.lower()))


def _tokenize_source_text(value: object) -> set[str]:
    if not isinstance(value, str):
        return set()
    return set(re.findall(r"\b[a-z0-9][a-z0-9&.'-]*\b", value.lower()))


def _boost_for_source_metadata(chunk: Chunk, plan: RetrievalPlan) -> float:
    metadata = chunk.metadata
    query = plan.original_query
    years = _query_years(query)
    query_tokens = _query_tokens(query)

    source_year = str(metadata.get("source_year") or "").strip()
    source_entity_tokens = _tokenize_source_text(metadata.get("source_entity"))
    source_title_tokens = _tokenize_source_text(metadata.get("source_title"))
    source_file_tokens = _tokenize_source_text(metadata.get("source_file"))
    source_page = str(metadata.get("source_page_number") or "").strip()
    source_title_text = str(metadata.get("source_title") or "")
    source_file_text = str(metadata.get("source_file") or "")

    boost = 0.0

    if years:
        if source_year and source_year in years:
            boost += 1.8
        elif source_year and source_year not in years:
            boost -= 1.8

        if any(year in source_title_text for year in years) or any(year in source_file_text for year in years):
            boost += 0.3

    entity_overlap = len(query_tokens & source_entity_tokens)
    title_overlap = len(query_tokens & source_title_tokens)
    file_overlap = len(query_tokens & source_file_tokens)

    if entity_overlap > 0:
        boost += min(1.2, 0.4 * entity_overlap)
    if title_overlap > 0:
        boost += min(0.8, 0.2 * title_overlap)
    if file_overlap > 0:
        boost += min(0.5, 0.15 * file_overlap)

    if years and entity_overlap > 0 and source_year and source_year not in years:
        boost -= 1.0

    if plan.detected_intent in {"NAVIGATION", "SOURCE_SEEKING"}:
        if source_page:
            boost += 0.2
        if title_overlap > 0 or file_overlap > 0:
            boost += 0.4

    return boost


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
        if metadata.get("source_file") or metadata.get("original_source_file") or metadata.get("source_path"):
            boost += 0.2
        if metadata.get("source_title") or metadata.get("title"):
            boost += 0.1

    return boost


def _boost_for_chunk_structure(chunk: Chunk, plan: RetrievalPlan) -> float:
    metadata = chunk.metadata
    chunk_type = metadata.get("chunk_type")
    source_section = metadata.get("source_section")
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
        if source_section == "table":
            boost += 0.4

    if detected_intent in {"SUMMARY", "EXPLANATION", "METHOD_CONTEXT", "LIMITATION", "INTERPRETATION"}:
        if chunk_type == "narrative":
            boost += 0.6
        elif chunk_type == "table":
            boost -= 0.3
        if source_section in {"pre_text", "post_text"}:
            boost += 0.2

    return boost


def rerank_matches_with_metadata(matches: List[RetrievalMatch], plan: RetrievalPlan) -> List[RetrievalMatch]:
    reranked: List[RetrievalMatch] = []

    for match in matches:
        document_type = match.chunk.metadata.get("document_type")
        boost = _boost_for_document_type(document_type, plan)
        boost += _boost_for_metadata(match.chunk, plan)
        boost += _boost_for_chunk_structure(match.chunk, plan)
        boost += _boost_for_source_metadata(match.chunk, plan)
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


def _desired_chunk_type_quotas(plan: RetrievalPlan) -> dict[str, int]:
    top_k = plan.top_k
    if top_k <= 1:
        return {}

    if plan.retrieval_mode == "statistical_preferred":
        return {"table": min(2, top_k), "mixed": min(2, max(0, top_k - 2)), "narrative": min(1, max(0, top_k - 4))}

    if plan.retrieval_mode in {"broad_hybrid", "source_metadata_preferred"}:
        return {"table": min(2, top_k), "narrative": min(2, max(0, top_k - 2)), "mixed": min(1, max(0, top_k - 4))}

    if plan.retrieval_mode == "narrative_preferred":
        return {"narrative": min(2, top_k), "mixed": min(2, max(0, top_k - 2)), "table": min(1, max(0, top_k - 4))}

    return {}


def _evidence_group_id(match: RetrievalMatch) -> str:
    metadata = match.chunk.metadata
    return str(metadata.get("context_id") or match.chunk.doc_id)


def _select_diverse_top_k(matches: List[RetrievalMatch], plan: RetrievalPlan) -> List[RetrievalMatch]:
    if len(matches) <= plan.top_k:
        return matches

    quotas = _desired_chunk_type_quotas(plan)
    selected: List[RetrievalMatch] = []
    selected_chunk_ids = set()
    selected_counts_by_group: dict[str, int] = {}
    max_per_group = 2 if plan.top_k >= 4 else 1

    def can_take(match: RetrievalMatch, *, enforce_group_cap: bool = True) -> bool:
        if match.chunk.chunk_id in selected_chunk_ids:
            return False
        if not enforce_group_cap:
            return True
        group_id = _evidence_group_id(match)
        return selected_counts_by_group.get(group_id, 0) < max_per_group

    def take(match: RetrievalMatch) -> None:
        selected.append(match)
        selected_chunk_ids.add(match.chunk.chunk_id)
        group_id = _evidence_group_id(match)
        selected_counts_by_group[group_id] = selected_counts_by_group.get(group_id, 0) + 1

    for chunk_type, quota in quotas.items():
        taken_for_type = 0
        for match in matches:
            if taken_for_type >= quota or len(selected) >= plan.top_k:
                break
            if match.chunk.metadata.get("chunk_type") != chunk_type:
                continue
            if not can_take(match):
                continue
            take(match)
            taken_for_type += 1

    for match in matches:
        if len(selected) >= plan.top_k:
            break
        if not can_take(match):
            continue
        take(match)

    for match in matches:
        if len(selected) >= plan.top_k:
            break
        if not can_take(match, enforce_group_cap=False):
            continue
        take(match)

    selected.sort(key=lambda item: item.adjusted_score if item.adjusted_score is not None else item.score, reverse=True)
    return selected[: plan.top_k]


def retrieve_with_metadata_boost(
    retriever: BaseRetriever,
    query: str,
    chunks: List[Chunk],
    plan: RetrievalPlan,
) -> dict:
    filtered_chunks = _filter_chunks_for_plan(chunks, plan)
    candidate_matches = retriever.retrieve(query, filtered_chunks, top_k=plan.candidate_pool_size)
    if not candidate_matches:
        return {
            "filtered_chunk_count": len(filtered_chunks),
            "candidate_matches": [],
            "matches": [],
        }

    reranked = rerank_matches_with_metadata(candidate_matches, plan)
    final_matches = _select_diverse_top_k(reranked, plan)
    return {
        "filtered_chunk_count": len(filtered_chunks),
        "candidate_matches": candidate_matches,
        "matches": final_matches,
    }
