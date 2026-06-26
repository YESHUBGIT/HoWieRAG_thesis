from typing import Dict, List

from howie_rag.core.schemas import Chunk
from howie_rag.retrieval.base import BaseRetriever, RetrievalMatch
from howie_rag.retrieval_planning.metadata_aware_retriever import rerank_matches_with_metadata
from howie_rag.retrieval_planning.metadata_aware_retriever import _select_diverse_top_k
from howie_rag.retrieval_planning.retrieval_plan_schema import RetrievalPlan


def _chunk_pool_name(chunk: Chunk) -> str:
    metadata = chunk.metadata
    chunk_type = metadata.get("chunk_type")
    if chunk_type in {"table", "mixed", "narrative"}:
        return chunk_type
    return "other"


def _is_metadata_rich(chunk: Chunk) -> bool:
    metadata = chunk.metadata
    return bool(
        metadata.get("source_page_number")
        or metadata.get("source_section")
        or (metadata.get("source_title") and metadata.get("source_title") != metadata.get("context_id"))
    )


def _build_pool_map(chunks: List[Chunk]) -> Dict[str, List[Chunk]]:
    pools: Dict[str, List[Chunk]] = {"table": [], "mixed": [], "narrative": [], "metadata": [], "all": list(chunks)}
    for chunk in chunks:
        pool_name = _chunk_pool_name(chunk)
        pools.setdefault(pool_name, []).append(chunk)
        if _is_metadata_rich(chunk):
            pools["metadata"].append(chunk)
    return pools


def _query_variants(plan: RetrievalPlan) -> List[str]:
    variants = [plan.original_query]
    rewritten = plan.query_for_retrieval.strip()
    if rewritten and rewritten not in variants:
        variants.append(rewritten)
    return variants


def _pool_limits(plan: RetrievalPlan) -> Dict[str, int]:
    strategy = plan.metadata_preferences.get("candidate_pools") if isinstance(plan.metadata_preferences, dict) else None
    if isinstance(strategy, dict) and strategy:
        return {str(pool): int(limit) for pool, limit in strategy.items() if int(limit) > 0}
    return {"all": plan.candidate_pool_size}


def retrieve_with_guided_plan(
    retriever: BaseRetriever,
    chunks: List[Chunk],
    plan: RetrievalPlan,
) -> dict:
    pool_map = _build_pool_map(chunks)
    query_variants = _query_variants(plan)
    pool_limits = _pool_limits(plan)

    candidate_by_chunk_id: Dict[str, RetrievalMatch] = {}
    per_pool_cap = max(plan.top_k, plan.candidate_pool_size // max(1, len(pool_limits)))

    for pool_name, pool_limit in pool_limits.items():
        pool_chunks = pool_map.get(pool_name, [])
        if not pool_chunks:
            continue
        pool_candidate_by_chunk_id: Dict[str, RetrievalMatch] = {}
        for query_index, query in enumerate(query_variants):
            matches = retriever.retrieve(query, pool_chunks, top_k=min(pool_limit, len(pool_chunks)))
            for match in matches:
                adjusted_score = match.score * (1.0 if query_index == 0 else 0.92)
                adjusted_match = RetrievalMatch(
                    chunk=match.chunk,
                    score=adjusted_score,
                    original_score=match.score,
                )
                existing = pool_candidate_by_chunk_id.get(match.chunk.chunk_id)
                if existing is None or adjusted_match.score > existing.score:
                    pool_candidate_by_chunk_id[match.chunk.chunk_id] = adjusted_match

        pool_candidates = list(pool_candidate_by_chunk_id.values())
        pool_candidates.sort(key=lambda match: match.score, reverse=True)
        pool_candidates = pool_candidates[:per_pool_cap]
        for match in pool_candidates:
            existing = candidate_by_chunk_id.get(match.chunk.chunk_id)
            if existing is None or match.score > existing.score:
                candidate_by_chunk_id[match.chunk.chunk_id] = match

    candidate_matches = list(candidate_by_chunk_id.values())
    candidate_matches.sort(key=lambda match: match.score, reverse=True)
    candidate_matches = candidate_matches[: plan.candidate_pool_size]

    if not candidate_matches:
        fallback_matches = retriever.retrieve(plan.original_query, chunks, top_k=plan.candidate_pool_size)
        return {
            "candidate_matches": fallback_matches,
            "matches": fallback_matches[: plan.top_k],
        }

    reranked = rerank_matches_with_metadata(candidate_matches, plan)
    final_matches = _select_diverse_top_k(reranked, plan)
    return {
        "candidate_matches": candidate_matches,
        "matches": final_matches,
    }
