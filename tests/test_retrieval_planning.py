from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.core.schemas import Chunk
from howie_rag.intent.rule_based import RuleBasedIntentClassifier
from howie_rag.retrieval.keyword_retriever import KeywordRetriever
from howie_rag.retrieval.base import RetrievalMatch
from howie_rag.retrieval_planning.experiment_variants import retrieve_with_variant
from howie_rag.retrieval_planning.metadata_aware_retriever import (
    rerank_matches_with_metadata,
    retrieve_with_metadata_boost,
)
from howie_rag.retrieval_planning.rule_based_planner import build_retrieval_plan


def test_rule_based_planner_maps_fact_to_normal() -> None:
    plan = build_retrieval_plan("What is student mobility?", "FACT")
    assert plan.retrieval_mode == "normal"
    assert plan.preferred_document_types == ["narrative", "mixed"]


def test_rule_based_planner_maps_trend_pattern_to_statistical() -> None:
    plan = build_retrieval_plan("What trend is visible?", "TREND_PATTERN")
    assert plan.retrieval_mode == "statistical_preferred"
    assert plan.preferred_document_types == ["statistical", "mixed"]
    assert plan.preferred_chunk_types == ["table"]


def test_rule_based_planner_maps_summary_to_narrative_preferred() -> None:
    plan = build_retrieval_plan("Summarize the report", "SUMMARY")
    assert plan.retrieval_mode == "narrative_preferred"
    assert plan.preferred_chunk_types == ["narrative"]


def test_rule_based_planner_keeps_fact_chunk_preferences_broad() -> None:
    plan = build_retrieval_plan("What was the 2019 revenue?", "FACT")
    assert plan.preferred_chunk_types == []


def test_rule_based_planner_falls_back_for_unknown() -> None:
    plan = build_retrieval_plan("Hello there", "UNKNOWN")
    assert plan.retrieval_mode == "normal"


def test_metadata_reranking_boosts_preferred_document_type() -> None:
    plan = build_retrieval_plan("What trend is visible?", "TREND_PATTERN")
    matches = [
        RetrievalMatch(
            chunk=Chunk(
                chunk_id="1",
                doc_id="doc-1",
                text="a",
                metadata={"document_type": "narrative", "chunk_type": "narrative", "table_line_ratio": 0.0},
            ),
            score=1.0,
        ),
        RetrievalMatch(
            chunk=Chunk(
                chunk_id="2",
                doc_id="doc-2",
                text="b",
                metadata={
                    "document_type": "statistical",
                    "has_tables": True,
                    "chunk_type": "table",
                    "has_table_like_content": True,
                    "table_line_ratio": 1.0,
                },
            ),
            score=1.0,
        ),
    ]
    reranked = rerank_matches_with_metadata(matches, plan)
    assert reranked[0].chunk.chunk_id == "2"
    assert reranked[0].adjusted_score > reranked[0].original_score


def test_metadata_reranking_handles_missing_metadata() -> None:
    plan = build_retrieval_plan("Summarize the report", "SUMMARY")
    matches = [RetrievalMatch(chunk=Chunk(chunk_id="1", doc_id="doc-1", text="a", metadata={}), score=1.0)]
    reranked = rerank_matches_with_metadata(matches, plan)
    assert reranked[0].chunk.chunk_id == "1"
    assert reranked[0].adjusted_score == 1.0


def test_metadata_reranking_prefers_table_chunk_for_fact_queries() -> None:
    plan = build_retrieval_plan("What was the 2019 revenue?", "FACT")
    matches = [
        RetrievalMatch(
            chunk=Chunk(
                chunk_id="1",
                doc_id="doc-1",
                text="Revenue discussion",
                metadata={"document_type": "mixed", "chunk_type": "narrative", "table_line_ratio": 0.0},
            ),
            score=10.0,
        ),
        RetrievalMatch(
            chunk=Chunk(
                chunk_id="2",
                doc_id="doc-1",
                text="| 2019 | 100 |",
                metadata={
                    "document_type": "mixed",
                    "has_tables": True,
                    "chunk_type": "table",
                    "has_table_like_content": True,
                    "table_line_ratio": 1.0,
                },
            ),
            score=9.0,
        ),
    ]

    reranked = rerank_matches_with_metadata(matches, plan)

    assert reranked[0].chunk.chunk_id == "2"
    assert reranked[0].adjusted_score > reranked[1].adjusted_score


def test_retrieve_with_metadata_boost_filters_to_preferred_chunk_types() -> None:
    plan = build_retrieval_plan("What was the 2019 revenue?", "FACT", top_k=1, candidate_pool_size=5)
    chunks = [
        Chunk(
            chunk_id="1",
            doc_id="doc-1",
            text="revenue in 2019 was discussed in the report",
            metadata={"document_type": "mixed", "chunk_type": "narrative"},
        ),
        Chunk(
            chunk_id="2",
            doc_id="doc-2",
            text="revenue 2019 100 2020 120",
            metadata={
                "document_type": "mixed",
                "chunk_type": "table",
                "has_tables": True,
                "has_table_like_content": True,
                "table_line_ratio": 1.0,
            },
        ),
    ]

    matches = retrieve_with_metadata_boost(KeywordRetriever(), "revenue 2019", chunks, plan)

    assert matches
    assert matches[0].chunk.chunk_id == "2"


def test_retrieve_with_metadata_boost_does_not_filter_fact_queries_too_hard() -> None:
    plan = build_retrieval_plan("What was the 2019 revenue?", "FACT", top_k=1, candidate_pool_size=5)
    chunks = [
        Chunk(
            chunk_id="1",
            doc_id="doc-1",
            text="revenue 2019 100",
            metadata={"document_type": "mixed", "chunk_type": "table", "has_table_like_content": True, "table_line_ratio": 1.0},
        ),
        Chunk(
            chunk_id="2",
            doc_id="doc-1",
            text="revenue 2019 described in the report",
            metadata={"document_type": "mixed", "chunk_type": "narrative", "table_line_ratio": 0.0},
        ),
    ]

    matches = retrieve_with_metadata_boost(KeywordRetriever(), "revenue 2019", chunks, plan)

    assert {match.chunk.chunk_id for match in matches} == {"1"}


def test_document_aware_variant_uses_metadata_preference() -> None:
    chunks = [
        Chunk(chunk_id="1", doc_id="doc-1", text="student mobility", metadata={"document_type": "narrative"}),
        Chunk(chunk_id="2", doc_id="doc-2", text="student mobility", metadata={"document_type": "statistical"}),
    ]
    output = retrieve_with_variant(
        query="student mobility",
        chunks=chunks,
        retriever_name="keyword",
        variant="document_aware",
        top_k=1,
        candidate_pool_size=2,
    )
    assert output["retrieval_plan"] is not None
    assert output["matches"][0].chunk.metadata["document_type"] == "narrative"


def test_naive_variant_does_not_use_planner() -> None:
    chunks = [Chunk(chunk_id="1", doc_id="doc-1", text="student mobility", metadata={})]
    output = retrieve_with_variant(
        query="student mobility",
        chunks=chunks,
        retriever_name="keyword",
        variant="naive",
        top_k=1,
        candidate_pool_size=2,
    )
    assert output["retrieval_plan"] is None


def test_intent_document_aware_variant_uses_intent_classifier() -> None:
    chunks = [
        Chunk(chunk_id="1", doc_id="doc-1", text="student mobility trend over time", metadata={"document_type": "statistical", "has_tables": True}),
        Chunk(chunk_id="2", doc_id="doc-2", text="student mobility narrative explanation", metadata={"document_type": "narrative"}),
    ]
    output = retrieve_with_variant(
        query="What trend do we see in student mobility?",
        chunks=chunks,
        retriever_name="keyword",
        variant="intent_document_aware",
        intent_classifier=RuleBasedIntentClassifier(),
        top_k=1,
        candidate_pool_size=2,
    )
    assert output["detected_intent"] == "TREND_PATTERN"
    assert output["retrieval_plan"].retrieval_mode == "statistical_preferred"
