from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.core.schemas import Chunk
from howie_rag.llm.base import BaseLLMClient
from howie_rag.intent.llm_intent import LLMIntentClassifier
from howie_rag.retrieval_planning.experiment_variants import retrieve_with_variant
from howie_rag.retrieval_planning.llm_planner import LLMRetrievalPlanner


class FakeLLMClient(BaseLLMClient):
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 400,
    ) -> str:
        return self.response_text


def test_llm_planner_parses_json_plan() -> None:
    planner = LLMRetrievalPlanner(
        FakeLLMClient(
            '{"detected_intent":"FACT","retrieval_mode":"statistical_preferred","preferred_document_types":["statistical","mixed"],"preferred_chunk_types":["table"],"metadata_preferences":{"prefer_source_metadata":true},"query_for_retrieval":"2019 revenue table","explanation":"Numeric question should prefer table evidence."}'
        )
    )

    plan = planner.build_plan("What was the 2019 revenue?", top_k=5, candidate_pool_size=30)

    assert plan.detected_intent == "FACT"
    assert plan.retrieval_mode == "statistical_preferred"
    assert plan.preferred_chunk_types == ["table"]
    assert plan.query_for_retrieval == "What was the 2019 revenue?"


def test_llm_planner_falls_back_when_output_is_invalid() -> None:
    planner = LLMRetrievalPlanner(FakeLLMClient("not valid json"))

    plan = planner.build_plan("What was the 2019 revenue?", top_k=5, candidate_pool_size=30)

    assert plan.detected_intent == "UNKNOWN"
    assert "Fallback to rule-based planner" in plan.explanation


def test_llm_document_aware_variant_uses_llm_plan_for_table_preference() -> None:
    planner = LLMRetrievalPlanner(
        FakeLLMClient(
            '{"detected_intent":"FACT","retrieval_mode":"statistical_preferred","preferred_document_types":["statistical","mixed"],"preferred_chunk_types":["table"],"metadata_preferences":{},"query_for_retrieval":"2019 revenue table","explanation":"Prefer table chunk."}'
        )
    )
    chunks = [
        Chunk(
            chunk_id="1",
            doc_id="doc-1",
            text="Revenue discussion for 2019.",
            metadata={"document_type": "mixed", "chunk_type": "narrative", "table_line_ratio": 0.0},
        ),
        Chunk(
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
    ]

    output = retrieve_with_variant(
        query="What was the 2019 revenue?",
        chunks=chunks,
        retriever_name="keyword",
        variant="llm_document_aware",
        llm_planner=planner,
        top_k=1,
        candidate_pool_size=2,
    )

    assert output["detected_intent"] == "FACT"
    assert output["retrieval_plan"].preferred_chunk_types == ["table"]
    assert output["matches"][0].chunk.chunk_id == "2"


def test_llm_intent_document_aware_variant_uses_llm_intent_only() -> None:
    classifier = LLMIntentClassifier(
        FakeLLMClient('{"intent":"TREND_PATTERN","confidence":0.9,"reasoning":"trend query"}')
    )
    chunks = [
        Chunk(
            chunk_id="1",
            doc_id="doc-1",
            text="mobility narrative explanation",
            metadata={"document_type": "narrative", "chunk_type": "narrative", "table_line_ratio": 0.0},
        ),
        Chunk(
            chunk_id="2",
            doc_id="doc-2",
            text="mobility trend 2019 10 2020 12",
            metadata={
                "document_type": "mixed",
                "has_tables": True,
                "chunk_type": "table",
                "has_table_like_content": True,
                "table_line_ratio": 1.0,
            },
        ),
    ]

    output = retrieve_with_variant(
        query="What trend do we see in student mobility?",
        chunks=chunks,
        retriever_name="keyword",
        variant="llm_intent_document_aware",
        llm_intent_classifier=classifier,
        top_k=1,
        candidate_pool_size=2,
    )

    assert output["detected_intent"] == "TREND_PATTERN"
    assert output["matches"][0].chunk.chunk_id == "2"


def test_llm_planner_guardrails_prevent_unjustified_statistical_mode() -> None:
    planner = LLMRetrievalPlanner(
        FakeLLMClient(
            '{"detected_intent":"SUMMARY","retrieval_mode":"statistical_preferred","preferred_document_types":["statistical"],"preferred_chunk_types":["table"],"metadata_preferences":{},"query_for_retrieval":"summary of the report","explanation":"bad initial choice"}'
        )
    )

    plan = planner.build_plan("Summarize the main findings of the report.", top_k=5, candidate_pool_size=30)

    assert plan.detected_intent == "SUMMARY"
    assert plan.retrieval_mode == "narrative_preferred"
    assert "table" not in plan.preferred_chunk_types
    assert plan.query_for_retrieval == "Summarize the main findings of the report."


def test_llm_planner_guardrails_force_navigation_to_source_metadata() -> None:
    planner = LLMRetrievalPlanner(
        FakeLLMClient(
            '{"detected_intent":"NAVIGATION","retrieval_mode":"statistical_preferred","preferred_document_types":["mixed"],"preferred_chunk_types":["table"],"metadata_preferences":{},"query_for_retrieval":"which page has this","explanation":"bad initial choice"}'
        )
    )

    plan = planner.build_plan("Which page in the report contains the debt maturities table?", top_k=5, candidate_pool_size=30)

    assert plan.detected_intent == "NAVIGATION"
    assert plan.retrieval_mode == "source_metadata_preferred"
    assert plan.metadata_preferences["prefer_source_metadata"] is True
