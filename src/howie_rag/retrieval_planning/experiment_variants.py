from typing import List, Optional

from howie_rag.core.schemas import Chunk
from howie_rag.intent.base import BaseIntentClassifier
from howie_rag.retrieval.base import BaseRetriever
from howie_rag.retrieval.base import RetrievalMatch
from howie_rag.retrieval.factory import create_retriever
from howie_rag.retrieval_planning.llm_planner import LLMRetrievalPlanner
from howie_rag.retrieval_planning.metadata_aware_retriever import retrieve_with_metadata_boost
from howie_rag.retrieval_planning.retrieval_plan_schema import RetrievalPlan
from howie_rag.retrieval_planning.rule_based_planner import build_retrieval_plan


def build_document_aware_default_plan(query: str, top_k: int, candidate_pool_size: int) -> RetrievalPlan:
    return RetrievalPlan(
        original_query=query,
        query_for_retrieval=query,
        detected_intent="UNUSED",
        retrieval_mode="document_aware_default",
        preferred_document_types=["narrative", "mixed"],
        preferred_chunk_types=[],
        metadata_preferences={},
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        explanation="Document-aware default plan prefers narrative and mixed documents.",
    )


def retrieve_with_variant(
    query: str,
    chunks: List[Chunk],
    retriever_name: str,
    variant: str,
    retriever: Optional[BaseRetriever] = None,
    intent_classifier: Optional[BaseIntentClassifier] = None,
    llm_planner: Optional[LLMRetrievalPlanner] = None,
    top_k: int = 5,
    candidate_pool_size: int = 30,
) -> dict:
    if retriever is None:
        retriever = create_retriever(retriever_name)
    normalized_variant = variant.lower()

    if normalized_variant == "naive":
        matches = retriever.retrieve(query, chunks, top_k=top_k)
        return {
            "variant": "naive",
            "detected_intent": None,
            "retrieval_plan": None,
            "matches": matches,
        }

    if normalized_variant == "document_aware":
        plan = build_document_aware_default_plan(query, top_k=top_k, candidate_pool_size=candidate_pool_size)
        matches = retrieve_with_metadata_boost(retriever, query, chunks, plan)
        return {
            "variant": "document_aware",
            "detected_intent": None,
            "retrieval_plan": plan,
            "matches": matches,
        }

    if normalized_variant == "intent_document_aware":
        if intent_classifier is None:
            raise ValueError("intent_classifier is required for intent_document_aware retrieval")
        intent_result = intent_classifier.classify(query)
        plan = build_retrieval_plan(
            query=query,
            detected_intent=intent_result.intent,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
        )
        matches = retrieve_with_metadata_boost(retriever, query, chunks, plan)
        return {
            "variant": "intent_document_aware",
            "detected_intent": intent_result.intent,
            "retrieval_plan": plan,
            "matches": matches,
        }

    if normalized_variant == "llm_document_aware":
        if llm_planner is None:
            raise ValueError("llm_planner is required for llm_document_aware retrieval")
        plan = llm_planner.build_plan(query=query, top_k=top_k, candidate_pool_size=candidate_pool_size)
        matches = retrieve_with_metadata_boost(retriever, plan.query_for_retrieval, chunks, plan)
        return {
            "variant": "llm_document_aware",
            "detected_intent": plan.detected_intent,
            "retrieval_plan": plan,
            "matches": matches,
        }

    raise ValueError(f"Unsupported retrieval planning variant: {variant}")
