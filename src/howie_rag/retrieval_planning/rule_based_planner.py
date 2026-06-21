from howie_rag.retrieval_planning.retrieval_plan_schema import RetrievalPlan


INTENT_PLAN_MAPPING = {
    "FACT": ("normal", ["narrative", "mixed"]),
    "SUMMARY": ("narrative_preferred", ["narrative", "mixed"]),
    "COMPARISON": ("broad_hybrid", ["mixed", "statistical", "narrative"]),
    "METHOD_CONTEXT": ("narrative_preferred", ["narrative", "mixed"]),
    "LIMITATION": ("narrative_preferred", ["narrative", "mixed"]),
    "TREND_PATTERN": ("statistical_preferred", ["statistical", "mixed"]),
    "EXPLANATION": ("narrative_preferred", ["narrative", "mixed"]),
    "SOURCE_SEEKING": ("source_metadata_preferred", ["narrative", "mixed", "statistical"]),
    "NAVIGATION": ("source_metadata_preferred", ["narrative", "mixed", "statistical"]),
    "INTERPRETATION": ("narrative_preferred", ["narrative", "mixed"]),
    "DECISION_SUPPORT": ("broad_hybrid", ["mixed", "statistical", "narrative"]),
    "FOLLOWUP": ("normal", []),
    "UNKNOWN": ("normal", []),
}


def build_retrieval_plan(
    query: str,
    detected_intent: str,
    top_k: int = 5,
    candidate_pool_size: int = 30,
) -> RetrievalPlan:
    retrieval_mode, preferred_document_types = INTENT_PLAN_MAPPING.get(detected_intent, ("normal", []))
    metadata_preferences = {
        "boost_tables_for_statistical": detected_intent == "TREND_PATTERN",
        "prefer_source_metadata": retrieval_mode == "source_metadata_preferred",
    }

    return RetrievalPlan(
        original_query=query,
        query_for_retrieval=query,
        detected_intent=detected_intent,
        retrieval_mode=retrieval_mode,
        preferred_document_types=preferred_document_types,
        preferred_chunk_types=[],
        metadata_preferences=metadata_preferences,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        explanation=(
            f"Rule-based plan for intent {detected_intent}: mode={retrieval_mode}, "
            f"preferred_document_types={preferred_document_types}"
        ),
    )
