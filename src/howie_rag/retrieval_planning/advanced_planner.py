from howie_rag.retrieval_planning.retrieval_plan_schema import RetrievalPlan
from howie_rag.retrieval_planning.rule_based_planner import build_retrieval_plan


INTENT_QUERY_EXPANSIONS = {
    "FACT": "value amount table reported disclosed",
    "SUMMARY": "summary overview main findings explanation",
    "COMPARISON": "compare difference change versus between table",
    "METHOD_CONTEXT": "method methodology sample participants narrative",
    "LIMITATION": "limitation caveat uncertainty narrative",
    "TREND_PATTERN": "trend change increase decrease over time year period table",
    "EXPLANATION": "explain why reason cause narrative",
    "SOURCE_SEEKING": "reported disclosed source section page note",
    "NAVIGATION": "reported disclosed section page table note",
    "INTERPRETATION": "interpret implication meaning narrative",
    "DECISION_SUPPORT": "compare best option effective adopt table narrative",
    "FOLLOWUP": "context previous answer related",
    "UNKNOWN": "",
}


INTENT_POOL_STRATEGIES = {
    "FACT": {"table": 10, "mixed": 8, "narrative": 4},
    "SUMMARY": {"narrative": 10, "mixed": 8, "table": 4},
    "COMPARISON": {"table": 10, "mixed": 8, "narrative": 8},
    "METHOD_CONTEXT": {"narrative": 10, "mixed": 8, "table": 3},
    "LIMITATION": {"narrative": 10, "mixed": 8, "table": 3},
    "TREND_PATTERN": {"table": 12, "mixed": 8, "narrative": 4},
    "EXPLANATION": {"narrative": 10, "mixed": 8, "table": 4},
    "SOURCE_SEEKING": {"metadata": 10, "narrative": 6, "table": 6, "mixed": 6},
    "NAVIGATION": {"metadata": 10, "table": 8, "narrative": 6, "mixed": 6},
    "INTERPRETATION": {"narrative": 10, "mixed": 8, "table": 4},
    "DECISION_SUPPORT": {"table": 8, "mixed": 8, "narrative": 8},
    "FOLLOWUP": {"mixed": 8, "narrative": 8, "table": 6},
    "UNKNOWN": {"mixed": 8, "narrative": 8, "table": 6},
}


def _rewritten_query(query: str, detected_intent: str) -> str:
    expansion = INTENT_QUERY_EXPANSIONS.get(detected_intent, "")
    if not expansion:
        return query
    return f"{query} {expansion}".strip()


def build_advanced_retrieval_plan(
    query: str,
    detected_intent: str,
    top_k: int = 5,
    candidate_pool_size: int = 30,
) -> RetrievalPlan:
    base_plan = build_retrieval_plan(
        query=query,
        detected_intent=detected_intent,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
    )
    base_plan.query_for_retrieval = _rewritten_query(query, detected_intent)
    base_plan.metadata_preferences["candidate_pools"] = INTENT_POOL_STRATEGIES.get(
        detected_intent,
        INTENT_POOL_STRATEGIES["UNKNOWN"],
    )
    base_plan.metadata_preferences["multi_query"] = True
    base_plan.explanation = f"{base_plan.explanation} | advanced_guided_retrieval"
    return base_plan
