import json
from typing import Any, Dict, Optional

from howie_rag.llm.base import BaseLLMClient
from howie_rag.retrieval_planning.retrieval_plan_schema import RetrievalPlan
from howie_rag.retrieval_planning.rule_based_planner import build_retrieval_plan


ALLOWED_INTENTS = {
    "FACT",
    "SUMMARY",
    "COMPARISON",
    "METHOD_CONTEXT",
    "LIMITATION",
    "TREND_PATTERN",
    "EXPLANATION",
    "SOURCE_SEEKING",
    "NAVIGATION",
    "INTERPRETATION",
    "DECISION_SUPPORT",
    "FOLLOWUP",
    "UNKNOWN",
}

ALLOWED_RETRIEVAL_MODES = {
    "normal",
    "narrative_preferred",
    "statistical_preferred",
    "source_metadata_preferred",
    "broad_hybrid",
}

ALLOWED_DOCUMENT_TYPES = {"narrative", "mixed", "statistical"}
ALLOWED_CHUNK_TYPES = {"narrative", "mixed", "table"}


PLANNER_SYSTEM_PROMPT = """You are a retrieval planner for a RAG system.
Return only valid JSON with no markdown fences.
Choose the best retrieval intent and retrieval configuration for a user question.
Prefer table and statistical evidence for finance, numeric, comparison, trend, and table-seeking questions.
Prefer narrative evidence for summary, explanation, limitation, and method questions.
Allowed intents: FACT, SUMMARY, COMPARISON, METHOD_CONTEXT, LIMITATION, TREND_PATTERN, EXPLANATION, SOURCE_SEEKING, NAVIGATION, INTERPRETATION, DECISION_SUPPORT, FOLLOWUP, UNKNOWN.
Allowed retrieval_mode values: normal, narrative_preferred, statistical_preferred, source_metadata_preferred, broad_hybrid.
Allowed preferred_document_types values: narrative, mixed, statistical.
Allowed preferred_chunk_types values: narrative, mixed, table.
Output JSON keys:
- detected_intent: string
- retrieval_mode: string
- preferred_document_types: list of strings
- preferred_chunk_types: list of strings
- metadata_preferences: object
- query_for_retrieval: string
- explanation: string
"""


def _extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Planner output did not contain a JSON object")

    return json.loads(stripped[start : end + 1])


def _normalize_string_list(value: Any, allowed_values: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        if not isinstance(item, str):
            continue
        stripped = item.strip().lower()
        if stripped in allowed_values and stripped not in normalized:
            normalized.append(stripped)
    return normalized


class LLMRetrievalPlanner:
    def __init__(self, llm_client: BaseLLMClient, temperature: float = 0.0, max_tokens: int = 300) -> None:
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def build_plan(self, query: str, top_k: int = 5, candidate_pool_size: int = 30) -> RetrievalPlan:
        user_prompt = (
            f"Question:\n{query}\n\n"
            "Choose the best retrieval plan for this question. "
            "If the question likely needs values from a table, set preferred_chunk_types to include 'table'."
        )

        try:
            response_text = self.llm_client.generate(
                system_prompt=PLANNER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            payload = _extract_json_object(response_text)
            return self._plan_from_payload(payload, query=query, top_k=top_k, candidate_pool_size=candidate_pool_size)
        except Exception as error:
            fallback = build_retrieval_plan(
                query=query,
                detected_intent="UNKNOWN",
                top_k=top_k,
                candidate_pool_size=candidate_pool_size,
            )
            fallback.explanation = f"Fallback to rule-based planner because LLM planner failed: {error}"
            return fallback

    def _plan_from_payload(
        self,
        payload: Dict[str, Any],
        *,
        query: str,
        top_k: int,
        candidate_pool_size: int,
    ) -> RetrievalPlan:
        detected_intent = payload.get("detected_intent")
        if not isinstance(detected_intent, str) or detected_intent.strip() not in ALLOWED_INTENTS:
            raise ValueError("Planner returned invalid detected_intent")
        normalized_intent = detected_intent.strip()

        retrieval_mode = payload.get("retrieval_mode")
        if not isinstance(retrieval_mode, str) or retrieval_mode.strip() not in ALLOWED_RETRIEVAL_MODES:
            raise ValueError("Planner returned invalid retrieval_mode")
        normalized_mode = retrieval_mode.strip()

        preferred_document_types = _normalize_string_list(
            payload.get("preferred_document_types"),
            ALLOWED_DOCUMENT_TYPES,
        )
        preferred_chunk_types = _normalize_string_list(
            payload.get("preferred_chunk_types"),
            ALLOWED_CHUNK_TYPES,
        )
        metadata_preferences = payload.get("metadata_preferences")
        if not isinstance(metadata_preferences, dict):
            metadata_preferences = {}

        query_for_retrieval = payload.get("query_for_retrieval")
        if not isinstance(query_for_retrieval, str) or not query_for_retrieval.strip():
            query_for_retrieval = query

        explanation = payload.get("explanation")
        if not isinstance(explanation, str):
            explanation = ""

        return RetrievalPlan(
            original_query=query,
            query_for_retrieval=query_for_retrieval.strip(),
            detected_intent=normalized_intent,
            retrieval_mode=normalized_mode,
            preferred_document_types=preferred_document_types,
            preferred_chunk_types=preferred_chunk_types,
            metadata_preferences=metadata_preferences,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
            explanation=explanation,
        )
