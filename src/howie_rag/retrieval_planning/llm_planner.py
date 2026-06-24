import json
import re
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
Do not default to statistical_preferred.
Use statistical_preferred only when the question clearly asks for numeric values, percentages, ratios, year-over-year changes, counts, totals, averages, or table lookups.
Use source_metadata_preferred when the question asks where in a report, which page, which section, which source, or according to which report section.
Use broad_hybrid for comparison questions when both table and narrative evidence may matter.
Use narrative_preferred for summary, explanation, limitation, interpretation, and method questions.
Use normal for simple factual questions that do not clearly need a table-focused or metadata-focused strategy.
Keep query_for_retrieval the same as the original question unless a very small rewrite is absolutely necessary.
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


NUMERIC_OR_TABLE_CUES = (
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
    "how many",
    "how much",
    "value",
    "amount",
    "revenue",
    "income",
    "expense",
    "margin",
    "balance",
    "debt",
    "shares",
)

SOURCE_METADATA_CUES = (
    "where",
    "which page",
    "which section",
    "which source",
    "according to",
    "reported in",
    "as disclosed in",
    "in the report",
)

NARRATIVE_CUES = (
    "summarize",
    "summary",
    "explain",
    "why",
    "how did",
    "limitation",
    "method",
    "interpret",
    "imply",
)


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


def _has_numeric_or_table_cues(query: str) -> bool:
    lowered = query.lower()
    if any(cue in lowered for cue in NUMERIC_OR_TABLE_CUES):
        return True
    if re.search(r"\b(19|20)\d{2}\b", lowered):
        return True
    if re.search(r"\b\d+(?:[.,]\d+)?%?\b", lowered):
        return True
    return False


def _has_source_metadata_cues(query: str) -> bool:
    lowered = query.lower()
    return any(cue in lowered for cue in SOURCE_METADATA_CUES)


def _has_narrative_cues(query: str) -> bool:
    lowered = query.lower()
    return any(cue in lowered for cue in NARRATIVE_CUES)


def _normalize_plan_with_guardrails(plan: RetrievalPlan) -> RetrievalPlan:
    query = plan.original_query
    has_numeric_cues = _has_numeric_or_table_cues(query)
    has_source_cues = _has_source_metadata_cues(query)
    has_narrative_cues = _has_narrative_cues(query)

    plan.query_for_retrieval = query

    if plan.detected_intent == "NAVIGATION":
        plan.retrieval_mode = "source_metadata_preferred"
    elif plan.detected_intent in {"SUMMARY", "EXPLANATION", "METHOD_CONTEXT", "LIMITATION", "INTERPRETATION"}:
        plan.retrieval_mode = "narrative_preferred"
    elif plan.detected_intent == "COMPARISON":
        plan.retrieval_mode = "broad_hybrid"
    elif plan.detected_intent in {"FACT", "TREND_PATTERN"} and has_numeric_cues:
        if plan.retrieval_mode == "normal":
            plan.retrieval_mode = "statistical_preferred"

    if plan.retrieval_mode == "statistical_preferred" and not has_numeric_cues:
        if has_source_cues:
            plan.retrieval_mode = "source_metadata_preferred"
        elif has_narrative_cues:
            plan.retrieval_mode = "narrative_preferred"
        else:
            plan.retrieval_mode = "normal"

    if has_source_cues and plan.retrieval_mode == "normal":
        plan.retrieval_mode = "source_metadata_preferred"

    if has_narrative_cues and plan.retrieval_mode == "normal":
        plan.retrieval_mode = "narrative_preferred"

    if plan.detected_intent in {"FACT", "COMPARISON", "TREND_PATTERN", "NAVIGATION"} and has_numeric_cues:
        if "table" not in plan.preferred_chunk_types:
            plan.preferred_chunk_types.append("table")

    if plan.retrieval_mode == "source_metadata_preferred":
        plan.metadata_preferences["prefer_source_metadata"] = True
        if not plan.preferred_document_types:
            plan.preferred_document_types = ["narrative", "mixed", "statistical"]

    elif plan.retrieval_mode == "statistical_preferred":
        plan.metadata_preferences["boost_tables_for_statistical"] = True
        if not plan.preferred_document_types:
            plan.preferred_document_types = ["statistical", "mixed"]
        if has_numeric_cues and "table" not in plan.preferred_chunk_types:
            plan.preferred_chunk_types.append("table")

    elif plan.retrieval_mode == "narrative_preferred":
        if not plan.preferred_document_types:
            plan.preferred_document_types = ["narrative", "mixed"]
        if not has_numeric_cues:
            plan.preferred_chunk_types = [chunk_type for chunk_type in plan.preferred_chunk_types if chunk_type != "table"]

    elif plan.retrieval_mode == "broad_hybrid":
        if not plan.preferred_document_types:
            plan.preferred_document_types = ["mixed", "statistical", "narrative"]

    plan.explanation = f"{plan.explanation} | guardrails_applied"
    return plan


class LLMRetrievalPlanner:
    def __init__(self, llm_client: BaseLLMClient, temperature: float = 0.0, max_tokens: int = 300) -> None:
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def build_plan(self, query: str, top_k: int = 5, candidate_pool_size: int = 30) -> RetrievalPlan:
        user_prompt = (
            f"Question:\n{query}\n\n"
            "Choose the best retrieval plan for this question. "
            "Do not overuse statistical_preferred. "
            "Keep query_for_retrieval equal to the original question unless a tiny rewrite is absolutely necessary. "
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
            plan = self._plan_from_payload(payload, query=query, top_k=top_k, candidate_pool_size=candidate_pool_size)
            return _normalize_plan_with_guardrails(plan)
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
