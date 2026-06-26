import json
from typing import Any, Dict

from howie_rag.core.schemas import IntentResult
from howie_rag.intent.base import BaseIntentClassifier
from howie_rag.intent.intent_labels import IntentLabel
from howie_rag.llm.base import BaseLLMClient


INTENT_SYSTEM_PROMPT = """You are an intent classifier for research and financial document questions.
Return only valid JSON with no markdown fences.
Allowed intents: FACT, SUMMARY, COMPARISON, METHOD_CONTEXT, LIMITATION, TREND_PATTERN, EXPLANATION, SOURCE_SEEKING, NAVIGATION, INTERPRETATION, DECISION_SUPPORT, FOLLOWUP, UNKNOWN.
Output keys:
- intent: string
- confidence: number between 0 and 1
- reasoning: short string
"""


def _extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Intent classifier output did not contain a JSON object")
    return json.loads(stripped[start : end + 1])


class LLMIntentClassifier(BaseIntentClassifier):
    def __init__(self, llm_client: BaseLLMClient, temperature: float = 0.0, max_tokens: int = 120) -> None:
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def classify(self, question: str) -> IntentResult:
        user_prompt = f"Question:\n{question}\n\nClassify the question intent."
        try:
            response = self.llm_client.generate(
                system_prompt=INTENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            payload = _extract_json_object(response)
            intent = str(payload.get("intent") or "UNKNOWN").strip()
            if intent not in {label.value for label in IntentLabel}:
                intent = IntentLabel.UNKNOWN.value
            confidence = payload.get("confidence")
            if not isinstance(confidence, (int, float)):
                confidence = 0.5 if intent != IntentLabel.UNKNOWN.value else 0.2
            reasoning = payload.get("reasoning")
            if not isinstance(reasoning, str):
                reasoning = ""
            return IntentResult(intent=intent, confidence=float(confidence), reasoning=reasoning)
        except Exception as error:
            return IntentResult(
                intent=IntentLabel.UNKNOWN.value,
                confidence=0.2,
                reasoning=f"LLM intent fallback due to parse failure: {error}",
            )
