from typing import List, Tuple

from howie_rag.core.schemas import IntentResult
from howie_rag.intent.base import BaseIntentClassifier
from howie_rag.intent.intent_labels import IntentLabel


class RuleBasedIntentClassifier(BaseIntentClassifier):
    _KEYWORDS_BY_INTENT: List[Tuple[IntentLabel, Tuple[str, ...]]] = [
        (
            IntentLabel.FOLLOWUP,
            (
                "what about",
                "how about",
                "and in",
                "and for",
                "what about in",
                "what about for",
            ),
        ),
        (
            IntentLabel.COMPARISON,
            (
                "compare",
                "comparison",
                "difference",
                "differences",
                "similarities",
                "versus",
                "vs",
                "between",
                "differ",
            ),
        ),
        (
            IntentLabel.DECISION_SUPPORT,
            (
                "should we",
                "should i",
                "what should",
                "recommend",
                "best option",
                "most effective to adopt",
                "which intervention",
                "what action",
                "prioritize",
            ),
        ),
        (
            IntentLabel.SOURCE_SEEKING,
            (
                "source",
                "citation",
                "reference",
                "references",
                "which paper",
                "which study",
                "who reported",
                "reported this",
            ),
        ),
        (
            IntentLabel.NAVIGATION,
            (
                "where can i find",
                "document",
                "report",
                "page",
                "link",
                "section",
                "appendix",
                "table",
            ),
        ),
        (
            IntentLabel.LIMITATION,
            (
                "limitation",
                "limitations",
                "caveat",
                "bias",
                "uncertainty",
                "weakness",
                "cannot conclude",
            ),
        ),
        (
            IntentLabel.INTERPRETATION,
            (
                "what do these findings imply",
                "what do the findings imply",
                "implication",
                "implications",
                "interpret",
                "interpretation",
                "suggest about",
                "indicate about",
                "mean for policy",
            ),
        ),
        (
            IntentLabel.METHOD_CONTEXT,
            (
                "method",
                "methodology",
                "sample",
                "survey design",
                "data collection",
                "participants",
                "research design",
                "how was the data collected",
            ),
        ),
        (
            IntentLabel.EXPLANATION,
            (
                "why",
                "explain",
                "explanation",
                "how did",
                "what does this mean",
            ),
        ),
        (
            IntentLabel.TREND_PATTERN,
            (
                "trend",
                "pattern",
                "over time",
                "increase",
                "decrease",
                "changes",
                "trajectory",
            ),
        ),
        (
            IntentLabel.SUMMARY,
            (
                "summarize",
                "summary",
                "overview",
                "key findings",
                "main findings",
                "main points",
                "what are the findings",
            ),
        ),
        (
            IntentLabel.FACT,
            (
                "what is",
                "what was",
                "when",
                "where",
                "how many",
                "how much",
                "which result",
                "what percentage",
                "sample size",
            ),
        ),
    ]

    def classify(self, question: str) -> IntentResult:
        normalized_question = question.lower()

        for intent, keywords in self._KEYWORDS_BY_INTENT:
            for keyword in keywords:
                if keyword == "sample" and "sample size" in normalized_question:
                    continue
                if keyword in normalized_question:
                    return IntentResult(
                        intent=intent.value,
                        confidence=0.9,
                        reasoning=f"Matched keyword: '{keyword}'",
                    )

        return IntentResult(
            intent=IntentLabel.UNKNOWN.value,
            confidence=0.2,
            reasoning="No keyword matched.",
        )
