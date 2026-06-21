from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.intent.intent_labels import IntentLabel
from howie_rag.intent.rule_based import RuleBasedIntentClassifier


def test_summary_intent() -> None:
    classifier = RuleBasedIntentClassifier()
    result = classifier.classify("What are the key findings about student mobility?")
    assert result.intent == IntentLabel.SUMMARY


def test_comparison_intent() -> None:
    classifier = RuleBasedIntentClassifier()
    result = classifier.classify("Compare study A and study B")
    assert result.intent == IntentLabel.COMPARISON


def test_navigation_intent() -> None:
    classifier = RuleBasedIntentClassifier()
    result = classifier.classify("Where can I find the report?")
    assert result.intent == IntentLabel.NAVIGATION


def test_fact_intent() -> None:
    classifier = RuleBasedIntentClassifier()
    result = classifier.classify("What is the sample size?")
    assert result.intent == IntentLabel.FACT


def test_limitation_intent() -> None:
    classifier = RuleBasedIntentClassifier()
    result = classifier.classify("What are the limitations of the study?")
    assert result.intent == IntentLabel.LIMITATION


def test_method_context_intent() -> None:
    classifier = RuleBasedIntentClassifier()
    result = classifier.classify("How was the data collected?")
    assert result.intent == IntentLabel.METHOD_CONTEXT


def test_unknown_intent() -> None:
    classifier = RuleBasedIntentClassifier()
    result = classifier.classify("Hello there")
    assert result.intent == IntentLabel.UNKNOWN


def test_trend_pattern_intent() -> None:
    classifier = RuleBasedIntentClassifier()
    result = classifier.classify("What trend do we see in enrollment over time?")
    assert result.intent == IntentLabel.TREND_PATTERN


def test_explanation_intent() -> None:
    classifier = RuleBasedIntentClassifier()
    result = classifier.classify("Why did mobility rates decrease?")
    assert result.intent == IntentLabel.EXPLANATION


def test_source_seeking_intent() -> None:
    classifier = RuleBasedIntentClassifier()
    result = classifier.classify("Which study reported this?")
    assert result.intent == IntentLabel.SOURCE_SEEKING


def test_interpretation_intent() -> None:
    classifier = RuleBasedIntentClassifier()
    result = classifier.classify("What do these findings imply for policy?")
    assert result.intent == IntentLabel.INTERPRETATION


def test_decision_support_intent() -> None:
    classifier = RuleBasedIntentClassifier()
    result = classifier.classify("Which intervention seems most effective to adopt?")
    assert result.intent == IntentLabel.DECISION_SUPPORT


def test_followup_intent() -> None:
    classifier = RuleBasedIntentClassifier()
    result = classifier.classify("What about in rural schools?")
    assert result.intent == IntentLabel.FOLLOWUP
