from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.core.schemas import IntentResult
from howie_rag.intent.base import BaseIntentClassifier
from howie_rag.intent.evaluation import IntentExample, evaluate_classifier, load_intent_dataset


class StubClassifier(BaseIntentClassifier):
    def __init__(self, predictions: dict[str, str]) -> None:
        self.predictions = predictions

    def classify(self, question: str) -> IntentResult:
        return IntentResult(intent=self.predictions[question], confidence=1.0)


def test_load_intent_dataset_v3_test_split() -> None:
    dataset_path = (
        Path(__file__).resolve().parents[1]
        / "intent_dataset"
        / "v3"
        / "howie_intent_dataset_390_13intents_v3.csv"
    )
    examples = load_intent_dataset(str(dataset_path), split="test")

    assert len(examples) == 65
    assert all(example.split == "test" for example in examples)


def test_evaluate_classifier_computes_expected_metrics() -> None:
    examples = [
        IntentExample(
            example_id="1",
            question="q1",
            intent="FACT",
            split="test",
            language="en",
            domain="higher_education_research",
            source_type_hint="mixed",
            requires_context=False,
        ),
        IntentExample(
            example_id="2",
            question="q2",
            intent="SUMMARY",
            split="test",
            language="en",
            domain="higher_education_research",
            source_type_hint="mixed",
            requires_context=False,
        ),
    ]
    classifier = StubClassifier({"q1": "FACT", "q2": "FACT"})

    results = evaluate_classifier(classifier, examples)

    assert results["total_examples"] == 2
    assert results["accuracy"] == 0.5
    assert results["per_label"]["FACT"]["precision"] == 0.5
    assert results["per_label"]["FACT"]["recall"] == 1.0
    assert results["per_label"]["SUMMARY"]["recall"] == 0.0
