from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.intent.evaluation import (
    evaluate_classifier,
    format_evaluation_report,
    load_intent_dataset,
)
from howie_rag.intent.rule_based import RuleBasedIntentClassifier


DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "intent_dataset"
    / "v3"
    / "howie_intent_dataset_390_13intents_v3.csv"
)


def main() -> int:
    dataset_path = str(DEFAULT_DATASET_PATH)
    split = "test"

    if len(sys.argv) > 3:
        print(
            "Usage: python scripts/evaluate_rule_based_intent.py "
            '[dataset_csv_path] [train|dev|test|all]'
        )
        return 1

    if len(sys.argv) >= 2:
        dataset_path = sys.argv[1]
    if len(sys.argv) == 3:
        split = sys.argv[2]

    selected_split = None if split == "all" else split
    examples = load_intent_dataset(dataset_path, split=selected_split)
    classifier = RuleBasedIntentClassifier()
    results = evaluate_classifier(classifier, examples)
    print(format_evaluation_report(dataset_path, selected_split, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
