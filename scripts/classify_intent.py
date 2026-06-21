from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.intent.rule_based import RuleBasedIntentClassifier


def main() -> int:
    if len(sys.argv) != 2:
        print('Usage: python scripts/classify_intent.py "Your question here"')
        return 1

    question = sys.argv[1]
    classifier = RuleBasedIntentClassifier()
    result = classifier.classify(question)

    print(f"Intent: {result.intent}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
