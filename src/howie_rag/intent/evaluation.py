import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from howie_rag.intent.base import BaseIntentClassifier
from howie_rag.intent.intent_labels import IntentLabel


@dataclass
class IntentExample:
    example_id: str
    question: str
    intent: str
    split: str
    language: str
    domain: str
    source_type_hint: str
    requires_context: bool


def _parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def load_intent_dataset(dataset_path: str, split: Optional[str] = None) -> List[IntentExample]:
    examples: List[IntentExample] = []

    with open(dataset_path, newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            if split is not None and row["split"] != split:
                continue

            examples.append(
                IntentExample(
                    example_id=row["id"],
                    question=row["question"],
                    intent=row["intent"],
                    split=row["split"],
                    language=row["language"],
                    domain=row["domain"],
                    source_type_hint=row["source_type_hint"],
                    requires_context=_parse_bool(row["requires_context"]),
                )
            )

    return examples


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _label_order() -> List[str]:
    return [label.value for label in IntentLabel]


def evaluate_classifier(
    classifier: BaseIntentClassifier, examples: Iterable[IntentExample]
) -> Dict[str, object]:
    example_list = list(examples)
    labels = _label_order()
    confusion_matrix: Dict[str, Dict[str, int]] = {
        actual: {predicted: 0 for predicted in labels} for actual in labels
    }

    correct_predictions = 0
    for example in example_list:
        prediction = classifier.classify(example.question)
        predicted_intent = prediction.intent

        if predicted_intent not in confusion_matrix[example.intent]:
            for actual_label in confusion_matrix:
                confusion_matrix[actual_label][predicted_intent] = 0
            labels.append(predicted_intent)

        confusion_matrix[example.intent][predicted_intent] += 1
        if predicted_intent == example.intent:
            correct_predictions += 1

    per_label: Dict[str, Dict[str, float]] = {}
    for label in labels:
        true_positive = confusion_matrix[label][label]
        false_positive = sum(
            confusion_matrix[other_label].get(label, 0)
            for other_label in labels
            if other_label != label
        )
        false_negative = sum(
            count for predicted_label, count in confusion_matrix[label].items() if predicted_label != label
        )

        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        support = sum(confusion_matrix[label].values())

        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }

    macro_precision = _safe_divide(
        sum(metrics["precision"] for metrics in per_label.values()), len(per_label)
    )
    macro_recall = _safe_divide(
        sum(metrics["recall"] for metrics in per_label.values()), len(per_label)
    )
    macro_f1 = _safe_divide(sum(metrics["f1"] for metrics in per_label.values()), len(per_label))
    accuracy = _safe_divide(correct_predictions, len(example_list))

    return {
        "total_examples": len(example_list),
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "labels": labels,
        "per_label": per_label,
        "confusion_matrix": confusion_matrix,
    }


def _format_confusion_matrix(labels: List[str], confusion_matrix: Dict[str, Dict[str, int]]) -> str:
    header = ["actual\\pred"] + labels
    rows = [header]
    for actual_label in labels:
        row = [actual_label]
        for predicted_label in labels:
            row.append(str(confusion_matrix[actual_label].get(predicted_label, 0)))
        rows.append(row)

    column_widths = [max(len(row[column_index]) for row in rows) for column_index in range(len(header))]

    lines = []
    for row in rows:
        cells = [cell.ljust(column_widths[index]) for index, cell in enumerate(row)]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def format_evaluation_report(
    dataset_path: str, split: Optional[str], results: Dict[str, object]
) -> str:
    labels = results["labels"]
    per_label = results["per_label"]
    confusion_matrix = results["confusion_matrix"]

    lines = [
        f"Dataset: {Path(dataset_path)}",
        f"Split: {split or 'all'}",
        f"Total examples: {results['total_examples']}",
        f"Accuracy: {results['accuracy']:.4f}",
        f"Macro precision: {results['macro_precision']:.4f}",
        f"Macro recall: {results['macro_recall']:.4f}",
        f"Macro F1: {results['macro_f1']:.4f}",
        "",
        "Per-label metrics:",
    ]

    for label in labels:
        metrics = per_label[label]
        lines.append(
            (
                f"- {label}: precision={metrics['precision']:.4f}, "
                f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, "
                f"support={int(metrics['support'])}"
            )
        )

    lines.extend(
        [
            "",
            "Confusion matrix:",
            _format_confusion_matrix(labels, confusion_matrix),
        ]
    )
    return "\n".join(lines)
