from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.datasets.ultradomain import (
    load_ultradomain_benchmark_records,
    load_ultradomain_source_documents,
)


def test_load_ultradomain_source_documents_from_single_file(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "cs.jsonl"
    rows = [
        {
            "input": "What is Spark Streaming?",
            "answers": ["A streaming library for Spark."],
            "context": "Spark Streaming extends Spark for real-time processing.",
            "context_id": "ctx-1",
            "label": "cs",
            "meta": {"title": "Machine Learning with Spark", "authors": "Nick Pentreath"},
        },
        {
            "input": "Another question",
            "answers": ["Another answer"],
            "context": "Spark Streaming extends Spark for real-time processing.",
            "context_id": "ctx-1",
            "label": "cs",
            "meta": {"title": "Machine Learning with Spark", "authors": "Nick Pentreath"},
        },
    ]
    jsonl_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    records = load_ultradomain_source_documents(str(jsonl_path))

    assert len(records) == 1
    assert records[0].context_id == "ctx-1"
    assert records[0].domain == "cs"
    assert records[0].title == "Machine Learning with Spark"
    assert records[0].metadata["authors"] == "Nick Pentreath"
    assert records[0].metadata["source_title"] == "Machine Learning with Spark"
    assert records[0].metadata["source_domain"] == "cs"


def test_load_ultradomain_benchmark_records_normalizes_answers_and_question_fallback(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "mix.jsonl"
    rows = [
        {
            "question": "What is the main idea?",
            "answers": "The main idea is explained in the passage.",
            "context": "A passage about the main idea.",
            "label": "mix",
            "meta": {},
        },
        {
            "input": "Should be skipped because answers missing",
            "context": "No answers here.",
            "label": "mix",
        },
    ]
    jsonl_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    records = load_ultradomain_benchmark_records(str(jsonl_path))

    assert len(records) == 1
    assert records[0].question == "What is the main idea?"
    assert records[0].gold_answers == ["The main idea is explained in the passage."]
    assert records[0].gold_context_id
    assert records[0].gold_doc_id


def test_load_ultradomain_source_documents_from_folder_uses_filename_as_domain(tmp_path: Path) -> None:
    first_file = tmp_path / "history.jsonl"
    second_file = tmp_path / "physics.jsonl"
    first_file.write_text(
        json.dumps({"input": "Q1", "answers": ["A1"], "context": "History context."}),
        encoding="utf-8",
    )
    second_file.write_text(
        json.dumps({"input": "Q2", "answers": ["A2"], "context": "Physics context."}),
        encoding="utf-8",
    )

    records = load_ultradomain_source_documents(str(tmp_path))

    assert len(records) == 2
    assert {record.domain for record in records} == {"history", "physics"}
