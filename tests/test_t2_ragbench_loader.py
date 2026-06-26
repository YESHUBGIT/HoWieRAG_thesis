from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.datasets.t2_ragbench import (
    load_t2_ragbench_benchmark_records,
    load_t2_ragbench_source_documents,
    source_records_to_documents,
)


def test_load_t2_ragbench_source_documents_deduplicates_contexts(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "FinQA" / "test"
    data_dir.mkdir(parents=True)
    rows = [
        {
            "id": "finqa_test_0",
            "context_id": "finqa_ctx_1",
            "split": "test",
            "question": "What is the 2019 revenue?",
            "program_answer": "10.0",
            "original_answer": "10",
            "context": "Revenue was 10 in 2019.\n| year | revenue |\n| 2019 | 10 |",
            "file_name": "raw/acme_2019.pdf",
            "company_name": "Acme",
            "report_year": "2019",
            "table": "| year | revenue |\n| 2019 | 10 |",
        },
        {
            "id": "finqa_test_1",
            "context_id": "finqa_ctx_1",
            "split": "test",
            "question": "What is the 2019 revenue again?",
            "program_answer": "10.0",
            "original_answer": "10",
            "context": "Revenue was 10 in 2019.\n| year | revenue |\n| 2019 | 10 |",
            "file_name": "raw/acme_2019.pdf",
        },
    ]
    (data_dir / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    records = load_t2_ragbench_source_documents(str(tmp_path))

    assert len(records) == 1
    assert records[0].context_id == "finqa_ctx_1"
    assert records[0].domain == "FinQA"
    assert records[0].title == "acme_2019"
    assert records[0].metadata["subset"] == "FinQA"
    assert records[0].metadata["split"] == "test"
    assert records[0].metadata["has_explicit_table"] is True
    assert records[0].metadata["source_title"] == "acme_2019"
    assert records[0].metadata["source_year"] == "2019"
    assert records[0].metadata["source_entity"] == "Acme"
    assert records[0].metadata["source_file"] == "raw/acme_2019.pdf"


def test_load_t2_ragbench_benchmark_records_normalizes_answers_and_supports_turn0(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "ConvFinQA"
    data_dir.mkdir(parents=True)
    rows = [
        {
            "id": "convfinqa_0",
            "context_id": "conv_ctx_1",
            "split": "all",
            "question": "What was revenue?",
            "program_answer": "206588.0",
            "original_answer": "['206588']",
            "context": "Revenue was 206588.",
            "file_name": "raw/jh_2009.pdf",
        }
    ]
    (data_dir / "turn_0.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    records = load_t2_ragbench_benchmark_records(str(tmp_path))

    assert len(records) == 1
    assert records[0].question_id == "convfinqa_0"
    assert records[0].domain == "ConvFinQA"
    assert records[0].gold_answers == ["206588.0", "206588"]
    assert records[0].metadata["subset"] == "ConvFinQA"
    assert records[0].metadata["split"] == "turn_0"
    assert records[0].metadata["source_subset"] == "ConvFinQA"
    assert records[0].metadata["source_file"] == "raw/jh_2009.pdf"


def test_load_t2_ragbench_records_support_subset_and_split_filters(tmp_path: Path) -> None:
    finqa_test_dir = tmp_path / "data" / "FinQA" / "test"
    finqa_test_dir.mkdir(parents=True)
    tatqa_dev_dir = tmp_path / "data" / "TAT-DQA" / "dev"
    tatqa_dev_dir.mkdir(parents=True)

    (finqa_test_dir / "metadata.jsonl").write_text(
        json.dumps(
            {
                "id": "finqa_test_0",
                "context_id": "finqa_ctx_1",
                "split": "test",
                "question": "What is the 2019 revenue?",
                "program_answer": "10.0",
                "original_answer": "10",
                "context": "Revenue was 10 in 2019.",
                "file_name": "raw/acme_2019.pdf",
            }
        ),
        encoding="utf-8",
    )
    (tatqa_dev_dir / "metadata.jsonl").write_text(
        json.dumps(
            {
                "id": "tatqa_dev_0",
                "context_id": "tatqa_ctx_1",
                "split": "dev",
                "question": "What is the 2020 revenue?",
                "program_answer": "12.0",
                "original_answer": "12",
                "context": "Revenue was 12 in 2020.",
                "file_name": "raw/beta_2020.pdf",
            }
        ),
        encoding="utf-8",
    )

    filtered_records = load_t2_ragbench_benchmark_records(str(tmp_path), subsets=["FinQA"], splits=["test"])

    assert len(filtered_records) == 1
    assert filtered_records[0].domain == "FinQA"
    assert filtered_records[0].metadata["split"] == "test"


def test_t2_ragbench_source_records_to_documents_preserve_metadata() -> None:
    records = load_t2_ragbench_source_documents  # keeps import used in this test module
    assert records is not None

    from howie_rag.datasets.schemas import SourceDocumentRecord

    source_records = [
        SourceDocumentRecord(
            doc_id="doc-1",
            text="Revenue was 10 in 2019.\n| year | revenue |\n| 2019 | 10 |",
            title="acme_2019",
            source_type="t2_ragbench_context",
            dataset_name="T2-RAGBench",
            domain="FinQA",
            context_id="finqa_ctx_1",
            metadata={"subset": "FinQA", "split": "test", "has_explicit_table": True},
        )
    ]

    documents = source_records_to_documents(source_records)

    assert len(documents) == 1
    assert documents[0].metadata["dataset_name"] == "T2-RAGBench"
    assert documents[0].metadata["subset"] == "FinQA"
    assert documents[0].metadata["has_explicit_table"] is True
    assert documents[0].metadata["source_title"] == "acme_2019"
    assert "document_type" in documents[0].metadata
