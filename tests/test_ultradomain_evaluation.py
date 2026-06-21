from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.chunking.text_chunker import chunk_documents
from howie_rag.datasets.schemas import BenchmarkQARecord, SourceDocumentRecord
from howie_rag.datasets.ultradomain import source_records_to_documents
from howie_rag.evaluation.ultradomain_retrieval import evaluate_ultradomain_retrieval


def test_source_records_to_documents_preserve_ultradomain_metadata() -> None:
    source_records = [
        SourceDocumentRecord(
            doc_id="doc-1",
            text="Student mobility increased over time.",
            title="Mobility report",
            source_type="ultradomain_context",
            dataset_name="UltraDomain",
            domain="education",
            context_id="ctx-1",
            metadata={"source_file": "education.jsonl"},
        )
    ]

    documents = source_records_to_documents(source_records)

    assert len(documents) == 1
    assert documents[0].metadata["dataset_name"] == "UltraDomain"
    assert documents[0].metadata["domain"] == "education"
    assert documents[0].metadata["context_id"] == "ctx-1"
    assert documents[0].metadata["source_type"] == "ultradomain_context"
    assert documents[0].metadata["original_source_file"] == "education.jsonl"


def test_chunk_documents_preserve_ultradomain_metadata() -> None:
    source_records = [
        SourceDocumentRecord(
            doc_id="doc-1",
            text="Student mobility increased over time. Financial aid also increased over time.",
            title="Mobility report",
            source_type="ultradomain_context",
            dataset_name="UltraDomain",
            domain="education",
            context_id="ctx-1",
            metadata={"source_file": "education.jsonl"},
        )
    ]

    documents = source_records_to_documents(source_records)
    chunks = chunk_documents(documents, chunk_size=40, overlap=5)

    assert chunks
    assert chunks[0].metadata["domain"] == "education"
    assert chunks[0].metadata["dataset_name"] == "UltraDomain"
    assert chunks[0].metadata["source_type"] == "ultradomain_context"
    assert chunks[0].metadata["context_id"] == "ctx-1"
    assert "document_type" in chunks[0].metadata
    assert "has_tables" in chunks[0].metadata
    assert "has_figures" in chunks[0].metadata
    assert chunks[0].metadata["original_source_file"] == "education.jsonl"


def test_evaluate_ultradomain_retrieval_computes_metrics() -> None:
    source_records = [
        SourceDocumentRecord(
            doc_id="doc-1",
            text="Student mobility increased over time.",
            title="Mobility report",
            source_type="ultradomain_context",
            dataset_name="UltraDomain",
            domain="education",
            context_id="ctx-1",
            metadata={"source_file": "education.jsonl"},
        ),
        SourceDocumentRecord(
            doc_id="doc-2",
            text="Cooking pasta requires boiling water.",
            title="Cooking note",
            source_type="ultradomain_context",
            dataset_name="UltraDomain",
            domain="cooking",
            context_id="ctx-2",
            metadata={"source_file": "cooking.jsonl"},
        ),
    ]
    documents = source_records_to_documents(source_records)
    chunks = chunk_documents(documents, chunk_size=80, overlap=0)
    questions = [
        BenchmarkQARecord(
            question_id="q-1",
            question="What happened to student mobility?",
            gold_answers=["It increased over time."],
            gold_doc_id="doc-1",
            gold_context_id="ctx-1",
            domain="education",
            dataset_name="UltraDomain",
            metadata={},
        )
    ]

    results = evaluate_ultradomain_retrieval(questions, chunks, retriever_name="keyword")

    assert results["question_count"] == 1
    assert results["hit_at_1"] == 1.0
    assert results["hit_at_5"] == 1.0
    assert results["mrr_at_5"] == 1.0
    assert results["precision_at_1"] == 1.0
