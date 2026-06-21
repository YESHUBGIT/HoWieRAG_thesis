import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from howie_rag.core.schemas import Document
from howie_rag.core.utils import stable_id
from howie_rag.datasets.schemas import BenchmarkQARecord, SourceDocumentRecord
from howie_rag.document_classification import classify_document_content


DATASET_NAME = "UltraDomain"


def _jsonl_files(path: str) -> List[Path]:
    candidate = Path(path)
    if candidate.is_file():
        return [candidate]
    if candidate.is_dir():
        return sorted(file_path for file_path in candidate.glob("*.jsonl") if file_path.is_file())
    raise FileNotFoundError(f"UltraDomain path not found: {path}")


def _domain_from_row_or_file(row: dict, file_path: Path) -> str:
    label = row.get("label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    return file_path.stem


def _normalize_answers(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item).strip()]
    return [str(value)]


def _row_question(row: dict) -> str:
    for key in ("input", "question"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _context_id(row: dict, context_text: str) -> str:
    value = row.get("context_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return stable_id(context_text)


def _title_and_authors(meta: dict) -> Tuple[str, str]:
    if not isinstance(meta, dict):
        return "", ""
    title = meta.get("title") if isinstance(meta.get("title"), str) else ""
    authors = meta.get("authors") if isinstance(meta.get("authors"), str) else ""
    return title, authors


def _source_metadata(row: dict, file_path: Path, context_id: str, title: str, authors: str, domain: str) -> dict:
    metadata = {
        "dataset_name": DATASET_NAME,
        "domain": domain,
        "context_id": context_id,
        "source_file": file_path.name,
    }
    if title:
        metadata["title"] = title
    if authors:
        metadata["authors"] = authors
    if "meta" in row:
        metadata["meta"] = row.get("meta")
    if "length" in row:
        metadata["length"] = row.get("length")
    if "_id" in row:
        metadata["row_id"] = row.get("_id")
    return metadata


def _qa_metadata(row: dict, file_path: Path, context_id: str, domain: str) -> dict:
    metadata = {
        "dataset_name": DATASET_NAME,
        "domain": domain,
        "context_id": context_id,
        "source_file": file_path.name,
    }
    if "meta" in row:
        metadata["meta"] = row.get("meta")
    if "_id" in row:
        metadata["row_id"] = row.get("_id")
    return metadata


def _iter_rows(path: str) -> Iterable[Tuple[Path, dict]]:
    for file_path in _jsonl_files(path):
        with file_path.open(encoding="utf-8") as file_handle:
            for line in file_handle:
                stripped = line.strip()
                if not stripped:
                    continue
                yield file_path, json.loads(stripped)


def load_ultradomain_source_documents(path: str) -> List[SourceDocumentRecord]:
    source_records_by_id: Dict[str, SourceDocumentRecord] = {}

    for file_path, row in _iter_rows(path):
        context_text = row.get("context")
        if not isinstance(context_text, str) or not context_text.strip():
            continue

        domain = _domain_from_row_or_file(row, file_path)
        context_id = _context_id(row, context_text)
        title, authors = _title_and_authors(row.get("meta", {}))
        doc_id = stable_id(f"{DATASET_NAME}:{domain}:{context_id}")

        if doc_id in source_records_by_id:
            continue

        source_records_by_id[doc_id] = SourceDocumentRecord(
            doc_id=doc_id,
            text=context_text,
            title=title,
            source_type="ultradomain_context",
            dataset_name=DATASET_NAME,
            domain=domain,
            context_id=context_id,
            metadata=_source_metadata(row, file_path, context_id, title, authors, domain),
        )

    return list(source_records_by_id.values())


def load_ultradomain_benchmark_records(path: str) -> List[BenchmarkQARecord]:
    source_records = load_ultradomain_source_documents(path)
    doc_id_by_context = {record.context_id: record.doc_id for record in source_records}

    qa_records: List[BenchmarkQARecord] = []

    for file_path, row in _iter_rows(path):
        question = _row_question(row)
        answers = _normalize_answers(row.get("answers"))
        context_text = row.get("context")
        if not question or not answers or not isinstance(context_text, str) or not context_text.strip():
            continue

        domain = _domain_from_row_or_file(row, file_path)
        context_id = _context_id(row, context_text)
        gold_doc_id = doc_id_by_context.get(context_id, stable_id(f"{DATASET_NAME}:{domain}:{context_id}"))
        row_identifier = row.get("_id") if row.get("_id") is not None else question
        question_id = stable_id(f"{DATASET_NAME}:{domain}:{context_id}:{row_identifier}")

        qa_records.append(
            BenchmarkQARecord(
                question_id=question_id,
                question=question,
                gold_answers=answers,
                gold_doc_id=gold_doc_id,
                gold_context_id=context_id,
                domain=domain,
                dataset_name=DATASET_NAME,
                metadata=_qa_metadata(row, file_path, context_id, domain),
            )
        )

    return qa_records


def source_records_to_documents(source_records: List[SourceDocumentRecord]) -> List[Document]:
    documents: List[Document] = []

    for record in source_records:
        title = record.title or f"{record.domain}_{record.context_id}"
        classification = classify_document_content(title=title, text=record.text, file_type="jsonl")
        documents.append(
            Document(
                doc_id=record.doc_id,
                title=title,
                text=record.text,
                metadata={
                    "source_type": record.source_type,
                    "dataset_name": record.dataset_name,
                    "domain": record.domain,
                    "context_id": record.context_id,
                    "original_source_file": record.metadata.get("source_file", ""),
                    **record.metadata,
                    **classification,
                },
            )
        )

    return documents


def load_ultradomain_documents(path: str) -> List[Document]:
    return source_records_to_documents(load_ultradomain_source_documents(path))
