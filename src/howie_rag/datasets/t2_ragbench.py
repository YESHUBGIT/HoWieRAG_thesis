import ast
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from howie_rag.core.schemas import Document
from howie_rag.core.utils import stable_id
from howie_rag.datasets.normalization import normalize_source_metadata
from howie_rag.datasets.schemas import BenchmarkQARecord, SourceDocumentRecord
from howie_rag.document_classification import classify_document_content


DATASET_NAME = "T2-RAGBench"
_OPTIONAL_METADATA_FIELDS = {
    "file_name",
    "company_name",
    "company_symbol",
    "report_year",
    "page_number",
    "company_sector",
    "company_industry",
    "company_headquarters",
    "company_date_added",
    "company_cik",
    "company_founded",
    "table",
    "pre_text",
    "post_text",
}


def _data_root(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_dir() and (candidate / "data").is_dir():
        return candidate / "data"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"T2-RAGBench path not found: {path}")


def _iter_dataset_files(path: str) -> Iterable[Tuple[str, str, Path]]:
    data_root = _data_root(path)
    for file_path in sorted(data_root.rglob("*.jsonl")):
        if file_path.name == "metadata.jsonl":
            yield file_path.parent.parent.name, file_path.parent.name, file_path
        elif file_path.name == "turn_0.jsonl":
            yield file_path.parent.name, "turn_0", file_path


def _normalized_filter_set(values: Optional[List[str]]) -> Optional[set[str]]:
    if not values:
        return None
    normalized = {value.strip() for value in values if isinstance(value, str) and value.strip()}
    return normalized or None


def _matches_filters(subset: str, split: str, subsets: Optional[set[str]], splits: Optional[set[str]]) -> bool:
    if subsets is not None and subset not in subsets:
        return False
    if splits is not None and split not in splits:
        return False
    return True


def _normalize_answers(*values: object) -> List[str]:
    answers: List[str] = []

    def add_answer(value: object) -> None:
        if value is None:
            return
        if isinstance(value, list):
            for item in value:
                add_answer(item)
            return
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped or stripped.lower() == "null":
                return
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = ast.literal_eval(stripped)
                except (ValueError, SyntaxError):
                    parsed = None
                if isinstance(parsed, list):
                    add_answer(parsed)
                    return
            answers.append(stripped)
            return
        answers.append(str(value))

    for value in values:
        add_answer(value)

    deduplicated_answers: List[str] = []
    seen = set()
    for answer in answers:
        if answer in seen:
            continue
        seen.add(answer)
        deduplicated_answers.append(answer)
    return deduplicated_answers


def _source_title(row: dict, context_id: str) -> str:
    file_name = row.get("file_name")
    if isinstance(file_name, str) and file_name.strip():
        return Path(file_name).stem
    company_name = row.get("company_name")
    report_year = row.get("report_year")
    if isinstance(company_name, str) and company_name.strip():
        if isinstance(report_year, str) and report_year.strip():
            return f"{company_name}_{report_year}"
        return company_name
    return context_id


def _source_metadata(row: dict, subset: str, split: str, context_id: str) -> dict:
    metadata = {
        "dataset_name": DATASET_NAME,
        "subset": subset,
        "split": split,
        "context_id": context_id,
        **normalize_source_metadata(
            dataset_name=DATASET_NAME,
            domain=subset,
            context_id=context_id,
            title=_source_title(row, context_id),
            source_file=str(row.get("file_name") or ""),
            subset=subset,
            split=split,
            source_year=str(row.get("report_year") or ""),
            source_entity=str(row.get("company_name") or ""),
            source_page_number=str(row.get("page_number") or ""),
        ),
    }
    for field_name in _OPTIONAL_METADATA_FIELDS:
        if field_name in row and row[field_name] is not None:
            metadata[field_name] = row[field_name]
    metadata["has_explicit_table"] = bool(row.get("table")) or "|" in str(row.get("context", ""))
    return metadata


def _qa_metadata(row: dict, subset: str, split: str, context_id: str) -> dict:
    metadata = {
        "dataset_name": DATASET_NAME,
        "subset": subset,
        "split": split,
        "context_id": context_id,
        **normalize_source_metadata(
            dataset_name=DATASET_NAME,
            domain=subset,
            context_id=context_id,
            title=_source_title(row, context_id),
            source_file=str(row.get("file_name") or ""),
            subset=subset,
            split=split,
            source_year=str(row.get("report_year") or ""),
            source_entity=str(row.get("company_name") or ""),
            source_page_number=str(row.get("page_number") or ""),
        ),
    }
    for field_name in _OPTIONAL_METADATA_FIELDS:
        if field_name in row and row[field_name] is not None:
            metadata[field_name] = row[field_name]
    return metadata


def _iter_rows(
    path: str,
    subsets: Optional[List[str]] = None,
    splits: Optional[List[str]] = None,
) -> Iterator[Tuple[str, str, Path, dict]]:
    normalized_subsets = _normalized_filter_set(subsets)
    normalized_splits = _normalized_filter_set(splits)
    for subset, split, file_path in _iter_dataset_files(path):
        if not _matches_filters(subset, split, normalized_subsets, normalized_splits):
            continue
        with file_path.open(encoding="utf-8") as file_handle:
            for line in file_handle:
                stripped = line.strip()
                if not stripped:
                    continue
                yield subset, split, file_path, json.loads(stripped)


def iter_t2_ragbench_source_documents(
    path: str,
    subsets: Optional[List[str]] = None,
    splits: Optional[List[str]] = None,
) -> Iterator[SourceDocumentRecord]:
    seen_doc_ids = set()

    for subset, split, _file_path, row in _iter_rows(path, subsets=subsets, splits=splits):
        context_text = row.get("context")
        context_id = row.get("context_id")
        if not isinstance(context_text, str) or not context_text.strip():
            continue
        if not isinstance(context_id, str) or not context_id.strip():
            continue

        doc_id = stable_id(f"{DATASET_NAME}:{subset}:{context_id}")
        if doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)

        yield SourceDocumentRecord(
            doc_id=doc_id,
            text=context_text,
            title=_source_title(row, context_id),
            source_type="t2_ragbench_context",
            dataset_name=DATASET_NAME,
            domain=subset,
            context_id=context_id,
            metadata=_source_metadata(row, subset, split, context_id),
        )


def load_t2_ragbench_source_documents(
    path: str,
    subsets: Optional[List[str]] = None,
    splits: Optional[List[str]] = None,
) -> List[SourceDocumentRecord]:
    return list(iter_t2_ragbench_source_documents(path, subsets=subsets, splits=splits))


def iter_t2_ragbench_benchmark_records(
    path: str,
    subsets: Optional[List[str]] = None,
    splits: Optional[List[str]] = None,
) -> Iterator[BenchmarkQARecord]:
    doc_id_by_context: Dict[Tuple[str, str], str] = {
        (record.domain, record.context_id): record.doc_id
        for record in iter_t2_ragbench_source_documents(path, subsets=subsets, splits=splits)
    }

    for subset, split, _file_path, row in _iter_rows(path, subsets=subsets, splits=splits):
        question = row.get("question")
        context_id = row.get("context_id")
        question_id = row.get("id")
        if not isinstance(question, str) or not question.strip():
            continue
        if not isinstance(context_id, str) or not context_id.strip():
            continue

        answers = _normalize_answers(row.get("program_answer"), row.get("original_answer"))
        if not answers:
            continue

        normalized_question_id = question_id if isinstance(question_id, str) and question_id.strip() else stable_id(
            f"{DATASET_NAME}:{subset}:{context_id}:{question}"
        )
        gold_doc_id = doc_id_by_context.get((subset, context_id), stable_id(f"{DATASET_NAME}:{subset}:{context_id}"))

        yield BenchmarkQARecord(
            question_id=normalized_question_id,
            question=question.strip(),
            gold_answers=answers,
            gold_doc_id=gold_doc_id,
            gold_context_id=context_id,
            domain=subset,
            dataset_name=DATASET_NAME,
            metadata=_qa_metadata(row, subset, split, context_id),
        )


def load_t2_ragbench_benchmark_records(
    path: str,
    subsets: Optional[List[str]] = None,
    splits: Optional[List[str]] = None,
) -> List[BenchmarkQARecord]:
    return list(iter_t2_ragbench_benchmark_records(path, subsets=subsets, splits=splits))


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
                    **record.metadata,
                    **classification,
                },
            )
        )

    return documents


def load_t2_ragbench_documents(
    path: str,
    subsets: Optional[List[str]] = None,
    splits: Optional[List[str]] = None,
) -> List[Document]:
    return source_records_to_documents(load_t2_ragbench_source_documents(path, subsets=subsets, splits=splits))
