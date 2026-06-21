from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.ingestion import text_loader
from howie_rag.ingestion.text_loader import load_text_documents


def test_load_text_documents_reads_supported_files(tmp_path: Path) -> None:
    (tmp_path / "report.txt").write_text("Report body", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# Notes", encoding="utf-8")
    (tmp_path / "ignore.json").write_text('{"a": 1}', encoding="utf-8")

    documents = load_text_documents(str(tmp_path))

    assert len(documents) == 2
    assert [document.title for document in documents] == ["notes", "report"]
    assert documents[0].metadata["document_type"] == "narrative"
    assert documents[0].metadata["file_type"] == "md"
    assert documents[1].metadata["file_type"] == "txt"


def test_load_text_documents_creates_stable_document_ids(tmp_path: Path) -> None:
    file_path = tmp_path / "study.txt"
    file_path.write_text("Study results", encoding="utf-8")

    first_load = load_text_documents(str(tmp_path))
    second_load = load_text_documents(str(tmp_path))

    assert first_load[0].doc_id == second_load[0].doc_id
    assert first_load[0].text == "Study results"


def test_load_text_documents_reads_pdf_files_with_extracted_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "methods.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(text_loader, "_read_pdf_text", lambda _: "[Page 1]\nMethod report text")

    documents = load_text_documents(str(tmp_path))

    assert len(documents) == 1
    assert documents[0].title == "methods"
    assert documents[0].metadata["document_type"] == "narrative"
    assert documents[0].metadata["file_type"] == "pdf"
    assert documents[0].text == "[Page 1]\nMethod report text"
