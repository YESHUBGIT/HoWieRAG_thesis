from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.chunking.text_chunker import chunk_document, chunk_documents
from howie_rag.core.schemas import Document


def test_chunk_document_splits_text_with_overlap() -> None:
    document = Document(doc_id="doc-1", title="study", text="abcdefghij", metadata={"source_path": "x"})

    chunks = chunk_document(document, chunk_size=4, overlap=1)

    assert [chunk.text for chunk in chunks] == ["abcd", "defg", "ghij"]
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1, 2]
    assert chunks[0].metadata["title"] == "study"
    assert chunks[0].doc_id == "doc-1"


def test_chunk_document_returns_no_chunks_for_empty_text() -> None:
    document = Document(doc_id="doc-1", title="empty", text="   ", metadata={})

    assert chunk_document(document) == []


def test_chunk_documents_combines_multiple_documents() -> None:
    documents = [
        Document(doc_id="doc-1", title="one", text="abcdef", metadata={}),
        Document(doc_id="doc-2", title="two", text="uvwxyz", metadata={}),
    ]

    chunks = chunk_documents(documents, chunk_size=3, overlap=0)

    assert [chunk.doc_id for chunk in chunks] == ["doc-1", "doc-1", "doc-2", "doc-2"]


def test_chunk_document_marks_table_chunks_and_preserves_headers() -> None:
    document = Document(
        doc_id="doc-1",
        title="finance",
        text=(
            "Revenue discussion before table.\n\n"
            "| year | revenue | cost |\n"
            "| 2022 | 10 | 4 |\n"
            "| 2023 | 12 | 5 |\n"
            "| 2024 | 14 | 6 |\n"
        ),
        metadata={"has_tables": True},
    )

    chunks = chunk_document(document, chunk_size=45, overlap=5)

    table_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "table"]
    assert table_chunks
    assert all(chunk.metadata["has_table_like_content"] is True for chunk in table_chunks)
    assert all("| year | revenue | cost |" in chunk.text for chunk in table_chunks)
    assert all(chunk.metadata["table_line_ratio"] > 0 for chunk in table_chunks)


def test_chunk_document_validates_arguments() -> None:
    document = Document(doc_id="doc-1", title="study", text="abcdef", metadata={})

    with pytest.raises(ValueError):
        chunk_document(document, chunk_size=0)

    with pytest.raises(ValueError):
        chunk_document(document, chunk_size=4, overlap=4)
