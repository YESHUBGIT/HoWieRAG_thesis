from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.core.schemas import Chunk
from howie_rag.retrieval.keyword_retriever import KeywordRetriever, retrieve_chunks


def test_retrieve_chunks_returns_best_matches_first() -> None:
    chunks = [
        Chunk(chunk_id="1", doc_id="doc-1", text="student mobility and financial aid", metadata={}),
        Chunk(chunk_id="2", doc_id="doc-1", text="doctoral careers and supervision", metadata={}),
        Chunk(chunk_id="3", doc_id="doc-2", text="student mobility across countries", metadata={}),
    ]

    matches = retrieve_chunks("student mobility", chunks, top_k=2)

    assert len(matches) == 2
    assert matches[0].chunk.chunk_id == "1"
    assert matches[0].score == 2
    assert matches[1].chunk.chunk_id == "3"


def test_retrieve_chunks_is_case_insensitive() -> None:
    chunks = [Chunk(chunk_id="1", doc_id="doc-1", text="Student Mobility Trends", metadata={})]

    matches = retrieve_chunks("student mobility", chunks)

    assert len(matches) == 1
    assert matches[0].score == 2


def test_retrieve_chunks_skips_zero_score_chunks() -> None:
    chunks = [Chunk(chunk_id="1", doc_id="doc-1", text="housing costs", metadata={})]

    assert retrieve_chunks("student mobility", chunks) == []


def test_retrieve_chunks_validates_top_k() -> None:
    with pytest.raises(ValueError):
        retrieve_chunks("query", [], top_k=0)


def test_keyword_retriever_class_matches_function_output() -> None:
    chunks = [Chunk(chunk_id="1", doc_id="doc-1", text="student mobility and funding", metadata={})]

    class_matches = KeywordRetriever().retrieve("student mobility", chunks, top_k=1)
    function_matches = retrieve_chunks("student mobility", chunks, top_k=1)

    assert class_matches == function_matches
