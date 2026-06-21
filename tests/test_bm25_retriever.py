from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.core.schemas import Chunk
from howie_rag.retrieval.bm25_retriever import BM25Retriever
from howie_rag.retrieval.factory import create_retriever


def test_bm25_retriever_returns_best_matches_first() -> None:
    chunks = [
        Chunk(chunk_id="1", doc_id="doc-1", text="student mobility and financial aid", metadata={}),
        Chunk(chunk_id="2", doc_id="doc-1", text="doctoral careers and supervision", metadata={}),
        Chunk(chunk_id="3", doc_id="doc-2", text="student mobility across countries", metadata={}),
    ]

    matches = BM25Retriever().retrieve("student mobility", chunks, top_k=2)

    assert len(matches) == 2
    assert matches[0].chunk.chunk_id in {"1", "3"}
    assert matches[0].score >= matches[1].score


def test_bm25_retriever_skips_zero_score_chunks() -> None:
    chunks = [Chunk(chunk_id="1", doc_id="doc-1", text="housing costs", metadata={})]

    assert BM25Retriever().retrieve("student mobility", chunks) == []


def test_bm25_retriever_validates_top_k() -> None:
    with pytest.raises(ValueError):
        BM25Retriever().retrieve("query", [], top_k=0)


def test_create_retriever_supports_keyword_and_bm25() -> None:
    assert create_retriever("keyword").__class__.__name__ == "KeywordRetriever"
    assert create_retriever("bm25").__class__.__name__ == "BM25Retriever"

    with pytest.raises(ValueError):
        create_retriever("unknown")
