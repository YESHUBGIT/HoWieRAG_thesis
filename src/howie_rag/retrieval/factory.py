from howie_rag.retrieval.base import BaseRetriever
from howie_rag.retrieval.bm25_retriever import BM25Retriever
from howie_rag.retrieval.keyword_retriever import KeywordRetriever


def create_retriever(name: str) -> BaseRetriever:
    normalized_name = name.lower()
    if normalized_name == "keyword":
        return KeywordRetriever()
    if normalized_name == "bm25":
        return BM25Retriever()
    raise ValueError(f"Unsupported retriever: {name}")
