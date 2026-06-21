from dataclasses import dataclass
from typing import List, Optional

from howie_rag.chunking.text_chunker import chunk_documents
from howie_rag.ingestion.text_loader import load_text_documents
from howie_rag.intent.rule_based import RuleBasedIntentClassifier
from howie_rag.retrieval.base import RetrievalMatch
from howie_rag.retrieval.factory import create_retriever


@dataclass
class ChatbotResponse:
    question: str
    intent: str
    confidence: float
    reasoning: Optional[str]
    retrieval_matches: List[RetrievalMatch]


def run_simple_chatbot(
    question: str,
    documents_dir: str,
    chunk_size: int = 300,
    overlap: int = 50,
    top_k: int = 3,
    retriever_name: str = "keyword",
) -> ChatbotResponse:
    classifier = RuleBasedIntentClassifier()
    intent_result = classifier.classify(question)

    documents = load_text_documents(documents_dir)
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    retriever = create_retriever(retriever_name)
    retrieval_matches = retriever.retrieve(question, chunks, top_k=top_k)

    return ChatbotResponse(
        question=question,
        intent=intent_result.intent,
        confidence=intent_result.confidence,
        reasoning=intent_result.reasoning,
        retrieval_matches=retrieval_matches,
    )
