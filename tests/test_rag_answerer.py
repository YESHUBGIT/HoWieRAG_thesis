from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.answering.rag_answerer import answer_with_rag, build_rag_user_prompt
from howie_rag.core.schemas import Chunk
from howie_rag.llm.base import BaseLLMClient
from howie_rag.retrieval.keyword_retriever import RetrievalMatch


class StubLLMClient(BaseLLMClient):
    def __init__(self) -> None:
        self.calls = []

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 400) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return "Stub answer"


def _build_match(title: str, text: str, score: int = 2) -> RetrievalMatch:
    return RetrievalMatch(
        chunk=Chunk(chunk_id=title, doc_id="doc-1", text=text, metadata={"title": title}),
        score=score,
    )


def test_build_rag_user_prompt_includes_sources_and_intent() -> None:
    prompt = build_rag_user_prompt(
        question="What trend do we see?",
        intent="TREND_PATTERN",
        retrieval_matches=[_build_match("mobility", "Student mobility increased.")],
    )

    assert "What trend do we see?" in prompt
    assert "TREND_PATTERN" in prompt
    assert "[Source 1: mobility]" in prompt
    assert "Student mobility increased." in prompt


def test_answer_with_rag_calls_llm_client() -> None:
    llm_client = StubLLMClient()
    answer = answer_with_rag(
        llm_client=llm_client,
        question="What trend do we see?",
        intent="TREND_PATTERN",
        retrieval_matches=[_build_match("mobility", "Student mobility increased.")],
        temperature=0.1,
        max_tokens=200,
    )

    assert answer == "Stub answer"
    assert len(llm_client.calls) == 1
    assert llm_client.calls[0]["temperature"] == 0.1
    assert llm_client.calls[0]["max_tokens"] == 200
