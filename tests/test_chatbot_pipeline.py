from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.chatbot.pipeline import run_simple_chatbot


def test_run_simple_chatbot_returns_intent_and_matches(tmp_path: Path) -> None:
    (tmp_path / "mobility.txt").write_text(
        "Student mobility increased across countries. Financial aid supported mobility.",
        encoding="utf-8",
    )
    (tmp_path / "housing.txt").write_text(
        "Student housing costs increased in major cities.", encoding="utf-8"
    )

    response = run_simple_chatbot(
        "What trend do we see in student mobility?",
        str(tmp_path),
        chunk_size=80,
        overlap=10,
        top_k=2,
        retriever_name="keyword",
    )

    assert response.intent == "TREND_PATTERN"
    assert response.retrieval_matches
    assert response.retrieval_matches[0].chunk.metadata["title"] == "mobility"


def test_run_simple_chatbot_handles_no_retrieval_match(tmp_path: Path) -> None:
    (tmp_path / "mobility.txt").write_text("Student mobility increased.", encoding="utf-8")

    response = run_simple_chatbot("quantum mechanics", str(tmp_path), top_k=2, retriever_name="keyword")

    assert response.intent == "UNKNOWN"
    assert response.retrieval_matches == []


def test_run_simple_chatbot_supports_bm25_retriever(tmp_path: Path) -> None:
    (tmp_path / "survey.txt").write_text(
        "Student survey methods and mobility indicators are documented here.",
        encoding="utf-8",
    )

    response = run_simple_chatbot("student survey methods", str(tmp_path), top_k=1, retriever_name="bm25")

    assert response.retrieval_matches
    assert response.retrieval_matches[0].chunk.metadata["title"] == "survey"
