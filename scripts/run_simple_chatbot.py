from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.chatbot.pipeline import run_simple_chatbot


def main() -> int:
    if len(sys.argv) not in {3, 4}:
        print(
            'Usage: python scripts/run_simple_chatbot.py "Your question here" path/to/documents [keyword|bm25]'
        )
        return 1

    question = sys.argv[1]
    documents_dir = sys.argv[2]
    retriever_name = sys.argv[3] if len(sys.argv) == 4 else "bm25"
    response = run_simple_chatbot(question, documents_dir, retriever_name=retriever_name)

    print(f"Question: {response.question}")
    print(f"Intent: {response.intent}")
    print(f"Confidence: {response.confidence}")
    print(f"Reasoning: {response.reasoning}")
    print(f"Retriever: {retriever_name}")
    print("Retrieved Chunks:")

    if not response.retrieval_matches:
        print("- No matching chunks found.")
        return 0

    for match in response.retrieval_matches:
        title = match.chunk.metadata.get("title", "unknown")
        print(f"- score={match.score} doc={match.chunk.doc_id} title={title}")
        print(f"  {match.chunk.text}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
