from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.answering.rag_answerer import answer_with_rag
from howie_rag.chatbot.pipeline import run_simple_chatbot
from howie_rag.llm.vllm_client import VLLMClient


DEFAULT_BASE_URL = os.environ.get("HOWIE_LLM_BASE_URL", "http://localhost:8000")
DEFAULT_MODEL_NAME = os.environ.get("HOWIE_LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")
DEFAULT_RETRIEVER_NAME = os.environ.get("HOWIE_RETRIEVER", "bm25")


def main() -> int:
    if len(sys.argv) not in {3, 4}:
        print(
            'Usage: python scripts/run_rag_chatbot.py "Your question here" path/to/documents [keyword|bm25]'
        )
        return 1

    question = sys.argv[1]
    documents_dir = sys.argv[2]
    retriever_name = sys.argv[3] if len(sys.argv) == 4 else DEFAULT_RETRIEVER_NAME

    pipeline_response = run_simple_chatbot(question, documents_dir, retriever_name=retriever_name)
    llm_client = VLLMClient(base_url=DEFAULT_BASE_URL, model_name=DEFAULT_MODEL_NAME)
    answer = answer_with_rag(
        llm_client=llm_client,
        question=pipeline_response.question,
        intent=pipeline_response.intent,
        retrieval_matches=pipeline_response.retrieval_matches,
    )

    print(f"Question: {pipeline_response.question}")
    print(f"Intent: {pipeline_response.intent}")
    print(f"Confidence: {pipeline_response.confidence}")
    print(f"Reasoning: {pipeline_response.reasoning}")
    print(f"Retriever: {retriever_name}")
    print("Answer:")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
