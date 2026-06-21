from typing import List

from howie_rag.llm.base import BaseLLMClient
from howie_rag.retrieval.base import RetrievalMatch


RAG_SYSTEM_PROMPT = (
    "You are a research assistant for higher-education and science-research questions. "
    "Answer using only the provided context. If the context is insufficient, say so clearly. "
    "Do not invent facts. Be concise and clear. When possible, mention which source titles support the answer."
)


def build_rag_user_prompt(question: str, intent: str, retrieval_matches: List[RetrievalMatch]) -> str:
    context_sections = []
    for index, match in enumerate(retrieval_matches, start=1):
        title = match.chunk.metadata.get("title", "unknown")
        context_sections.append(
            f"[Source {index}: {title}]\n{match.chunk.text}"
        )

    context_text = "\n\n".join(context_sections) if context_sections else "No context available."
    return (
        f"Question:\n{question}\n\n"
        f"Predicted intent:\n{intent}\n\n"
        f"Context:\n{context_text}\n\n"
        "Instructions:\n"
        "- Answer the question using only the context above.\n"
        "- If the context is not enough, say so clearly.\n"
        "- If multiple sources support the answer, synthesize them.\n"
        ' - End with a short "Sources used:" line listing source titles.\n'
    )


def answer_with_rag(
    llm_client: BaseLLMClient,
    question: str,
    intent: str,
    retrieval_matches: List[RetrievalMatch],
    temperature: float = 0.2,
    max_tokens: int = 400,
) -> str:
    user_prompt = build_rag_user_prompt(question, intent, retrieval_matches)
    return llm_client.generate(
        system_prompt=RAG_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
