from typing import List

from howie_rag.core.schemas import Chunk, Document
from howie_rag.core.utils import stable_id


def chunk_document(document: Document, chunk_size: int = 300, overlap: int = 50) -> List[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be 0 or greater")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[Chunk] = []
    text = document.text.strip()
    if not text:
        return chunks

    start_index = 0
    chunk_index = 0
    step = chunk_size - overlap

    while start_index < len(text):
        end_index = min(start_index + chunk_size, len(text))
        chunk_text = text[start_index:end_index].strip()

        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=stable_id(f"{document.doc_id}:{chunk_index}:{chunk_text}"),
                    doc_id=document.doc_id,
                    text=chunk_text,
                    metadata={
                        "title": document.title,
                        "chunk_index": chunk_index,
                        "start_index": start_index,
                        "end_index": end_index,
                        **document.metadata,
                    },
                )
            )

        if end_index == len(text):
            break

        chunk_index += 1
        start_index += step

    return chunks


def chunk_documents(
    documents: List[Document], chunk_size: int = 300, overlap: int = 50
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size=chunk_size, overlap=overlap))
    return chunks
