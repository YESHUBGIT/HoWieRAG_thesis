import re
from typing import Dict, List, Tuple

from howie_rag.core.schemas import Chunk, Document
from howie_rag.core.utils import stable_id


def _chunk_metadata(document: Document, chunk_index: int, chunk_text: str, start_index: int, end_index: int) -> dict:
    lines = [line.strip() for line in chunk_text.splitlines() if line.strip()]
    table_line_count = sum(1 for line in lines if _is_table_like_line(line))
    line_count = len(lines)
    table_ratio = table_line_count / line_count if line_count else 0.0

    if table_line_count == 0:
        chunk_type = "narrative"
    elif table_ratio >= 0.7:
        chunk_type = "table"
    else:
        chunk_type = "mixed"

    return {
        "title": document.title,
        "chunk_index": chunk_index,
        "start_index": start_index,
        "end_index": end_index,
        "chunk_type": chunk_type,
        "has_table_like_content": table_line_count > 0,
        "table_line_count": table_line_count,
        "line_count": line_count,
        "table_line_ratio": round(table_ratio, 4),
        **document.metadata,
    }


def _append_chunk(
    chunks: List[Chunk],
    document: Document,
    chunk_index: int,
    chunk_text: str,
    start_index: int,
    end_index: int,
) -> int:
    normalized_text = chunk_text.strip()
    if not normalized_text:
        return chunk_index

    chunks.append(
        Chunk(
            chunk_id=stable_id(f"{document.doc_id}:{chunk_index}:{normalized_text}"),
            doc_id=document.doc_id,
            text=normalized_text,
            metadata=_chunk_metadata(document, chunk_index, normalized_text, start_index, end_index),
        )
    )
    return chunk_index + 1


def _chunk_text_fixed(
    document: Document, text: str, chunk_size: int, overlap: int, start_offset: int = 0, chunk_index: int = 0
) -> Tuple[List[Chunk], int]:
    chunks: List[Chunk] = []
    start_index = 0
    step = chunk_size - overlap

    while start_index < len(text):
        end_index = min(start_index + chunk_size, len(text))
        chunk_text = text[start_index:end_index]
        chunk_index = _append_chunk(
            chunks,
            document,
            chunk_index,
            chunk_text,
            start_offset + start_index,
            start_offset + end_index,
        )

        if end_index == len(text):
            break

        start_index += step

    return chunks, chunk_index


def _is_table_like_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.count("|") >= 2:
        return True
    if stripped.count(";") >= 2 or stripped.count("\t") >= 2:
        return True
    numeric_tokens = re.findall(r"\b\d+(?:[.,]\d+)?\b", stripped)
    return len(numeric_tokens) >= 4


def _has_table_structure(text: str) -> bool:
    return any(_is_table_like_line(line) for line in text.splitlines())


def _split_structured_blocks(text: str) -> List[Dict[str, object]]:
    blocks: List[Dict[str, object]] = []
    current_lines: List[str] = []
    current_type = "narrative"
    current_start = 0
    line_start = 0

    for line_with_break in text.splitlines(keepends=True):
        line_end = line_start + len(line_with_break)
        stripped = line_with_break.strip()
        if not stripped:
            if current_lines:
                blocks.append(
                    {
                        "text": "".join(current_lines).strip(),
                        "type": current_type,
                        "start": current_start,
                        "end": line_start,
                    }
                )
                current_lines = []
            line_start = line_end
            continue

        line_type = "table" if _is_table_like_line(stripped) else "narrative"
        if not current_lines:
            current_lines = [line_with_break]
            current_type = line_type
            current_start = line_start
        elif line_type == current_type:
            current_lines.append(line_with_break)
        else:
            blocks.append(
                {
                    "text": "".join(current_lines).strip(),
                    "type": current_type,
                    "start": current_start,
                    "end": line_start,
                }
            )
            current_lines = [line_with_break]
            current_type = line_type
            current_start = line_start

        line_start = line_end

    if current_lines:
        blocks.append(
            {
                "text": "".join(current_lines).strip(),
                "type": current_type,
                "start": current_start,
                "end": len(text),
            }
        )

    return blocks


def _table_header_line_count(lines: List[str]) -> int:
    if len(lines) >= 2 and lines[0].count("|") >= 2 and lines[1].count("|") >= 2:
        return 2
    return 1


def _chunk_table_block(
    document: Document,
    block_text: str,
    block_start: int,
    block_end: int,
    chunk_size: int,
    chunk_index: int,
) -> Tuple[List[Chunk], int]:
    lines = [line.rstrip() for line in block_text.splitlines() if line.strip()]
    if not lines:
        return [], chunk_index
    if len(block_text) <= chunk_size:
        chunks: List[Chunk] = []
        chunk_index = _append_chunk(chunks, document, chunk_index, block_text, block_start, block_end)
        return chunks, chunk_index

    header_line_count = _table_header_line_count(lines)
    header_lines = lines[:header_line_count]
    data_lines = lines[header_line_count:] or lines[header_line_count - 1 :]

    chunks: List[Chunk] = []
    current_lines = list(header_lines)

    for line in data_lines:
        candidate_lines = current_lines + [line]
        candidate_text = "\n".join(candidate_lines)
        if len(candidate_text) <= chunk_size or len(current_lines) == len(header_lines):
            current_lines = candidate_lines
            continue

        chunk_index = _append_chunk(chunks, document, chunk_index, "\n".join(current_lines), block_start, block_end)
        current_lines = list(header_lines) + [line]

    if current_lines:
        chunk_index = _append_chunk(chunks, document, chunk_index, "\n".join(current_lines), block_start, block_end)

    return chunks, chunk_index


def _chunk_text_table_aware(document: Document, text: str, chunk_size: int, overlap: int) -> List[Chunk]:
    blocks = _split_structured_blocks(text)
    chunks: List[Chunk] = []
    chunk_index = 0

    for block in blocks:
        block_text = str(block["text"])
        block_start = int(block["start"])
        block_end = int(block["end"])
        if block["type"] == "table":
            block_chunks, chunk_index = _chunk_table_block(
                document,
                block_text,
                block_start,
                block_end,
                chunk_size,
                chunk_index,
            )
        else:
            block_chunks, chunk_index = _chunk_text_fixed(
                document,
                block_text,
                chunk_size,
                overlap,
                start_offset=block_start,
                chunk_index=chunk_index,
            )
        chunks.extend(block_chunks)

    return chunks


def chunk_document(document: Document, chunk_size: int = 300, overlap: int = 50) -> List[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be 0 or greater")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = document.text.strip()
    if not text:
        return []

    if _has_table_structure(text):
        return _chunk_text_table_aware(document, text, chunk_size=chunk_size, overlap=overlap)

    chunks, _ = _chunk_text_fixed(document, text, chunk_size=chunk_size, overlap=overlap)
    return chunks


def chunk_documents(
    documents: List[Document], chunk_size: int = 300, overlap: int = 50
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size=chunk_size, overlap=overlap))
    return chunks
