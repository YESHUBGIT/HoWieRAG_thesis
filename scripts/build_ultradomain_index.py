from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.chunking.text_chunker import chunk_documents
from howie_rag.core.schemas import Document
from howie_rag.datasets.schemas import SourceDocumentRecord
from howie_rag.datasets.ultradomain import source_records_to_documents


def render_progress(current: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[no work]"
    completed = int(width * current / total)
    bar = "#" * completed + "-" * (width - completed)
    return f"[{bar}] {current}/{total}"


def main() -> int:
    if len(sys.argv) != 3:
        print('Usage: python scripts/build_ultradomain_index.py processed/documents.jsonl output_dir')
        return 1

    source_documents_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading source records from: {source_documents_path}")
    lines = [line for line in source_documents_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    total_records = len(lines)
    source_records = []
    for index, line in enumerate(lines, start=1):
        source_records.append(SourceDocumentRecord(**json.loads(line)))
        if index == 1 or index == total_records or index % 100 == 0:
            print(f"load  {render_progress(index, total_records)}")

    print("Converting source records into classified Document objects...")
    documents = []
    for index, record in enumerate(source_records, start=1):
        documents.extend(source_records_to_documents([record]))
        if index == 1 or index == total_records or index % 100 == 0:
            print(f"docs  {render_progress(index, total_records)}")

    print("Chunking classified documents...")
    chunks = []
    total_documents = len(documents)
    for index, document in enumerate(documents, start=1):
        chunks.extend(chunk_documents([document]))
        if index == 1 or index == total_documents or index % 100 == 0:
            print(f"chunk {render_progress(index, total_documents)} | chunks so far={len(chunks)}")

    documents_path = output_dir / "documents_classified.jsonl"
    chunks_path = output_dir / "chunks.jsonl"

    print("Writing classified documents...")
    with documents_path.open("w", encoding="utf-8") as file_handle:
        for document in documents:
            file_handle.write(document.model_dump_json() + "\n")

    print("Writing chunks...")
    with chunks_path.open("w", encoding="utf-8") as file_handle:
        for chunk in chunks:
            file_handle.write(chunk.model_dump_json() + "\n")

    print(f"Classified documents written: {documents_path}")
    print(f"Chunks written: {chunks_path}")
    print(f"Document count: {len(documents)}")
    print(f"Chunk count: {len(chunks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
