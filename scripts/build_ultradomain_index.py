from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.chunking.text_chunker import chunk_documents
from howie_rag.core.schemas import Document
from howie_rag.datasets.schemas import SourceDocumentRecord
from howie_rag.datasets.ultradomain import source_records_to_documents


def main() -> int:
    if len(sys.argv) != 3:
        print('Usage: python scripts/build_ultradomain_index.py processed/documents.jsonl output_dir')
        return 1

    source_documents_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    source_records = []
    with source_documents_path.open(encoding="utf-8") as file_handle:
        for line in file_handle:
            stripped = line.strip()
            if not stripped:
                continue
            source_records.append(SourceDocumentRecord(**json.loads(stripped)))

    documents = source_records_to_documents(source_records)
    chunks = chunk_documents(documents)

    documents_path = output_dir / "documents_classified.jsonl"
    chunks_path = output_dir / "chunks.jsonl"

    with documents_path.open("w", encoding="utf-8") as file_handle:
        for document in documents:
            file_handle.write(document.model_dump_json() + "\n")

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
