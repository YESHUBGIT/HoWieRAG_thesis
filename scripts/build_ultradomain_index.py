import argparse
from pathlib import Path
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.chunking.text_chunker import chunk_document
from howie_rag.datasets.schemas import SourceDocumentRecord
from howie_rag.datasets.ultradomain import source_records_to_documents


def render_progress(current: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[no work]"
    completed = int(width * current / total)
    bar = "#" * completed + "-" * (width - completed)
    return f"[{bar}] {current}/{total}"


def iter_source_records(source_documents_path: Path):
    with source_documents_path.open(encoding="utf-8") as file_handle:
        for line in file_handle:
            if not line.strip():
                continue
            yield SourceDocumentRecord(**json.loads(line))


def estimate_chunk_count(text_length: int, chunk_size: int, overlap: int) -> int:
    if text_length <= 0:
        return 0
    step = chunk_size - overlap
    return 1 + max(0, (text_length - chunk_size + step - 1) // step)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a classified UltraDomain chunk index.")
    parser.add_argument("source_documents", help="Path to processed documents.jsonl")
    parser.add_argument("output_dir", help="Directory for classified documents and chunks")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=150, help="Chunk overlap in characters")
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Print detailed progress every N documents, plus large documents",
    )
    parser.add_argument(
        "--large-doc-threshold",
        type=int,
        default=200000,
        help="Always log documents at or above this many characters",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    source_documents_path = Path(args.source_documents)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    documents_path = output_dir / "documents_classified.jsonl"
    chunks_path = output_dir / "chunks.jsonl"

    print(f"Loading source records from: {source_documents_path}", flush=True)
    with source_documents_path.open(encoding="utf-8") as file_handle:
        total_records = sum(1 for line in file_handle if line.strip())

    print(
        (
            "Converting source records into classified Document objects and chunking... "
            f"chunk_size={args.chunk_size}, overlap={args.overlap}, log_every={args.log_every}"
        ),
        flush=True,
    )
    document_count = 0
    chunk_count = 0
    start_time = time.perf_counter()
    with documents_path.open("w", encoding="utf-8") as documents_file_handle, chunks_path.open(
        "w", encoding="utf-8"
    ) as chunks_file_handle:
        for record_index, record in enumerate(iter_source_records(source_documents_path), start=1):
            if record_index == 1 or record_index == total_records or record_index % 100 == 0:
                print(f"load  {render_progress(record_index, total_records)}", flush=True)

            record_text_length = len(record.text)
            should_log_detail = (
                record_index == 1
                or record_index == total_records
                or record_index % args.log_every == 0
                or record_text_length >= args.large_doc_threshold
            )
            if should_log_detail:
                elapsed_before = time.perf_counter() - start_time
                print(
                    (
                        f"doc start   record={record_index}/{total_records} "
                        f"chars={record_text_length} elapsed={elapsed_before:.1f}s"
                    ),
                    flush=True,
                )

            document_start_time = time.perf_counter()
            documents = source_records_to_documents([record])
            document_elapsed = time.perf_counter() - document_start_time
            if should_log_detail:
                elapsed_after = time.perf_counter() - start_time
                print(
                    (
                        f"doc done    record={record_index}/{total_records} "
                        f"documents={len(documents)} doc_time={document_elapsed:.1f}s elapsed={elapsed_after:.1f}s"
                    ),
                    flush=True,
                )

            for document in documents:
                document_count += 1
                documents_file_handle.write(document.model_dump_json() + "\n")

                if document_count == 1 or document_count == total_records or document_count % 100 == 0:
                    print(f"docs  {render_progress(document_count, total_records)}", flush=True)

                text_length = len(document.text)
                estimated_chunks = estimate_chunk_count(text_length, args.chunk_size, args.overlap)
                if should_log_detail:
                    elapsed_before = time.perf_counter() - start_time
                    print(
                        (
                            f"chunk start doc={document_count}/{total_records} "
                            f"chars={text_length} est_chunks={estimated_chunks} elapsed={elapsed_before:.1f}s"
                        ),
                        flush=True,
                    )

                chunk_start_time = time.perf_counter()
                chunks = chunk_document(document, chunk_size=args.chunk_size, overlap=args.overlap)
                chunk_elapsed = time.perf_counter() - chunk_start_time
                chunk_count += len(chunks)
                for chunk in chunks:
                    chunks_file_handle.write(chunk.model_dump_json() + "\n")

                if should_log_detail:
                    total_elapsed = time.perf_counter() - start_time
                    print(
                        (
                            f"chunk done  {render_progress(document_count, total_records)} "
                            f"doc_chunks={len(chunks)} chunks_so_far={chunk_count} "
                            f"doc_time={chunk_elapsed:.1f}s elapsed={total_elapsed:.1f}s"
                        ),
                        flush=True,
                    )

    print(f"Classified documents written: {documents_path}", flush=True)
    print(f"Chunks written: {chunks_path}", flush=True)
    print(f"Document count: {document_count}", flush=True)
    print(f"Chunk count: {chunk_count}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
