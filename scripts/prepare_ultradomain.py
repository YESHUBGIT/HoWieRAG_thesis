from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.datasets.ultradomain import (
    load_ultradomain_benchmark_records,
    load_ultradomain_source_documents,
)


def main() -> int:
    if len(sys.argv) != 3:
        print('Usage: python scripts/prepare_ultradomain.py path/to/ultradomain output_dir')
        return 1

    input_path = sys.argv[1]
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    source_records = load_ultradomain_source_documents(input_path)
    benchmark_records = load_ultradomain_benchmark_records(input_path)

    documents_path = output_dir / "documents.jsonl"
    questions_path = output_dir / "questions.jsonl"
    legacy_documents_path = output_dir / "source_documents.jsonl"
    legacy_questions_path = output_dir / "benchmark_questions.jsonl"

    with documents_path.open("w", encoding="utf-8") as file_handle:
        for record in source_records:
            file_handle.write(record.model_dump_json() + "\n")

    legacy_documents_path.write_text(documents_path.read_text(encoding="utf-8"), encoding="utf-8")

    with questions_path.open("w", encoding="utf-8") as file_handle:
        for record in benchmark_records:
            file_handle.write(record.model_dump_json() + "\n")

    legacy_questions_path.write_text(questions_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Source documents written: {documents_path}")
    print(f"Benchmark questions written: {questions_path}")
    print(f"Source document count: {len(source_records)}")
    print(f"Benchmark question count: {len(benchmark_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
