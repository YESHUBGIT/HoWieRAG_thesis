from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.datasets.t2_ragbench import (
    iter_t2_ragbench_benchmark_records,
    iter_t2_ragbench_source_documents,
)


def main() -> int:
    if len(sys.argv) != 3:
        print('Usage: python scripts/prepare_t2_ragbench.py path/to/T2-RAGBench output_dir')
        return 1

    input_path = sys.argv[1]
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    documents_path = output_dir / "documents.jsonl"
    questions_path = output_dir / "questions.jsonl"
    legacy_documents_path = output_dir / "source_documents.jsonl"
    legacy_questions_path = output_dir / "benchmark_questions.jsonl"

    source_record_count = 0
    with documents_path.open("w", encoding="utf-8") as file_handle:
        for record in iter_t2_ragbench_source_documents(input_path):
            source_record_count += 1
            file_handle.write(record.model_dump_json() + "\n")

    shutil.copyfile(documents_path, legacy_documents_path)

    benchmark_record_count = 0
    with questions_path.open("w", encoding="utf-8") as file_handle:
        for record in iter_t2_ragbench_benchmark_records(input_path):
            benchmark_record_count += 1
            file_handle.write(record.model_dump_json() + "\n")

    shutil.copyfile(questions_path, legacy_questions_path)

    print(f"Source documents written: {documents_path}")
    print(f"Benchmark questions written: {questions_path}")
    print(f"Source document count: {source_record_count}")
    print(f"Benchmark question count: {benchmark_record_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
