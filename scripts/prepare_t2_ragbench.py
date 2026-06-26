import argparse
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.datasets.t2_ragbench import (
    iter_t2_ragbench_benchmark_records,
    iter_t2_ragbench_source_documents,
)


def _csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare T2-RAGBench into common HoWieRAG JSONL format.")
    parser.add_argument("input_path", help="Path to T2-RAGBench root or data directory")
    parser.add_argument("output_dir", help="Directory for prepared documents/questions JSONL")
    parser.add_argument("--subsets", type=_csv_values, help="Comma-separated subset filter, e.g. FinQA,ConvFinQA")
    parser.add_argument("--splits", type=_csv_values, help="Comma-separated split filter, e.g. test or train,dev,test")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    input_path = args.input_path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    documents_path = output_dir / "documents.jsonl"
    questions_path = output_dir / "questions.jsonl"
    legacy_documents_path = output_dir / "source_documents.jsonl"
    legacy_questions_path = output_dir / "benchmark_questions.jsonl"

    source_record_count = 0
    with documents_path.open("w", encoding="utf-8") as file_handle:
        for record in iter_t2_ragbench_source_documents(input_path, subsets=args.subsets, splits=args.splits):
            source_record_count += 1
            file_handle.write(record.model_dump_json() + "\n")

    shutil.copyfile(documents_path, legacy_documents_path)

    benchmark_record_count = 0
    with questions_path.open("w", encoding="utf-8") as file_handle:
        for record in iter_t2_ragbench_benchmark_records(input_path, subsets=args.subsets, splits=args.splits):
            benchmark_record_count += 1
            file_handle.write(record.model_dump_json() + "\n")

    shutil.copyfile(questions_path, legacy_questions_path)

    print(f"Source documents written: {documents_path}")
    print(f"Benchmark questions written: {questions_path}")
    print(f"Source document count: {source_record_count}")
    print(f"Benchmark question count: {benchmark_record_count}")
    print(f"Subset filter: {args.subsets or 'all'}")
    print(f"Split filter: {args.splits or 'all'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
