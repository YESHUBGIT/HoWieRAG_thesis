import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.evaluation.ultradomain_retrieval import (
    evaluate_ultradomain_retrieval,
    load_benchmark_records,
    load_chunk_records,
    save_retrieval_results,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run UltraDomain retrieval evaluation.")
    parser.add_argument("questions", help="Prepared UltraDomain questions.jsonl path")
    parser.add_argument("chunks", help="Built UltraDomain chunks.jsonl path")
    parser.add_argument("retriever", choices=["keyword", "bm25"], help="Retriever to evaluate")
    parser.add_argument("output_dir", help="Output directory for evaluation results")
    parser.add_argument("--log-every", type=int, default=25)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    benchmark_path = args.questions
    chunks_path = args.chunks
    retriever_name = args.retriever
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_records = load_benchmark_records(benchmark_path)
    chunks = load_chunk_records(chunks_path)
    results = evaluate_ultradomain_retrieval(
        benchmark_records,
        chunks,
        retriever_name=retriever_name,
        log_every=args.log_every,
    )

    json_path = output_dir / f"retrieval_results_{retriever_name}.json"
    csv_path = output_dir / f"retrieval_results_{retriever_name}.csv"
    save_retrieval_results(results, str(json_path), str(csv_path))

    print(f"Retriever: {retriever_name}", flush=True)
    print(f"Questions: {results['question_count']}", flush=True)
    print(f"Hit@1: {results['hit_at_1']:.4f}", flush=True)
    print(f"Hit@5: {results['hit_at_5']:.4f}", flush=True)
    print(f"MRR@5: {results['mrr_at_5']:.4f}", flush=True)
    print(f"Precision@1: {results['precision_at_1']:.4f}", flush=True)
    print(f"JSON results: {json_path}", flush=True)
    print(f"CSV results: {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
