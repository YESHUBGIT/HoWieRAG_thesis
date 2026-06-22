import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.evaluation.ultradomain_retrieval import (
    evaluate_ultradomain_retrieval,
    load_benchmark_records,
    load_chunk_records,
    save_per_question_debug,
    save_retrieval_results,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run UltraDomain retrieval planning experiments.")
    parser.add_argument("--questions", required=True, help="Prepared UltraDomain questions.jsonl path")
    parser.add_argument("--chunks", required=True, help="Built UltraDomain chunks.jsonl path")
    parser.add_argument("--retriever", choices=["keyword", "bm25"], required=True)
    parser.add_argument(
        "--variant",
        choices=["naive", "document_aware", "intent_document_aware", "all"],
        required=True,
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-pool-size", type=int, default=30)
    parser.add_argument("--log-every", type=int, default=25)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    benchmark_records = load_benchmark_records(args.questions)
    chunks = load_chunk_records(args.chunks)

    output_dir = Path("evaluation_results/ultradomain")
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [args.variant] if args.variant != "all" else ["naive", "document_aware", "intent_document_aware"]
    comparison_rows = []

    for variant in variants:
        results = evaluate_ultradomain_retrieval(
            benchmark_records,
            chunks,
            retriever_name=args.retriever,
            variant=variant,
            top_k=args.top_k,
            candidate_pool_size=args.candidate_pool_size,
            log_every=args.log_every,
        )

        json_path = output_dir / f"ultradomain_retrieval_{variant}.json"
        csv_path = output_dir / f"ultradomain_retrieval_{variant}.csv"
        debug_path = output_dir / f"per_question_debug_{variant}.jsonl"

        save_retrieval_results(results, str(json_path), str(csv_path))
        save_per_question_debug(results, str(debug_path))

        comparison_rows.append(
            {
                "variant": variant,
                "retriever": args.retriever,
                "question_count": results["question_count"],
                "hit_at_1": results["hit_at_1"],
                "hit_at_5": results["hit_at_5"],
                "mrr_at_5": results["mrr_at_5"],
                "precision_at_1": results["precision_at_1"],
                "average_retrieved_chunks_per_question": results["average_retrieved_chunks_per_question"],
            }
        )

        print(f"Variant: {variant}", flush=True)
        print(f"  Hit@1: {results['hit_at_1']:.4f}", flush=True)
        print(f"  Hit@5: {results['hit_at_5']:.4f}", flush=True)
        print(f"  MRR@5: {results['mrr_at_5']:.4f}", flush=True)
        print(f"  Precision@1: {results['precision_at_1']:.4f}", flush=True)

    comparison_path = output_dir / "ultradomain_retrieval_comparison.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=[
                "variant",
                "retriever",
                "question_count",
                "hit_at_1",
                "hit_at_5",
                "mrr_at_5",
                "precision_at_1",
                "average_retrieved_chunks_per_question",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"Comparison CSV: {comparison_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
