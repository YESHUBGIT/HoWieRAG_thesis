import argparse
import csv
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.evaluation.ultradomain_retrieval import (
    evaluate_ultradomain_retrieval,
    load_benchmark_records,
    load_chunk_records,
    save_per_question_debug,
    save_retrieval_results,
    select_benchmark_records,
)
from howie_rag.intent.llm_intent import LLMIntentClassifier
from howie_rag.llm.vllm_client import VLLMClient
from howie_rag.retrieval_planning.llm_planner import LLMRetrievalPlanner


DEFAULT_PLANNER_BASE_URL = os.environ.get("HOWIE_PLANNER_LLM_BASE_URL", os.environ.get("HOWIE_LLM_BASE_URL", "http://localhost:8000"))
DEFAULT_PLANNER_MODEL = os.environ.get("HOWIE_PLANNER_LLM_MODEL", os.environ.get("HOWIE_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run UltraDomain retrieval planning experiments.")
    parser.add_argument("--questions", required=True, help="Prepared UltraDomain questions.jsonl path")
    parser.add_argument("--chunks", required=True, help="Built UltraDomain chunks.jsonl path")
    parser.add_argument("--retriever", choices=["keyword", "bm25", "field_bm25"], required=True)
    parser.add_argument(
        "--variant",
        choices=[
            "naive",
            "document_aware",
            "intent_document_aware",
            "llm_intent_document_aware",
            "llm_document_aware",
            "intent_guided_retrieval",
            "llm_guided_retrieval",
            "all",
            "all_core",
            "all_non_llm",
            "all_llm",
            "all_full",
        ],
        required=True,
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-pool-size", type=int, default=30)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--planner-base-url", default=DEFAULT_PLANNER_BASE_URL)
    parser.add_argument("--planner-model", default=DEFAULT_PLANNER_MODEL)
    parser.add_argument("--planner-max-tokens", type=int, default=300)
    parser.add_argument("--output-dir", default="evaluation_results/ultradomain")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    benchmark_records = select_benchmark_records(
        load_benchmark_records(args.questions),
        max_questions=args.max_questions,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
    )
    chunks = load_chunk_records(args.chunks)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.variant in {"all", "all_core"}:
        variants = ["naive", "document_aware", "intent_document_aware"]
    elif args.variant == "all_non_llm":
        variants = ["naive", "document_aware", "intent_document_aware", "intent_guided_retrieval"]
    elif args.variant == "all_llm":
        variants = ["llm_intent_document_aware", "llm_document_aware", "llm_guided_retrieval"]
    elif args.variant == "all_full":
        variants = [
            "naive",
            "document_aware",
            "intent_document_aware",
            "intent_guided_retrieval",
            "llm_intent_document_aware",
            "llm_document_aware",
            "llm_guided_retrieval",
        ]
    else:
        variants = [args.variant]
    comparison_rows = []
    llm_planner = None
    llm_intent_classifier = None
    if any(variant in {"llm_document_aware", "llm_guided_retrieval"} for variant in variants):
        llm_planner = LLMRetrievalPlanner(
            VLLMClient(base_url=args.planner_base_url, model_name=args.planner_model),
            max_tokens=args.planner_max_tokens,
        )
    if any(variant == "llm_intent_document_aware" for variant in variants):
        llm_intent_classifier = LLMIntentClassifier(
            VLLMClient(base_url=args.planner_base_url, model_name=args.planner_model),
        )

    for variant in variants:
        json_path = output_dir / f"ultradomain_retrieval_{variant}.json"
        csv_path = output_dir / f"ultradomain_retrieval_{variant}.csv"
        debug_path = output_dir / f"per_question_debug_{variant}.jsonl"
        partial_json_path = output_dir / f"ultradomain_retrieval_{variant}.partial.json"
        partial_csv_path = output_dir / f"ultradomain_retrieval_{variant}.partial.csv"
        partial_debug_path = output_dir / f"per_question_debug_{variant}.partial.jsonl"

        def save_partial(results: dict, *, variant_name: str = variant) -> None:
            save_retrieval_results(results, str(partial_json_path), str(partial_csv_path))
            save_per_question_debug(results, str(partial_debug_path))
            print(
                f"checkpoint saved: variant={variant_name} questions={results['question_count']} -> {partial_json_path}",
                flush=True,
            )

        results = evaluate_ultradomain_retrieval(
            benchmark_records,
            chunks,
            retriever_name=args.retriever,
            variant=variant,
            top_k=args.top_k,
            candidate_pool_size=args.candidate_pool_size,
            log_every=args.log_every,
            save_every=args.save_every,
            checkpoint_callback=save_partial if args.save_every else None,
            llm_intent_classifier=llm_intent_classifier,
            llm_planner=llm_planner,
        )

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
                "oracle_candidate_hit_rate": results["oracle_candidate_hit_rate"],
                "average_retrieved_chunks_per_question": results["average_retrieved_chunks_per_question"],
            }
        )

        print(f"Variant: {variant}", flush=True)
        print(f"  Hit@1: {results['hit_at_1']:.4f}", flush=True)
        print(f"  Hit@5: {results['hit_at_5']:.4f}", flush=True)
        print(f"  MRR@5: {results['mrr_at_5']:.4f}", flush=True)
        print(f"  Precision@1: {results['precision_at_1']:.4f}", flush=True)
        print(f"  Oracle candidate hit rate: {results['oracle_candidate_hit_rate']:.4f}", flush=True)

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
                "oracle_candidate_hit_rate",
                "average_retrieved_chunks_per_question",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"Comparison CSV: {comparison_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
