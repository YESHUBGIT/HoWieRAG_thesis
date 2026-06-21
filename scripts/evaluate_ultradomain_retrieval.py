from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from howie_rag.evaluation.ultradomain_retrieval import (
    evaluate_ultradomain_retrieval,
    load_benchmark_records,
    load_chunk_records,
    save_retrieval_results,
)


def main() -> int:
    if len(sys.argv) != 5:
        print(
            'Usage: python scripts/evaluate_ultradomain_retrieval.py processed/benchmark_questions.jsonl index/chunks.jsonl [keyword|bm25] output_dir'
        )
        return 1

    benchmark_path = sys.argv[1]
    chunks_path = sys.argv[2]
    retriever_name = sys.argv[3]
    output_dir = Path(sys.argv[4])
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_records = load_benchmark_records(benchmark_path)
    chunks = load_chunk_records(chunks_path)
    results = evaluate_ultradomain_retrieval(benchmark_records, chunks, retriever_name=retriever_name)

    json_path = output_dir / f"retrieval_results_{retriever_name}.json"
    csv_path = output_dir / f"retrieval_results_{retriever_name}.csv"
    save_retrieval_results(results, str(json_path), str(csv_path))

    print(f"Retriever: {retriever_name}")
    print(f"Questions: {results['question_count']}")
    print(f"Hit@1: {results['hit_at_1']:.4f}")
    print(f"Hit@5: {results['hit_at_5']:.4f}")
    print(f"MRR@5: {results['mrr_at_5']:.4f}")
    print(f"Precision@1: {results['precision_at_1']:.4f}")
    print(f"JSON results: {json_path}")
    print(f"CSV results: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
