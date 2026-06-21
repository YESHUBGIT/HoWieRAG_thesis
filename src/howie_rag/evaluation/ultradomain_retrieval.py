import json
from pathlib import Path
from typing import Dict, List, Tuple

from howie_rag.core.schemas import Chunk
from howie_rag.datasets.schemas import BenchmarkQARecord
from howie_rag.intent.rule_based import RuleBasedIntentClassifier
from howie_rag.retrieval_planning.experiment_variants import retrieve_with_variant


def load_chunk_records(file_path: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    with open(file_path, encoding="utf-8") as file_handle:
        for line in file_handle:
            stripped = line.strip()
            if not stripped:
                continue
            chunks.append(Chunk(**json.loads(stripped)))
    return chunks


def load_benchmark_records(file_path: str) -> List[BenchmarkQARecord]:
    records: List[BenchmarkQARecord] = []
    with open(file_path, encoding="utf-8") as file_handle:
        for line in file_handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(BenchmarkQARecord(**json.loads(stripped)))
    return records


def _is_correct_match(benchmark: BenchmarkQARecord, chunk: Chunk) -> bool:
    same_doc = chunk.doc_id == benchmark.gold_doc_id
    same_context = chunk.metadata.get("context_id") == benchmark.gold_context_id
    return same_doc or same_context


def evaluate_ultradomain_retrieval(
    benchmark_records: List[BenchmarkQARecord],
    chunks: List[Chunk],
    retriever_name: str,
    variant: str = "naive",
    top_k: int = 5,
    candidate_pool_size: int = 30,
) -> Dict[str, object]:
    intent_classifier = RuleBasedIntentClassifier()

    per_question_results = []
    hit_at_1_total = 0.0
    hit_at_5_total = 0.0
    mrr_at_5_total = 0.0
    precision_at_1_total = 0.0
    retrieved_chunks_total = 0.0

    for record in benchmark_records:
        retrieval_output = retrieve_with_variant(
            query=record.question,
            chunks=chunks,
            retriever_name=retriever_name,
            variant=variant,
            intent_classifier=intent_classifier,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
        )
        matches = retrieval_output["matches"]
        retrieval_plan = retrieval_output["retrieval_plan"]
        predicted_intent = retrieval_output["detected_intent"]
        if predicted_intent is None:
            predicted_intent = intent_classifier.classify(record.question).intent

        hit_at_1 = 1.0 if matches and _is_correct_match(record, matches[0].chunk) else 0.0
        hit_at_5 = 0.0
        reciprocal_rank = 0.0

        for rank, match in enumerate(matches[:5], start=1):
            if _is_correct_match(record, match.chunk):
                hit_at_5 = 1.0
                reciprocal_rank = 1.0 / rank
                break

        precision_at_1 = hit_at_1

        hit_at_1_total += hit_at_1
        hit_at_5_total += hit_at_5
        mrr_at_5_total += reciprocal_rank
        precision_at_1_total += precision_at_1
        retrieved_chunks_total += len(matches)

        per_question_results.append(
            {
                "question_id": record.question_id,
                "question": record.question,
                "predicted_intent": predicted_intent,
                "retrieval_mode": retrieval_plan.retrieval_mode if retrieval_plan else "normal",
                "preferred_document_types": retrieval_plan.preferred_document_types if retrieval_plan else [],
                "gold_doc_id": record.gold_doc_id,
                "gold_context_id": record.gold_context_id,
                "hit_at_1": hit_at_1,
                "hit_at_5": hit_at_5,
                "mrr_at_5": reciprocal_rank,
                "precision_at_1": precision_at_1,
                "retrieved": [
                    {
                        "chunk_id": match.chunk.chunk_id,
                        "doc_id": match.chunk.doc_id,
                        "context_id": match.chunk.metadata.get("context_id"),
                        "original_score": match.original_score if match.original_score is not None else match.score,
                        "adjusted_score": match.adjusted_score if match.adjusted_score is not None else match.score,
                    }
                    for match in matches
                ],
            }
        )

    total = len(benchmark_records) or 1
    return {
        "retriever_name": retriever_name,
        "variant": variant,
        "question_count": len(benchmark_records),
        "hit_at_1": hit_at_1_total / total,
        "hit_at_5": hit_at_5_total / total,
        "mrr_at_5": mrr_at_5_total / total,
        "precision_at_1": precision_at_1_total / total,
        "average_retrieved_chunks_per_question": retrieved_chunks_total / total,
        "per_question": per_question_results,
    }


def save_retrieval_results(results: Dict[str, object], json_path: str, csv_path: str) -> None:
    Path(json_path).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    per_question = results.get("per_question", [])
    csv_lines = [
        "question_id,question,predicted_intent,gold_doc_id,gold_context_id,hit_at_1,hit_at_5,mrr_at_5,precision_at_1"
    ]
    for item in per_question:
        question = str(item["question"]).replace('"', '""')
        csv_lines.append(
            f'{item["question_id"]},"{question}",{item["predicted_intent"]},{item["gold_doc_id"]},{item["gold_context_id"]},{item["hit_at_1"]},{item["hit_at_5"]},{item["mrr_at_5"]},{item["precision_at_1"]}'
        )
    Path(csv_path).write_text("\n".join(csv_lines), encoding="utf-8")


def save_per_question_debug(results: Dict[str, object], file_path: str) -> None:
    lines = [json.dumps(item, ensure_ascii=False) for item in results.get("per_question", [])]
    Path(file_path).write_text("\n".join(lines), encoding="utf-8")
