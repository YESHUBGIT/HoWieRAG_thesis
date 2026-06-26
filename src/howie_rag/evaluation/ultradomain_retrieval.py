import json
from pathlib import Path
import random
import time
from typing import Callable, Dict, List, Optional

from howie_rag.core.schemas import Chunk
from howie_rag.datasets.schemas import BenchmarkQARecord
from howie_rag.intent.llm_intent import LLMIntentClassifier
from howie_rag.intent.rule_based import RuleBasedIntentClassifier
from howie_rag.retrieval.factory import create_retriever
from howie_rag.retrieval_planning.llm_planner import LLMRetrievalPlanner
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


def _first_correct_candidate_rank(benchmark: BenchmarkQARecord, candidate_matches: List) -> Optional[int]:
    for rank, match in enumerate(candidate_matches, start=1):
        if _is_correct_match(benchmark, match.chunk):
            return rank
    return None


def _render_progress(current: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[no work]"
    completed = int(width * current / total)
    bar = "#" * completed + "-" * (width - completed)
    return f"[{bar}] {current}/{total}"


def select_benchmark_records(
    benchmark_records: List[BenchmarkQARecord],
    max_questions: Optional[int] = None,
    sample_size: Optional[int] = None,
    sample_seed: int = 42,
) -> List[BenchmarkQARecord]:
    if max_questions is not None and max_questions <= 0:
        raise ValueError("max_questions must be greater than 0")
    if sample_size is not None and sample_size <= 0:
        raise ValueError("sample_size must be greater than 0")
    if max_questions is not None and sample_size is not None:
        raise ValueError("max_questions and sample_size cannot be used together")

    selected_records = benchmark_records
    if max_questions is not None:
        selected_records = benchmark_records[:max_questions]
    elif sample_size is not None and sample_size < len(benchmark_records):
        rng = random.Random(sample_seed)
        sampled_indices = sorted(rng.sample(range(len(benchmark_records)), sample_size))
        selected_records = [benchmark_records[index] for index in sampled_indices]

    return selected_records


def _build_results_snapshot(
    retriever_name: str,
    variant: str,
    question_count: int,
    hit_at_1_total: float,
    hit_at_5_total: float,
    mrr_at_5_total: float,
    precision_at_1_total: float,
    retrieved_chunks_total: float,
    oracle_candidate_hit_total: float,
    per_question_results: List[Dict[str, object]],
) -> Dict[str, object]:
    total = question_count or 1
    return {
        "retriever_name": retriever_name,
        "variant": variant,
        "question_count": question_count,
        "hit_at_1": hit_at_1_total / total,
        "hit_at_5": hit_at_5_total / total,
        "mrr_at_5": mrr_at_5_total / total,
        "precision_at_1": precision_at_1_total / total,
        "average_retrieved_chunks_per_question": retrieved_chunks_total / total,
        "oracle_candidate_hit_rate": oracle_candidate_hit_total / total,
        "per_question": per_question_results,
    }


def evaluate_ultradomain_retrieval(
    benchmark_records: List[BenchmarkQARecord],
    chunks: List[Chunk],
    retriever_name: str,
    variant: str = "naive",
    top_k: int = 5,
    candidate_pool_size: int = 30,
    log_every: int = 25,
    save_every: Optional[int] = None,
    checkpoint_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    llm_intent_classifier: Optional[LLMIntentClassifier] = None,
    llm_planner: Optional[LLMRetrievalPlanner] = None,
) -> Dict[str, object]:
    intent_classifier = RuleBasedIntentClassifier()
    retriever = create_retriever(retriever_name)

    if save_every is not None and save_every <= 0:
        raise ValueError("save_every must be greater than 0")

    per_question_results = []
    hit_at_1_total = 0.0
    hit_at_5_total = 0.0
    mrr_at_5_total = 0.0
    precision_at_1_total = 0.0
    retrieved_chunks_total = 0.0
    oracle_candidate_hit_total = 0.0
    question_count = len(benchmark_records)
    start_time = time.perf_counter()

    for index, record in enumerate(benchmark_records, start=1):
        if index == 1 or index == question_count or index % log_every == 0:
            elapsed = time.perf_counter() - start_time
            print(
                (
                    f"eval  {_render_progress(index, question_count)} "
                    f"variant={variant} retriever={retriever_name} elapsed={elapsed:.1f}s"
                ),
                flush=True,
            )

        retrieval_output = retrieve_with_variant(
            query=record.question,
            chunks=chunks,
            retriever_name=retriever_name,
            variant=variant,
            retriever=retriever,
            intent_classifier=intent_classifier,
            llm_intent_classifier=llm_intent_classifier,
            llm_planner=llm_planner,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
        )
        matches = retrieval_output["matches"]
        candidate_matches = retrieval_output.get("candidate_matches", matches)
        retrieval_plan = retrieval_output["retrieval_plan"]
        predicted_intent = retrieval_output["detected_intent"]
        if predicted_intent is None:
            predicted_intent = intent_classifier.classify(record.question).intent

        first_correct_candidate_rank = _first_correct_candidate_rank(record, candidate_matches)

        hit_at_1 = 1.0 if matches and _is_correct_match(record, matches[0].chunk) else 0.0
        hit_at_5 = 0.0
        reciprocal_rank = 0.0

        for rank, match in enumerate(matches[:5], start=1):
            if _is_correct_match(record, match.chunk):
                hit_at_5 = 1.0
                reciprocal_rank = 1.0 / rank
                break

        precision_at_1 = hit_at_1
        oracle_candidate_hit = 1.0 if any(_is_correct_match(record, match.chunk) for match in candidate_matches) else 0.0

        hit_at_1_total += hit_at_1
        hit_at_5_total += hit_at_5
        mrr_at_5_total += reciprocal_rank
        precision_at_1_total += precision_at_1
        retrieved_chunks_total += len(matches)
        oracle_candidate_hit_total += oracle_candidate_hit

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
                "oracle_candidate_hit": oracle_candidate_hit,
                "first_correct_candidate_rank": first_correct_candidate_rank,
                "candidate_pool_size": len(candidate_matches),
                "candidate_retrieved": [
                    {
                        "chunk_id": match.chunk.chunk_id,
                        "doc_id": match.chunk.doc_id,
                        "context_id": match.chunk.metadata.get("context_id"),
                        "title": match.chunk.metadata.get("title"),
                        "is_correct": _is_correct_match(record, match.chunk),
                        "original_score": match.original_score if match.original_score is not None else match.score,
                    }
                    for match in candidate_matches
                ],
                "retrieved": [
                    {
                        "chunk_id": match.chunk.chunk_id,
                        "doc_id": match.chunk.doc_id,
                        "title": match.chunk.metadata.get("title"),
                        "context_id": match.chunk.metadata.get("context_id"),
                        "document_type": match.chunk.metadata.get("document_type"),
                        "chunk_type": match.chunk.metadata.get("chunk_type"),
                        "has_tables": match.chunk.metadata.get("has_tables"),
                        "has_table_like_content": match.chunk.metadata.get("has_table_like_content"),
                        "table_line_ratio": match.chunk.metadata.get("table_line_ratio"),
                        "source_file": match.chunk.metadata.get("original_source_file")
                        or match.chunk.metadata.get("source_file")
                        or match.chunk.metadata.get("file_name"),
                        "is_correct": _is_correct_match(record, match.chunk),
                        "original_score": match.original_score if match.original_score is not None else match.score,
                        "adjusted_score": match.adjusted_score if match.adjusted_score is not None else match.score,
                    }
                    for match in matches
                ],
            }
        )

        if checkpoint_callback is not None and save_every is not None and index % save_every == 0:
            checkpoint_callback(
                _build_results_snapshot(
                    retriever_name=retriever_name,
                    variant=variant,
                    question_count=index,
                    hit_at_1_total=hit_at_1_total,
                    hit_at_5_total=hit_at_5_total,
                    mrr_at_5_total=mrr_at_5_total,
                    precision_at_1_total=precision_at_1_total,
                    retrieved_chunks_total=retrieved_chunks_total,
                    oracle_candidate_hit_total=oracle_candidate_hit_total,
                    per_question_results=per_question_results,
                )
            )

    return _build_results_snapshot(
        retriever_name=retriever_name,
        variant=variant,
        question_count=question_count,
        hit_at_1_total=hit_at_1_total,
        hit_at_5_total=hit_at_5_total,
        mrr_at_5_total=mrr_at_5_total,
        precision_at_1_total=precision_at_1_total,
        retrieved_chunks_total=retrieved_chunks_total,
        oracle_candidate_hit_total=oracle_candidate_hit_total,
        per_question_results=per_question_results,
    )


def save_retrieval_results(results: Dict[str, object], json_path: str, csv_path: str) -> None:
    Path(json_path).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    per_question = results.get("per_question", [])
    csv_lines = [
        "question_id,question,predicted_intent,gold_doc_id,gold_context_id,hit_at_1,hit_at_5,mrr_at_5,precision_at_1,oracle_candidate_hit,first_correct_candidate_rank,candidate_pool_size"
    ]
    for item in per_question:
        question = str(item["question"]).replace('"', '""')
        csv_lines.append(
            f'{item["question_id"]},"{question}",{item["predicted_intent"]},{item["gold_doc_id"]},{item["gold_context_id"]},{item["hit_at_1"]},{item["hit_at_5"]},{item["mrr_at_5"]},{item["precision_at_1"]},{item.get("oracle_candidate_hit", 0.0)},{item.get("first_correct_candidate_rank", "")},{item.get("candidate_pool_size", 0)}'
        )
    Path(csv_path).write_text("\n".join(csv_lines), encoding="utf-8")


def save_per_question_debug(results: Dict[str, object], file_path: str) -> None:
    lines = [json.dumps(item, ensure_ascii=False) for item in results.get("per_question", [])]
    Path(file_path).write_text("\n".join(lines), encoding="utf-8")
