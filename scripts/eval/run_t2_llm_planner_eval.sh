#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

source "$SCRIPT_DIR/use_eval_env.sh" >/dev/null

QUESTIONS_PATH="$HOWIE_T2_PROCESSED_DIR/questions.jsonl"
CHUNKS_PATH="$HOWIE_T2_INDEX_DIR/chunks.jsonl"

if [[ ! -f "$QUESTIONS_PATH" ]]; then
  printf 'Missing questions file: %s\n' "$QUESTIONS_PATH" >&2
  printf 'Run: python scripts/prepare_t2_ragbench.py <dataset_path> %s\n' "$HOWIE_T2_PROCESSED_DIR" >&2
  exit 1
fi

if [[ ! -f "$CHUNKS_PATH" ]]; then
  printf 'Missing chunks file: %s\n' "$CHUNKS_PATH" >&2
  printf 'Run: python scripts/build_ultradomain_index.py %s/documents.jsonl %s\n' "$HOWIE_T2_PROCESSED_DIR" "$HOWIE_T2_INDEX_DIR" >&2
  exit 1
fi

python "$ROOT_DIR/scripts/evaluate_ultradomain_retrieval_planning.py" \
  --questions "$QUESTIONS_PATH" \
  --chunks "$CHUNKS_PATH" \
  --retriever bm25 \
  --variant llm_document_aware \
  --sample-size "${HOWIE_SAMPLE_SIZE:-250}" \
  --sample-seed "${HOWIE_SAMPLE_SEED:-42}" \
  --top-k "${HOWIE_TOP_K:-5}" \
  --candidate-pool-size "${HOWIE_CANDIDATE_POOL_SIZE:-30}" \
  --log-every "${HOWIE_LOG_EVERY:-25}" \
  --save-every "${HOWIE_SAVE_EVERY:-25}" \
  --planner-base-url "$HOWIE_PLANNER_LLM_BASE_URL" \
  --planner-model "$HOWIE_PLANNER_LLM_MODEL"
