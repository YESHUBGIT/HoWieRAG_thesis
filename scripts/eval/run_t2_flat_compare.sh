#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

source "$SCRIPT_DIR/use_eval_env.sh" >/dev/null

QUESTIONS_PATH="${HOWIE_T2_PROCESSED_DIR:-$ROOT_DIR/t2_ragbench_processed_test}/questions.jsonl"
CHUNKS_PATH="${HOWIE_T2_FLAT_INDEX_DIR:-$ROOT_DIR/t2_ragbench_index_flat}/chunks.jsonl"

if [[ ! -f "$QUESTIONS_PATH" || ! -f "$CHUNKS_PATH" ]]; then
  printf 'Missing flat T2 files. Expected %s and %s\n' "$QUESTIONS_PATH" "$CHUNKS_PATH" >&2
  exit 1
fi

python "$ROOT_DIR/scripts/evaluate_ultradomain_retrieval_planning.py" \
  --questions "$QUESTIONS_PATH" \
  --chunks "$CHUNKS_PATH" \
  --retriever bm25 \
  --variant all \
  --sample-size "${HOWIE_SAMPLE_SIZE:-250}" \
  --sample-seed "${HOWIE_SAMPLE_SEED:-42}" \
  --top-k "${HOWIE_TOP_K:-5}" \
  --candidate-pool-size "${HOWIE_CANDIDATE_POOL_SIZE:-30}" \
  --log-every "${HOWIE_LOG_EVERY:-25}" \
  --save-every "${HOWIE_SAVE_EVERY:-25}"
