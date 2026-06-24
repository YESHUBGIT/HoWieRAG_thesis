#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(dirname "$(dirname "$(realpath "$0")")")"

source "$ROOT_DIR/scripts/use_planner_llm_env.sh" >/dev/null

python "$ROOT_DIR/scripts/evaluate_ultradomain_retrieval_planning.py" \
  --questions "$ROOT_DIR/t2_ragbench_processed/questions.jsonl" \
  --chunks "$ROOT_DIR/t2_ragbench_index/chunks.jsonl" \
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
