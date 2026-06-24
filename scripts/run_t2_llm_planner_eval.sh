#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(dirname "$(dirname "$(realpath "$0")")")"

exec "$ROOT_DIR/scripts/eval/run_t2_llm_planner_eval.sh"
