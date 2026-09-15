#!/usr/bin/env bash
set -euo pipefail

# Usage: run_eval.sh <model_path> <task_name> [forget_split] [holdout_split] [retain_logs_path] [Hydra overrides...]
# Optional values inherit YAML defaults when omitted or empty.
MODEL_PATH="${1:?Usage: run_eval.sh <model_path> <task_name> [forget_split] [holdout_split] [retain_logs_path]}"
TASK_NAME="${2:?Usage: run_eval.sh <model_path> <task_name> [forget_split] [holdout_split] [retain_logs_path]}"
EXTRA_ARGS=()
if [ -n "${MODEL:-}" ]; then EXTRA_ARGS+=("model=${MODEL}"); fi
if [ -n "${3:-}" ]; then EXTRA_ARGS+=("dataset_defaults.tofu.forget_split=$3"); fi
if [ -n "${4:-}" ]; then EXTRA_ARGS+=("dataset_defaults.tofu.holdout_split=$4"); fi
if [ -n "${5:-}" ]; then EXTRA_ARGS+=("eval.tofu.retain_logs_path=$5"); fi
if [ "$#" -ge 5 ]; then shift 5; else shift "$#"; fi

python -u src/eval.py --config-name=eval.yaml \
  "model.model_args.pretrained_model_name_or_path=${MODEL_PATH}" \
  "task_name=${TASK_NAME}" "${EXTRA_ARGS[@]}" "$@"
