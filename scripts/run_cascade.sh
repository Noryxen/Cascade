#!/usr/bin/env bash
set -euo pipefail

# Usage: run_cascade.sh [tofu|muse|wmdp] [Hydra overrides...]
# MODEL, FORGET_SPLIT and RETAIN_SPLIT override YAML only when explicitly set.
DATASET="${1:-tofu}"
if [ "$#" -gt 0 ]; then shift; fi
TASK_NAME="${TASK_NAME:-${DATASET}_Cascade_${MODEL:-default}_${FORGET_SPLIT:-default}}"
EXTRA_ARGS=()
if [ -n "${MODEL:-}" ]; then EXTRA_ARGS+=("model=${MODEL}"); fi

case "${DATASET}" in
  tofu)
    EXTRA_ARGS+=("eval=tofu")
    if [ -n "${FORGET_SPLIT:-}" ]; then
      EXTRA_ARGS+=("dataset_defaults.tofu.forget_split=${FORGET_SPLIT}")
    fi
    if [ -n "${RETAIN_SPLIT:-}" ]; then
      EXTRA_ARGS+=("dataset_defaults.tofu.retain_split=${RETAIN_SPLIT}")
    fi
    ;;
  muse)
    EXTRA_ARGS+=("data/datasets@data.forget=MUSE_forget")
    EXTRA_ARGS+=("data/datasets@data.retain=MUSE_retain")
    EXTRA_ARGS+=("eval=muse")
    ;;
  wmdp)
    EXTRA_ARGS+=("data/datasets@data.forget=WMDP_forget")
    EXTRA_ARGS+=("data/datasets@data.retain=WMDP_retain")
    EXTRA_ARGS+=("eval=lm_eval" "eval.lm_eval.tasks=[wmdp_cyber]")
    ;;
  *)
    echo "Unknown dataset: ${DATASET} (expected tofu|muse|wmdp)" >&2
    exit 1
    ;;
esac

python -u src/train.py --config-name=unlearn.yaml \
  "task_name=${TASK_NAME}" "${EXTRA_ARGS[@]}" "$@"
