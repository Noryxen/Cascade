#!/usr/bin/env bash
set -euo pipefail

# run_finetune.sh — Fine-tune a base model on TOFU or MUSE-News.
# Usage: run_finetune.sh <dataset> <split> [model]
#   dataset: tofu | muse
#   split: full | retain90 (tofu) | full | retain (muse)
#   model: inherits the YAML default when omitted
# Additional Hydra overrides may follow the model argument.

DATASET="${1:?Usage: run_finetune.sh <tofu|muse> <split> [model]}"
SPLIT="${2:?Usage: run_finetune.sh <tofu|muse> <split> [model]}"
MODEL="${3:-${MODEL:-}}"
TASK_NAME="${TASK_NAME:-${DATASET}_${MODEL:-default}_${SPLIT}}"
if [ "$#" -ge 3 ]; then shift 3; else shift 2; fi

EXTRA_ARGS=()
if [ -n "${MODEL}" ]; then EXTRA_ARGS+=("model=${MODEL}"); fi

case "${DATASET}" in
  tofu)
    EXTRA_ARGS+=("eval=tofu")
    case "${SPLIT}" in
      full)
        EXTRA_ARGS+=("data/datasets@data.train=TOFU_QA_full")
        ;;
      retain90)
        EXTRA_ARGS+=("data/datasets@data.train=TOFU_QA_retain")
        ;;
      *)
        echo "Unknown split for tofu: ${SPLIT} (expected full|retain90)" >&2
        exit 1
        ;;
    esac
    ;;
  muse)
    EXTRA_ARGS+=("eval=muse")
    case "${SPLIT}" in
      full)
        EXTRA_ARGS+=("data/datasets@data.train=MUSE_train")
        EXTRA_ARGS+=("data.train.MUSE_train.args.hf_args.split=full")
        ;;
      retain)
        EXTRA_ARGS+=("data/datasets@data.train=MUSE_train")
        EXTRA_ARGS+=("data.train.MUSE_train.args.hf_args.split=retain")
        ;;
      *)
        echo "Unknown split for muse: ${SPLIT} (expected full|retain)" >&2
        exit 1
        ;;
    esac
    ;;
  *)
    echo "Unknown dataset: ${DATASET} (expected tofu|muse)" >&2
    exit 1
    ;;
esac

python -u src/train.py \
  --config-name=train.yaml \
  task_name="${TASK_NAME}" \
  "${EXTRA_ARGS[@]}" "$@"
