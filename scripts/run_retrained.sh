#!/usr/bin/env bash
set -euo pipefail

# run_retrained.sh — Train the Retrained model (retain-only dataset).
# Usage: run_retrained.sh <tofu|muse> [model]
#   tofu: retain90 split → ideal upper bound (Retrained) in the paper
#   muse: retain split → ideal upper bound (Retrained) in the paper

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-tofu}"
SPLIT="retain90"
if [ "${DATASET}" = "muse" ]; then
  SPLIT="retain"
fi
bash "${SCRIPT_DIR}/run_finetune.sh" "${DATASET}" "${SPLIT}" "${2:-${MODEL:-}}" "${@:3}"
