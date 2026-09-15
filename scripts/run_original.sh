#!/usr/bin/env bash
set -euo pipefail

# run_original.sh — Train the Original model (full fine-tune).
# Usage: run_original.sh <tofu|muse> [model]
#   tofu: full TOFU dataset → serves as pre-unlearning reference (Original)
#   muse: full MUSE-News dataset → serves as pre-unlearning reference (Original)
#   WMDP uses the instruct model directly — no fine-tuning needed.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_finetune.sh" "${1:-tofu}" full "${2:-${MODEL:-}}" "${@:3}"
