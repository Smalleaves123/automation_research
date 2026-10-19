#!/usr/bin/env bash
set -e

# Usage:
#   bash scripts/train.sh
#
# If GPU memory is insufficient, edit configs/default.yaml:
#   data.batch_size: 128 -> 64/32
#   train.accum_steps: 1 -> 2/4 (keep the effective batch size)

python -m src.train --config configs/default.yaml