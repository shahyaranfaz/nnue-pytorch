#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 ITERATION" >&2
  echo "Example: $0 20" >&2
  exit 2
}

[[ $# -eq 1 ]] || usage
[[ "$1" =~ ^[1-9][0-9]*$ ]] || usage

readonly TRAINER_ROOT=/mnt/d/nnue/nnue-pytorch
readonly VENV=/home/fifap/venvs/marlinflow/bin/activate
readonly ROCM_ENV=/mnt/d/nnue/pytorch_paths.sh
readonly RUN_ROOT=/mnt/c/bullet_data/v2.9/runs/net2
readonly ITERATION=$((10#$1))
readonly EPOCH_INDEX=$((ITERATION - 1))

CHECKPOINT="$(
  find "$RUN_ROOT" \
    -path "*/checkpoints/epoch=${EPOCH_INDEX}-step=*.ckpt" \
    -type f -printf '%T@ %p\n' |
    sort -n |
    tail -1 |
    cut -d' ' -f2-
)"
[[ -n "$CHECKPOINT" && -f "$CHECKPOINT" ]] || {
  echo "No checkpoint found for iteration $ITERATION (epoch index $EPOCH_INDEX)." >&2
  exit 1
}

printf -v ITERATION_LABEL '%02d' "$ITERATION"
readonly OUTPUT="$RUN_ROOT/nets/net2_e${ITERATION_LABEL}.nnue"
[[ ! -e "$OUTPUT" ]] || {
  echo "Refusing to overwrite existing output: $OUTPUT" >&2
  exit 1
}

echo "iteration=$ITERATION"
echo "checkpoint=$CHECKPOINT"
echo "output=$OUTPUT"

mkdir -p "$(dirname "$OUTPUT")"
cd "$TRAINER_ROOT"
source "$VENV"
source "$ROCM_ENV"
unset ROCM_HOME

python serialize.py "$CHECKPOINT" "$OUTPUT" \
  --architecture=shayveri-bucketed \
  --features='ShayveriKB16^' \
  --shayveri-factorizer \
  --ft-compression=none

python - "$OUTPUT" <<'PY'
import os
import struct
import sys

path = sys.argv[1]
expected_header = (
    0x4E4E5545,  # magic
    4,           # format version
    16,          # king buckets
    512,         # hidden width
    8,           # output buckets
    1,           # SCReLU
    12_582_912,  # feature weights
    1_024,       # feature bias
    16_384,      # output weights
    32,          # output bias
)
expected_size = 40 + sum(expected_header[6:])

with open(path, "rb") as stream:
    header = struct.unpack("<10I", stream.read(40))

if header != expected_header:
    raise SystemExit(f"Unexpected Net2 header: {header}")
if os.path.getsize(path) != expected_size:
    raise SystemExit(
        f"Unexpected Net2 size: {os.path.getsize(path)}, expected {expected_size}"
    )

print(f"Net2 v4 export verified: {path} ({expected_size} bytes)")
PY

sha256sum "$OUTPUT"
