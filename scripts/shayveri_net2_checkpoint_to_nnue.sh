#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 CHECKPOINT OUTPUT.nnue" >&2
  exit 2
}

[[ $# -eq 2 ]] || usage

readonly CHECKPOINT="$1"
readonly OUTPUT="$2"
readonly TRAINER_ROOT=/mnt/d/nnue/nnue-pytorch
readonly VENV=/home/fifap/venvs/marlinflow/bin/activate
readonly ROCM_ENV=/mnt/d/nnue/pytorch_paths.sh

[[ -f "$CHECKPOINT" ]] || {
  echo "Checkpoint does not exist: $CHECKPOINT" >&2
  exit 1
}
[[ "$OUTPUT" == *.nnue ]] || {
  echo "Output must end in .nnue: $OUTPUT" >&2
  exit 1
}
[[ ! -e "$OUTPUT" ]] || {
  echo "Refusing to overwrite existing output: $OUTPUT" >&2
  exit 1
}

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
