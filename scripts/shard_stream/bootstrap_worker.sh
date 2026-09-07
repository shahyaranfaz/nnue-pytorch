#!/usr/bin/env bash
set -euo pipefail

readonly LOCAL_ROOT=${V211_LOCAL_ROOT:-/tmp/anfazsha-v211}
readonly REPO=${V211_REPO:-/student/anfazsha/nnue-pytorch}
readonly SMOKE_SHARD=${1:-}

[[ -f "$REPO/train.py" ]] || {
  echo "Missing nnue-pytorch checkout: $REPO" >&2
  exit 1
}

mkdir -p "$LOCAL_ROOT"

if [[ ! -x "$LOCAL_ROOT/venv/bin/python" ]]; then
  echo "Creating local venv at $LOCAL_ROOT/venv"
  python3 -m venv --system-site-packages "$LOCAL_ROOT/venv"
fi

# shellcheck disable=SC1091
source "$REPO/scripts/shard_stream/worker_env.sh"

python -m pip install --disable-pip-version-check --no-cache-dir \
  -r "$REPO/requirements.txt"

python - <<'PY'
import lightning
import torch
import tyro

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("lightning", lightning.__version__)
print("tyro", tyro.__version__ if hasattr(tyro, "__version__") else "available")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print("gpu", torch.cuda.get_device_name(0))
PY

[[ -f "$REPO/build/libtraining_data_loader.so" ]] || {
  echo "Missing native loader: $REPO/build/libtraining_data_loader.so" >&2
  exit 1
}
[[ -x "$REPO/build/training_data_loader_bench" ]] || {
  echo "Missing loader benchmark: $REPO/build/training_data_loader_bench" >&2
  exit 1
}

if [[ -n "$SMOKE_SHARD" ]]; then
  [[ -f "$SMOKE_SHARD" ]] || {
    echo "Missing smoke shard: $SMOKE_SHARD" >&2
    exit 1
  }
  "$REPO/build/training_data_loader_bench" -i 5 -p 2 -c 0 "$SMOKE_SHARD"
fi

echo "Worker bootstrap complete: $(hostname)"
du -sh "$LOCAL_ROOT"
