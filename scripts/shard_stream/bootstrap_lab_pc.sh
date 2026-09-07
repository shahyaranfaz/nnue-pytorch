#!/usr/bin/env bash
set -euo pipefail

readonly REPO=${V211_REPO:-/student/anfazsha/nnue-pytorch}
readonly LOCAL_ROOT=${V211_LOCAL_ROOT:-/tmp/anfazsha-v211}

case "$(hostname -s)" in
  dh2010pc16|dh2010pc19|dh2010pc22|dh2010pc25) ;;
  *) echo "This bootstrap is only for the four configured training PCs." >&2; exit 1 ;;
esac

[[ -f "$REPO/train.py" ]] || { echo "Missing trainer: $REPO" >&2; exit 1; }
mkdir -p "$LOCAL_ROOT"
if [[ ! -x "$LOCAL_ROOT/venv/bin/python" ]]; then
  python3 -m venv --system-site-packages "$LOCAL_ROOT/venv"
fi

# shellcheck disable=SC1091
source "$REPO/scripts/shard_stream/worker_env.sh"
python -m pip install --disable-pip-version-check --no-cache-dir -r requirements.txt
python - <<'PY'
import lightning
import torch
import tyro
assert torch.cuda.is_available()
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY

test -x build/training_data_loader_bench
smoke_shard=$(find /student/anfazsha/v2_11/ready -maxdepth 1 -type f -name '*.binpack' -print -quit)
if [[ -n "$smoke_shard" ]]; then
  build/training_data_loader_bench -i 5 -p 2 -c 0 "$smoke_shard"
else
  echo "No ready shard; skipping loader smoke test"
fi
echo "Bootstrap complete on $(hostname -s); local usage: $(du -sh "$LOCAL_ROOT" | cut -f1)"
