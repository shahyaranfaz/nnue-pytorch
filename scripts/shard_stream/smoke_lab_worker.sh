#!/usr/bin/env bash
set -euo pipefail

readonly REPO=${V211_REPO:-/student/anfazsha/nnue-pytorch}
readonly ROOT=${V211_ROOT:-/student/anfazsha/v2_11}
readonly LOCAL_ROOT=${V211_LOCAL_ROOT:-/tmp/anfazsha-v211}
readonly BATCH_SIZE=16384
readonly EPOCH_SIZE=1048576
readonly FULL_STEPS=128

case "$(hostname -s)" in
  dh2010pc16) readonly LANE=lane_a; readonly LAMBDA=0.74; readonly LR=0.0004375; readonly PARENT="$ROOT/parents/v2_5_factorized.pt"; readonly PARENT_SHA=c9b37e262cb917650b445e54d3bb6153d4698b84dcb094c657d75145cd242a79 ;;
  dh2010pc19) readonly LANE=lane_b; readonly LAMBDA=0.74; readonly LR=0.0000200; readonly PARENT="$ROOT/parents/net1_35B_factorized.pt"; readonly PARENT_SHA=abe2ce14392ea07d0f2eb3279468299fc546bbd1a14f5367877d1c1adfe684e1 ;;
  dh2010pc22) readonly LANE=lane_c; readonly LAMBDA=0.90; readonly LR=0.0000200; readonly PARENT="$ROOT/parents/net1_35B_factorized.pt"; readonly PARENT_SHA=abe2ce14392ea07d0f2eb3279468299fc546bbd1a14f5367877d1c1adfe684e1 ;;
  dh2010pc25) readonly LANE=lane_d; readonly LAMBDA=0.74; readonly LR=0.0000200; readonly PARENT="$ROOT/parents/net1_35B_factorized.pt"; readonly PARENT_SHA=abe2ce14392ea07d0f2eb3279468299fc546bbd1a14f5367877d1c1adfe684e1 ;;
  *) echo "No smoke lane is assigned to $(hostname -s)" >&2; exit 1 ;;
esac

readonly SHARD=${SMOKE_SHARD:-$ROOT/ready/v210_00000.binpack}
readonly SMOKE_ROOT="$LOCAL_ROOT/smoke/$LANE"
[[ -f "$SHARD" ]] || { echo "Missing smoke shard: $SHARD" >&2; exit 1; }
[[ -f "$PARENT" ]] || { echo "Missing frozen parent: $PARENT" >&2; exit 1; }
actual_parent_sha=$(sha256sum "$PARENT" | cut -d' ' -f1)
[[ "$actual_parent_sha" == "$PARENT_SHA" ]] || {
  echo "Parent hash mismatch: expected $PARENT_SHA, got $actual_parent_sha" >&2
  exit 1
}
[[ ! -e "$SMOKE_ROOT" ]] || {
  echo "Refusing to overwrite prior smoke output: $SMOKE_ROOT" >&2
  echo "Remove that exact directory only if you intend to rerun the disposable smoke test." >&2
  exit 1
}

# shellcheck disable=SC1091
source "$REPO/scripts/shard_stream/worker_env.sh"
mkdir -p "$SMOKE_ROOT"

checkpoint=""
for target_epoch in 1 2; do
  phase_root="$SMOKE_ROOT/phase_$target_epoch"
  args=(
    "$SHARD"
    --architecture=shayveri-direct --features=ShayveriKB16^
    --shayveri-factorizer --loss-function=stockfish
    --lambda="$LAMBDA" --start-lambda="$LAMBDA" --end-lambda="$LAMBDA"
    --optimizer-name=rangerlite --lr="$LR" --one-cycle-steps="$FULL_STEPS"
    --batch-size="$BATCH_SIZE" --epoch-size="$EPOCH_SIZE"
    --max-epochs="$target_epoch" --validation-size=0 --num-workers=2
    --accelerator=cuda --compile-backend=inductor
    --network-save-period=1 --save-top-k=1 --swa-start-epoch=-1
    --seed=42 --default-root-dir="$phase_root"
  )
  if [[ "$LANE" != lane_a ]]; then
    args+=(--no-wld-filtered --soft-early-fen-skipping=-1)
  fi
  if [[ -n "$checkpoint" ]]; then
    args+=(--resume-from-checkpoint="$checkpoint")
  else
    args+=(--resume-from-model="$PARENT")
  fi

  echo "Smoke phase $target_epoch/2 for $LANE"
  python -u train.py "${args[@]}" 2>&1 | tee "$SMOKE_ROOT/phase_$target_epoch.log"
  produced=$(find "$phase_root" -path '*/checkpoints/last.ckpt' -type f -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
  [[ -n "$produced" && -f "$produced" ]] || { echo "No phase $target_epoch checkpoint produced" >&2; exit 1; }
  automatic_net="${produced%.ckpt}.nnue"
  [[ -f "$automatic_net" ]] || { echo "No automatic phase $target_epoch NNUE export produced" >&2; exit 1; }
  checkpoint="$SMOKE_ROOT/phase_$target_epoch.ckpt"
  cp -- "$produced" "$checkpoint.partial"
  mv -- "$checkpoint.partial" "$checkpoint"

  python - "$checkpoint" "$((target_epoch * EPOCH_SIZE / BATCH_SIZE))" <<'PY'
import math
import sys
import torch

path, expected_step = sys.argv[1], int(sys.argv[2])
checkpoint = torch.load(path, weights_only=False, map_location="cpu")
actual_step = checkpoint.get("global_step")
if actual_step != expected_step:
    raise SystemExit(f"global_step mismatch: expected {expected_step}, got {actual_step}")
optimizers = checkpoint.get("optimizer_states") or []
schedulers = checkpoint.get("lr_schedulers") or []
if len(optimizers) != 1 or len(schedulers) != 1:
    raise SystemExit("checkpoint did not preserve exactly one optimizer and scheduler")
rates = [group.get("lr") for group in optimizers[0].get("param_groups", [])]
if not rates or any(rate is None or not math.isfinite(rate) for rate in rates):
    raise SystemExit(f"invalid optimizer learning rates: {rates}")
print(f"checkpoint continuity OK: global_step={actual_step}, lr={rates}")
PY
done

net="$SMOKE_ROOT/${LANE}_smoke.nnue"
roundtrip="$SMOKE_ROOT/${LANE}_smoke_roundtrip.nnue"
python serialize.py "$checkpoint" "$net" \
  --architecture=shayveri-direct --features=ShayveriKB16^ \
  --shayveri-factorizer --ft-compression=none --device=cpu
python serialize.py "$net" "$roundtrip" \
  --architecture=shayveri-direct --features=ShayveriKB16^ \
  --shayveri-factorizer --ft-compression=none --device=cpu
cmp -- "$net" "$roundtrip"
sha256sum "$checkpoint" "$net"
echo "Smoke passed for $LANE: 2,097,152 presentations, resume and NNUE round trip verified"
