#!/usr/bin/env bash
set -euo pipefail

readonly REPO=${V211_NET4_REPO:-/mnt/d/nnue/nnue-pytorch}
readonly VENV=${V211_NET4_VENV:-/home/fifap/venvs/marlinflow/bin/activate}
readonly ROCM_ENV=${V211_NET4_ROCM_ENV:-/mnt/d/nnue/pytorch_paths.sh}
readonly RUN_ROOT=${V211_NET4_RUN_ROOT:-/mnt/d/nnue/runs/v211_net4}
readonly PREFLIGHT_ROOT=${V211_NET4_PREFLIGHT_ROOT:-/mnt/d/nnue/preflight/v211_net4}
readonly PARENT=${V211_NET4_PARENT:-/mnt/d/nnue/v211_parents/v2_5_factorized.pt}
readonly ENGINE=${V211_NET4_ENGINE:-/mnt/d/nnue/SHAYVERI/SHAYVERI}
readonly DATA=${V211_NET4_DATA:-/mnt/d/nnue/robotmoon}

readonly PARENT_SHA=c9b37e262cb917650b445e54d3bb6153d4698b84dcb094c657d75145cd242a79
readonly MANIFEST_SHA=bb3ef74734d8a7e41542e6715e44b718d775608c32ce61308f7cc58a12322c56
readonly BATCH_SIZE=16384
readonly EPOCH_SIZE=1000013824
readonly EPOCHS=80
readonly ONE_CYCLE_STEPS=2441440
readonly REFINEMENT_STEPS=2441440
readonly MAX_LR=0.0004375
readonly REFINEMENT_LR=0.00002
readonly REFINEMENT_FINAL_LR=0.000001

usage() {
  echo "usage: $0 preflight | start | recover CHECKPOINT" >&2
  exit 2
}

write_manifest() {
  local manifest=$1
  mkdir -p "$(dirname "$manifest")"
  sed "s|@DATA@|$DATA|g" > "$manifest" <<'FILES'
@DATA@/farseer_relabel/T60T70wIsRightFarseerT60T74T75T76.split_0.relabel-BT4-tf13tune.binpack
@DATA@/farseer_relabel/T60T70wIsRightFarseerT60T74T75T76.split_1.relabel-BT4-tf13tune.binpack
@DATA@/farseer_relabel/T60T70wIsRightFarseerT60T74T75T76.split_2.relabel-BT4-tf13tune.binpack
@DATA@/farseer_relabel/T60T70wIsRightFarseerT60T74T75T76.split_3.relabel-BT4-tf13tune.binpack
@DATA@/farseer_relabel/T60T70wIsRightFarseerT60T74T75T76.split_4.relabel-BT4-tf13tune.binpack
@DATA@/hard_relabel/dfrc_n5000.relabel-BT4-tf13tune.binpack
@DATA@/hard_relabel/multinet_pv-2_diff-100_nodes-5000.relabel-BT4-tf13tune.binpack
@DATA@/hard_relabel/nodes5000pv2_UHO.relabel-BT4-tf13tune.binpack
@DATA@/hard_relabel/wrongIsRight_nodes5000pv2.relabel-BT4-tf13tune.binpack
@DATA@/leela96_relabel/leela96-filt-v2.min.split_0.relabel-BT4-tf13tune.binpack
@DATA@/leela96_relabel/leela96-filt-v2.min.split_1.relabel-BT4-tf13tune.binpack
@DATA@/leela96_relabel/leela96-filt-v2.min.split_2.relabel-BT4-tf13tune.binpack
@DATA@/leela96_relabel/leela96-filt-v2.min.split_3.relabel-BT4-tf13tune.binpack
@DATA@/leela96_relabel/leela96-filt-v2.min.split_4.relabel-BT4-tf13tune.binpack
@DATA@/t80_2024/test80-2024-01-jan-2tb7p.min-v2.v6.binpack
@DATA@/t80_2024/test80-2024-02-feb-2tb7p.min-v2.v6.binpack
@DATA@/t80_2024/test80-2024-03-mar-2tb7p.min-v2.v6.binpack
@DATA@/t80_2024/test80-2024-04-apr-2tb7p.min-v2.v6.binpack
@DATA@/t80_2024/test80-2024-05-may-2tb7p.min-v2.v6.binpack
@DATA@/t80_2024/test80-2024-06-jun-2tb7p.min-v2.v6.binpack
FILES
}

validate_manifest() {
  local manifest=$1
  local file
  [[ "$(wc -l < "$manifest")" -eq 20 ]] || { echo "manifest must contain 20 files" >&2; exit 1; }
  if [[ "$DATA" == /mnt/d/nnue/robotmoon ]]; then
    actual_manifest_sha=$(sha256sum "$manifest" | cut -d' ' -f1)
    [[ "$actual_manifest_sha" == "$MANIFEST_SHA" ]] || {
      echo "manifest hash mismatch: expected $MANIFEST_SHA, got $actual_manifest_sha" >&2
      exit 1
    }
  fi
  while IFS= read -r file; do
    [[ -f "$file" ]] || { echo "missing corpus file: $file" >&2; exit 1; }
  done < "$manifest"
}

activate_environment() {
  [[ -f "$VENV" ]] || { echo "missing environment: $VENV" >&2; exit 1; }
  [[ -f "$ROCM_ENV" ]] || { echo "missing ROCm environment: $ROCM_ENV" >&2; exit 1; }
  # shellcheck disable=SC1090
  source "$VENV"
  # shellcheck disable=SC1090
  source "$ROCM_ENV"
  unset ROCM_HOME
  cd "$REPO"
}

verify_parent() {
  [[ -f "$PARENT" ]] || { echo "missing parent: $PARENT" >&2; exit 1; }
  actual_parent_sha=$(sha256sum "$PARENT" | cut -d' ' -f1)
  [[ "$actual_parent_sha" == "$PARENT_SHA" ]] || {
    echo "parent hash mismatch: expected $PARENT_SHA, got $actual_parent_sha" >&2
    exit 1
  }
}

checkpoint_step_and_lr() {
  python - "$1" "$2" <<'PY'
import math
import sys
import torch

path, expected_step = sys.argv[1], int(sys.argv[2])
checkpoint = torch.load(path, weights_only=False, map_location="cpu")
step = checkpoint.get("global_step")
if step != expected_step:
    raise SystemExit(f"global_step mismatch: expected {expected_step}, got {step}")
optimizers = checkpoint.get("optimizer_states") or []
schedulers = checkpoint.get("lr_schedulers") or []
if len(optimizers) != 1 or len(schedulers) != 1:
    raise SystemExit("expected exactly one optimizer and scheduler")
rates = [group.get("lr") for group in optimizers[0].get("param_groups", [])]
if not rates or any(rate is None or not math.isfinite(rate) for rate in rates):
    raise SystemExit(f"invalid learning rates: {rates}")
print(f"checkpoint OK: global_step={step}, lr={rates}")
PY
}

run_preflight() {
  [[ ! -e "$PREFLIGHT_ROOT" ]] || {
    echo "refusing to overwrite preflight: $PREFLIGHT_ROOT" >&2
    exit 1
  }
  manifest="$PREFLIGHT_ROOT/train_files.txt"
  write_manifest "$manifest"
  validate_manifest "$manifest"
  activate_environment
  mkdir -p "$PREFLIGHT_ROOT"
  first_file=$(sed -n '1p' "$manifest")
  checkpoint=""

  for target_epoch in 1 2; do
    phase_root="$PREFLIGHT_ROOT/phase_$target_epoch"
    args=(
      "$first_file"
      --architecture=shayveri-direct --features=ShayveriKB16^
      --shayveri-factorizer --loss-function=stockfish
      --lambda=0.74 --start-lambda=0.74 --end-lambda=0.74
      --optimizer-name=rangerlite --lr="$MAX_LR"
      --one-cycle-steps=64 --one-cycle-refinement-steps=64
      --one-cycle-warmup-pct=0.2 --one-cycle-start-div=25 --one-cycle-final-div=50
      --one-cycle-refinement-lr="$REFINEMENT_LR"
      --one-cycle-refinement-final-lr="$REFINEMENT_FINAL_LR"
      --batch-size="$BATCH_SIZE" --epoch-size=1048576 --max-epochs="$target_epoch"
      --validation-size=0 --num-workers=2 --accelerator=cuda --compile-backend=inductor
      --network-save-period=1 --save-top-k=-1 --swa-start-epoch=-1
      --seed=42 --default-root-dir="$phase_root"
    )
    if [[ -n "$checkpoint" ]]; then
      args+=(--resume-from-checkpoint="$checkpoint")
    else
      args+=(--resume-from-model="$PARENT")
    fi
    python -u train.py "${args[@]}" 2>&1 | tee "$PREFLIGHT_ROOT/phase_$target_epoch.log"
    produced=$(find "$phase_root" -path '*/checkpoints/last.ckpt' -type f -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
    [[ -n "$produced" && -f "$produced" ]] || { echo "preflight checkpoint missing" >&2; exit 1; }
    [[ -f "${produced%.ckpt}.nnue" ]] || { echo "automatic NNUE missing" >&2; exit 1; }
    checkpoint="$PREFLIGHT_ROOT/phase_$target_epoch.ckpt"
    cp -- "$produced" "$checkpoint.partial"
    mv -- "$checkpoint.partial" "$checkpoint"
    checkpoint_step_and_lr "$checkpoint" "$((target_epoch * 64))"
  done

  net="$PREFLIGHT_ROOT/phase_2.nnue"
  automatic=$(find "$PREFLIGHT_ROOT/phase_2" -path '*/checkpoints/last.nnue' -type f -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
  cp -- "$automatic" "$net"
  roundtrip="$PREFLIGHT_ROOT/phase_2_roundtrip.nnue"
  python serialize.py "$net" "$roundtrip" --architecture=shayveri-direct \
    --features=ShayveriKB16^ --shayveri-factorizer --ft-compression=none --device=cpu
  cmp -- "$net" "$roundtrip"

  [[ -x "$ENGINE" ]] || { echo "missing SHAYVERI executable: $ENGINE" >&2; exit 1; }
  printf 'uci\nsetoption name EvalFile value %s\nisready\nposition startpos\ngo nodes 1000\nquit\n' "$net" |
    timeout 30s "$ENGINE" > "$PREFLIGHT_ROOT/engine.log"
  grep -q '^bestmove ' "$PREFLIGHT_ROOT/engine.log" || {
    echo "SHAYVERI fixed-node search failed" >&2
    exit 1
  }
  sha256sum "$checkpoint" "$net"
  echo "Net4 preflight passed: resume, phase boundary, NNUE round trip, and engine search verified"
}

run_production() {
  local mode=$1
  local checkpoint=${2:-}
  local manifest="$RUN_ROOT/train_files.txt"
  if [[ "$mode" == start ]]; then
    [[ ! -e "$RUN_ROOT" ]] || { echo "refusing to overwrite run: $RUN_ROOT" >&2; exit 1; }
    mkdir -p "$RUN_ROOT"
    write_manifest "$manifest"
  else
    [[ -f "$checkpoint" ]] || { echo "missing recovery checkpoint: $checkpoint" >&2; exit 1; }
    [[ -f "$manifest" ]] || { echo "missing frozen production manifest" >&2; exit 1; }
    echo "WARNING: recovery restores training state but not the exact multi-file loader cursor" >&2
  fi
  validate_manifest "$manifest"
  mapfile -t files < "$manifest"
  activate_environment

  args=(
    "${files[@]}"
    --architecture=shayveri-direct --features=ShayveriKB16^
    --shayveri-factorizer --loss-function=stockfish
    --lambda=0.74 --start-lambda=0.74 --end-lambda=0.74
    --optimizer-name=rangerlite --lr="$MAX_LR"
    --one-cycle-steps="$ONE_CYCLE_STEPS"
    --one-cycle-warmup-pct=0.2 --one-cycle-start-div=25 --one-cycle-final-div=50
    --one-cycle-refinement-steps="$REFINEMENT_STEPS"
    --one-cycle-refinement-lr="$REFINEMENT_LR"
    --one-cycle-refinement-final-lr="$REFINEMENT_FINAL_LR"
    --batch-size="$BATCH_SIZE" --epoch-size="$EPOCH_SIZE" --max-epochs="$EPOCHS"
    --validation-size=0 --num-workers=2 --accelerator=cuda --compile-backend=inductor
    --network-save-period=5 --save-top-k=-1 --swa-start-epoch=-1
    --seed=42 --default-root-dir="$RUN_ROOT"
  )
  if [[ "$mode" == start ]]; then
    args+=(--resume-from-model="$PARENT")
  else
    checkpoint_step_and_lr "$checkpoint" "$(python - "$checkpoint" <<'PY'
import sys, torch
print(torch.load(sys.argv[1], weights_only=False, map_location="cpu")["global_step"])
PY
    )"
    args+=(--resume-from-checkpoint="$checkpoint")
  fi

  {
    echo "mode=$mode"
    echo "trainer_commit=$(git rev-parse HEAD)"
    echo "parent_sha256=$PARENT_SHA"
    echo "manifest_sha256=$MANIFEST_SHA"
    echo "epoch_size=$EPOCH_SIZE"
    echo "epochs=$EPOCHS"
    echo "accepted_presentations=$((EPOCH_SIZE * EPOCHS))"
    echo "one_cycle_steps=$ONE_CYCLE_STEPS"
    echo "refinement_steps=$REFINEMENT_STEPS"
    echo "lambda=0.74"
    echo "max_lr=$MAX_LR"
    echo "refinement_lr=$REFINEMENT_LR"
    echo "refinement_final_lr=$REFINEMENT_FINAL_LR"
    echo "checkpoint=${checkpoint:-factorized-v2.5-parent}"
    date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
  } | tee -a "$RUN_ROOT/run_history.txt"

  /usr/bin/time -v python -u train.py "${args[@]}" 2>&1 | tee "$RUN_ROOT/train_$mode.log"
  date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ' | tee -a "$RUN_ROOT/run_history.txt"
}

verify_parent
case "${1:-}" in
  preflight) [[ $# -eq 1 ]] || usage; run_preflight ;;
  start) [[ $# -eq 1 ]] || usage; run_production start ;;
  recover) [[ $# -eq 2 ]] || usage; run_production recover "$2" ;;
  *) usage ;;
esac
