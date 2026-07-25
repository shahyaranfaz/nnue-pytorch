#!/usr/bin/env bash
set -euo pipefail

# Control experiment for Net2:
# continue the unchanged Net1 v3 network under Net2's optimizer and LR schedule.
#
#   bash scripts/audit_shayveri_v3_schedule.sh prepare
#   bash scripts/audit_shayveri_v3_schedule.sh start
#   bash scripts/audit_shayveri_v3_schedule.sh export 5
#   bash scripts/audit_shayveri_v3_schedule.sh export 10

readonly TRAINER_ROOT=/mnt/d/nnue/nnue-pytorch
readonly VENV=/home/fifap/venvs/marlinflow/bin/activate
readonly ROCM_ENV=/mnt/d/nnue/pytorch_paths.sh
readonly DATA_ROOT=/mnt/c/bullet_data/v2.9
readonly RUN_ROOT="$DATA_ROOT/runs/net2_audit_v3_schedule"
readonly ARTIFACT_ROOT="$DATA_ROOT/artifacts/net2_audit_v3_schedule"
readonly PARENT_NET="$DATA_ROOT/runs/net1/nets/net1_e40.nnue"
readonly WARM_MODEL="$ARTIFACT_ROOT/net1_e40_direct_factorized.pt"
readonly ROUNDTRIP_NET="$ARTIFACT_ROOT/net1_e40_roundtrip.nnue"
readonly MANIFEST="$DATA_ROOT/runs/net1/train_files.txt"
readonly EXPECTED_MANIFEST_SHA=bb3ef74734d8a7e41542e6715e44b718d775608c32ce61308f7cc58a12322c56

readonly EPOCH_SIZE=1000013824
readonly AUDIT_EPOCHS=10
readonly FULL_RUN_STEPS=4882880

usage() {
  echo "Usage:"
  echo "  $0 prepare"
  echo "  $0 start"
  echo "  $0 export ITERATION"
  exit 2
}

activate_trainer() {
  cd "$TRAINER_ROOT"
  source "$VENV"
  source "$ROCM_ENV"
  unset ROCM_HOME
}

validate_manifest() {
  local actual_sha
  local file

  [[ -f "$MANIFEST" ]] || {
    echo "Missing manifest: $MANIFEST" >&2
    exit 1
  }
  [[ "$(wc -l < "$MANIFEST")" -eq 20 ]] || {
    echo "Manifest must contain exactly 20 files." >&2
    exit 1
  }
  actual_sha="$(sha256sum "$MANIFEST" | cut -d' ' -f1)"
  [[ "$actual_sha" == "$EXPECTED_MANIFEST_SHA" ]] || {
    echo "Manifest hash mismatch: $actual_sha" >&2
    exit 1
  }
  while IFS= read -r file; do
    [[ -f "$file" ]] || {
      echo "Missing training file: $file" >&2
      exit 1
    }
  done < "$MANIFEST"
}

prepare() {
  [[ -f "$PARENT_NET" ]] || {
    echo "Missing Net1 parent: $PARENT_NET" >&2
    exit 1
  }
  [[ ! -e "$ARTIFACT_ROOT" ]] || {
    echo "Audit artifacts already exist: $ARTIFACT_ROOT" >&2
    exit 1
  }
  mkdir -p "$ARTIFACT_ROOT"

  activate_trainer
  python serialize.py "$PARENT_NET" "$WARM_MODEL" \
    --architecture=shayveri-direct \
    --features='ShayveriKB16^' \
    --shayveri-factorizer

  python serialize.py "$WARM_MODEL" "$ROUNDTRIP_NET" \
    --architecture=shayveri-direct \
    --features='ShayveriKB16^' \
    --shayveri-factorizer \
    --ft-compression=none

  cmp "$PARENT_NET" "$ROUNDTRIP_NET"
  sha256sum "$PARENT_NET" "$ROUNDTRIP_NET" "$WARM_MODEL" |
    tee "$ARTIFACT_ROOT/sha256.txt"
  echo "Byte-identical v3 audit warm start OK"
}

start() {
  validate_manifest
  [[ -f "$WARM_MODEL" ]] || {
    echo "Run prepare first: $WARM_MODEL is missing" >&2
    exit 1
  }
  [[ ! -e "$RUN_ROOT" ]] || {
    echo "Audit run already exists: $RUN_ROOT" >&2
    exit 1
  }
  mkdir -p "$RUN_ROOT"
  cp "$MANIFEST" "$RUN_ROOT/train_files.txt"
  mapfile -t TRAIN_FILES < "$MANIFEST"

  activate_trainer
  {
    echo "control=unchanged-v3-net1-e40"
    echo "parent=$PARENT_NET"
    echo "epoch_size=$EPOCH_SIZE"
    echo "audit_epochs=$AUDIT_EPOCHS"
    echo "full_run_steps=$FULL_RUN_STEPS"
    echo "manifest_sha=$EXPECTED_MANIFEST_SHA"
    date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
  } | tee "$RUN_ROOT/run_history.txt"

  /usr/bin/time -v python train.py \
    "${TRAIN_FILES[@]}" \
    --resume-from-model="$WARM_MODEL" \
    --architecture=shayveri-direct \
    --features='ShayveriKB16^' \
    --shayveri-factorizer \
    --loss-function=stockfish \
    --lambda=0.74 \
    --optimizer-name=rangerlite \
    --lr=0.0004375 \
    --one-cycle-steps="$FULL_RUN_STEPS" \
    --batch-size=16384 \
    --epoch-size="$EPOCH_SIZE" \
    --max-epochs="$AUDIT_EPOCHS" \
    --validation-size=0 \
    --num-workers=2 \
    --accelerator=cuda \
    --compile-backend=inductor \
    --network-save-period=5 \
    --save-top-k=-1 \
    --swa-start-epoch=-1 \
    --default-root-dir="$RUN_ROOT" \
    2>&1 | tee "$RUN_ROOT/train.log"
}

export_iteration() {
  local iteration="$1"
  local epoch=$((iteration - 1))
  local checkpoint
  local output="$RUN_ROOT/nets/audit_v3_e${iteration}.nnue"

  [[ "$iteration" == "5" || "$iteration" == "10" ]] || usage
  checkpoint="$(
    find "$RUN_ROOT" -path "*/checkpoints/epoch=${epoch}-step=*.ckpt" \
      -printf '%T@ %p\n' |
      sort -n |
      tail -1 |
      cut -d' ' -f2-
  )"
  [[ -f "$checkpoint" ]] || {
    echo "Missing checkpoint for iteration $iteration" >&2
    exit 1
  }
  [[ ! -e "$output" ]] || {
    echo "Output already exists: $output" >&2
    exit 1
  }
  mkdir -p "$RUN_ROOT/nets"

  activate_trainer
  python serialize.py "$checkpoint" "$output" \
    --architecture=shayveri-direct \
    --features='ShayveriKB16^' \
    --shayveri-factorizer \
    --ft-compression=none
  sha256sum "$output"
}

[[ $# -ge 1 ]] || usage
case "$1" in
  prepare)
    [[ $# -eq 1 ]] || usage
    prepare
    ;;
  start)
    [[ $# -eq 1 ]] || usage
    start
    ;;
  export)
    [[ $# -eq 2 ]] || usage
    export_iteration "$2"
    ;;
  *)
    usage
    ;;
esac
