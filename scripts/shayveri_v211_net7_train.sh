#!/usr/bin/env bash
set -euo pipefail

readonly REPO=${V211_NET7_REPO:-/mnt/d/nnue/nnue-pytorch}
readonly VENV=${V211_NET7_VENV:-/home/fifap/venvs/marlinflow/bin/activate}
readonly ROCM_ENV=${V211_NET7_ROCM_ENV:-/mnt/d/nnue/pytorch_paths.sh}
readonly DATA=${V211_NET7_DATA:-/mnt/d/nnue/robotmoon/stockfish_new}
readonly RUN_ROOT=${V211_NET7_RUN_ROOT:-/mnt/d/nnue/runs/v211_net7}
readonly PREFLIGHT_ROOT=${V211_NET7_PREFLIGHT_ROOT:-/mnt/d/nnue/preflight/v211_net7}
readonly PARENT=${V211_NET7_PARENT:-/mnt/d/nnue/v211_parents/net1_35B_factorized.pt}
readonly ENGINE=${V211_NET7_ENGINE:-/mnt/d/nnue/SHAYVERI/SHAYVERI}

readonly PARENT_SHA=abe2ce14392ea07d0f2eb3279468299fc546bbd1a14f5367877d1c1adfe684e1
readonly BATCH_SIZE=16384
readonly EPOCH_SIZE=499990528
readonly EPOCHS=10
readonly ONE_CYCLE_STEPS=305170
readonly MAX_LR=0.000020
readonly SEED=42

readonly -a FILES=(
  "$DATA/data_pv-2_diff-100_nodes-5000.binpack"
  "$DATA/fishpack32.binpack"
  "$DATA/test80-2022-08-aug-16tb7p.v6-dd.min.binpack"
  "$DATA/training_data_pylon.binpack"
  "$DATA/wrongNNUE_02_d9.binpack"
)

usage() { echo "usage: $0 preflight | start | recover CHECKPOINT" >&2; exit 2; }

activate_environment() {
  [[ -f "$VENV" ]] || { echo "missing environment: $VENV" >&2; exit 1; }
  [[ -f "$ROCM_ENV" ]] || { echo "missing ROCm environment: $ROCM_ENV" >&2; exit 1; }
  source "$VENV"
  source "$ROCM_ENV"
  unset ROCM_HOME
  cd "$REPO"
}

verify_inputs() {
  [[ -f "$PARENT" ]] || { echo "missing parent: $PARENT" >&2; exit 1; }
  local actual file
  actual=$(sha256sum "$PARENT" | cut -d' ' -f1)
  [[ "$actual" == "$PARENT_SHA" ]] || {
    echo "parent hash mismatch: expected $PARENT_SHA, got $actual" >&2; exit 1;
  }
  [[ "${#FILES[@]}" -eq 5 ]]
  for file in "${FILES[@]}"; do
    [[ -f "$file" ]] || { echo "missing corpus file: $file" >&2; exit 1; }
  done
}

build_args() {
  local root=$1 epochs=$2 epoch_size=${3:-$EPOCH_SIZE} cycle_steps=${4:-$ONE_CYCLE_STEPS}
  TRAIN_ARGS=(
    "${FILES[@]}"
    --architecture=shayveri-direct --features=ShayveriKB16^
    --shayveri-factorizer --loss-function=stockfish
    --lambda=0.74 --start-lambda=0.74 --end-lambda=0.74
    --optimizer-name=rangerlite --lr="$MAX_LR"
    --one-cycle-steps="$cycle_steps"
    --one-cycle-warmup-pct=0.2 --one-cycle-start-div=25 --one-cycle-final-div=50
    --batch-size="$BATCH_SIZE" --epoch-size="$epoch_size" --max-epochs="$epochs"
    --validation-size=0 --num-workers=2 --no-wld-filtered --soft-early-fen-skipping=-1
    --accelerator=cuda --compile-backend=inductor
    --network-save-period=1 --save-top-k=-1 --swa-start-epoch=-1
    --seed="$SEED" --default-root-dir="$root"
  )
}

write_manifest() {
  local output=$1 file
  : > "$output"
  for file in "${FILES[@]}"; do
    printf '%s\t%s\t%s\n' "$(sha256sum "$file" | cut -d' ' -f1)" "$(stat -c %s "$file")" "$file" >> "$output"
  done
}

run_preflight() {
  [[ ! -e "$PREFLIGHT_ROOT" ]] || { echo "refusing to overwrite preflight: $PREFLIGHT_ROOT" >&2; exit 1; }
  mkdir -p "$PREFLIGHT_ROOT"
  write_manifest "$PREFLIGHT_ROOT/train_files.tsv"
  build_args "$PREFLIGHT_ROOT/smoke" 1 1048576 64
  TRAIN_ARGS+=(--resume-from-model="$PARENT")
  python -u train.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$PREFLIGHT_ROOT/train.log"
  local checkpoint net roundtrip
  checkpoint=$(find "$PREFLIGHT_ROOT/smoke" -path '*/checkpoints/last.ckpt' -type f -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
  [[ -n "$checkpoint" && -f "$checkpoint" ]] || { echo "preflight checkpoint missing" >&2; exit 1; }
  net=${checkpoint%.ckpt}.nnue
  [[ -f "$net" ]] || { echo "automatic NNUE missing" >&2; exit 1; }
  roundtrip="$PREFLIGHT_ROOT/roundtrip.nnue"
  python serialize.py "$net" "$roundtrip" --architecture=shayveri-direct \
    --features=ShayveriKB16^ --shayveri-factorizer --ft-compression=none --device=cpu
  cmp -- "$net" "$roundtrip"
  [[ -x "$ENGINE" ]] || { echo "missing engine: $ENGINE" >&2; exit 1; }
  printf 'uci\nsetoption name EvalFile value %s\nisready\nposition startpos\ngo nodes 1000\nquit\n' "$net" |
    timeout 30s "$ENGINE" > "$PREFLIGHT_ROOT/engine.log"
  grep -q '^bestmove ' "$PREFLIGHT_ROOT/engine.log"
  echo "Net7 preflight passed: training, automatic export, round trip, and engine search verified"
}

run_production() {
  local mode=$1 checkpoint=${2:-}
  if [[ "$mode" == start ]]; then
    [[ ! -e "$RUN_ROOT" ]] || { echo "refusing to overwrite run: $RUN_ROOT" >&2; exit 1; }
    mkdir -p "$RUN_ROOT"
    write_manifest "$RUN_ROOT/train_files.tsv"
  else
    [[ -d "$RUN_ROOT" && -f "$RUN_ROOT/train_files.tsv" ]] || { echo "missing run root" >&2; exit 1; }
    [[ -f "$checkpoint" ]] || { echo "missing recovery checkpoint: $checkpoint" >&2; exit 1; }
    echo "WARNING: recovery cannot restore the exact multi-file loader cursor" >&2
  fi
  build_args "$RUN_ROOT" "$EPOCHS"
  if [[ "$mode" == start ]]; then
    TRAIN_ARGS+=(--resume-from-model="$PARENT")
  else
    TRAIN_ARGS+=(--resume-from-checkpoint="$checkpoint")
  fi
  {
    echo "experiment=Net7"
    echo "mode=$mode"
    echo "trainer_commit=$(git rev-parse HEAD)"
    echo "parent_sha256=$PARENT_SHA"
    echo "corpus=stockfish_new"
    echo "epoch_size=$EPOCH_SIZE"
    echo "epochs=$EPOCHS"
    echo "one_cycle_steps=$ONE_CYCLE_STEPS"
    echo "accepted_presentations=$((EPOCH_SIZE * EPOCHS))"
    echo "maximum_lr=$MAX_LR"
    echo "lambda=0.74"
    echo "seed=$SEED"
    date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
  } | tee -a "$RUN_ROOT/run_history.txt"
  /usr/bin/time -v python -u train.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$RUN_ROOT/train_$mode.log"
  date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ' | tee -a "$RUN_ROOT/run_history.txt"
  touch "$RUN_ROOT/training_finished"
}

activate_environment
verify_inputs
case "${1:-}" in
  preflight) [[ $# -eq 1 ]] || usage; run_preflight ;;
  start) [[ $# -eq 1 ]] || usage; run_production start ;;
  recover) [[ $# -eq 2 ]] || usage; run_production recover "$2" ;;
  *) usage ;;
esac

