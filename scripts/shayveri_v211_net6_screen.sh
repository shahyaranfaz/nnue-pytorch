#!/usr/bin/env bash
set -euo pipefail

readonly REPO=${V211_NET6_REPO:-/mnt/d/nnue/nnue-pytorch}
readonly VENV=${V211_NET6_VENV:-/home/fifap/venvs/marlinflow/bin/activate}
readonly ROCM_ENV=${V211_NET6_ROCM_ENV:-/mnt/d/nnue/pytorch_paths.sh}
readonly DATA=${V211_NET6_DATA:-/mnt/d/nnue/robotmoon}
readonly RUN_ROOT=${V211_NET6_RUN_ROOT:-/mnt/d/nnue/runs/v211_net6}
readonly PREFLIGHT_ROOT=${V211_NET6_PREFLIGHT_ROOT:-/mnt/d/nnue/preflight/v211_net6}
readonly PARENT=${V211_NET6_PARENT:-/mnt/d/nnue/v211_parents/net1_35B_factorized.pt}
readonly ENGINE=${V211_NET6_ENGINE:-/mnt/d/nnue/SHAYVERI/SHAYVERI}

readonly PARENT_SHA=abe2ce14392ea07d0f2eb3279468299fc546bbd1a14f5367877d1c1adfe684e1
readonly BATCH_SIZE=16384
readonly EPOCH_SIZE=399933440
readonly EPOCHS=2
readonly ONE_CYCLE_STEPS=48820
readonly MAX_LR=0.000020
readonly SEED=42

readonly -a T80_2023=(
  "$DATA/t80_2023/test80-2023-01-jan-16tb7p.v6-sk20.min.binpack"
  "$DATA/t80_2023/test80-2023-02-feb-16tb7p.v6-dd.min.binpack"
  "$DATA/t80_2023/test80-2023-03-mar-2tb7p.v6-sk16.min.binpack"
  "$DATA/t80_2023/test80-2023-04-apr-2tb7p.v6-sk16.min.binpack"
  "$DATA/t80_2023/test80-2023-05-may-2tb7p.v6.min.binpack"
  "$DATA/t80_2023/test80-2023-06-jun-2tb7p.min-v2.v6.binpack"
  "$DATA/t80_2023/test80-2023-07-jul-2tb7p.min-v2.v6.binpack"
  "$DATA/t80_2023/test80-2023-08-aug-2tb7p.v6.min.binpack"
  "$DATA/t80_2023/test80-2023-09-sep-2tb7p.min-v2.v6.binpack"
  "$DATA/t80_2023/test80-2023-10-oct-2tb7p.min-v2.v6.binpack"
  "$DATA/t80_2023/test80-2023-11-nov-2tb7p.min-v2.v6.binpack"
  "$DATA/t80_2023/test80-2023-12-dec-2tb7p.min-v2.v6.binpack"
)
readonly -a STOCKFISH_NEW=(
  "$DATA/stockfish_new/data_pv-2_diff-100_nodes-5000.binpack"
  "$DATA/stockfish_new/fishpack32.binpack"
  "$DATA/stockfish_new/test80-2022-08-aug-16tb7p.v6-dd.min.binpack"
  "$DATA/stockfish_new/training_data_pylon.binpack"
  "$DATA/stockfish_new/wrongNNUE_02_d9.binpack"
)
readonly -a T80_2024=(
  "$DATA/t80_2024/test80-2024-01-jan-2tb7p.min-v2.v6.binpack"
  "$DATA/t80_2024/test80-2024-02-feb-2tb7p.min-v2.v6.binpack"
  "$DATA/t80_2024/test80-2024-03-mar-2tb7p.min-v2.v6.binpack"
  "$DATA/t80_2024/test80-2024-04-apr-2tb7p.min-v2.v6.binpack"
  "$DATA/t80_2024/test80-2024-05-may-2tb7p.min-v2.v6.binpack"
  "$DATA/t80_2024/test80-2024-06-jun-2tb7p.min-v2.v6.binpack"
)

usage() {
  echo "usage: $0 preflight | run A|B|C|D | recover A|B|C|D CHECKPOINT | all" >&2
  exit 2
}

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
  local actual
  actual=$(sha256sum "$PARENT" | cut -d' ' -f1)
  [[ "$actual" == "$PARENT_SHA" ]] || {
    echo "parent hash mismatch: expected $PARENT_SHA, got $actual" >&2
    exit 1
  }
  [[ "${#T80_2023[@]}" -eq 12 && "${#STOCKFISH_NEW[@]}" -eq 5 && "${#T80_2024[@]}" -eq 6 ]]
  local file
  for file in "${T80_2023[@]}" "${STOCKFISH_NEW[@]}" "${T80_2024[@]}"; do
    [[ -f "$file" ]] || { echo "missing corpus file: $file" >&2; exit 1; }
  done
}

lane_name() {
  case "${1^^}" in
    A) echo t80_2023 ;;
    B) echo stockfish_new ;;
    C) echo t80_2024 ;;
    D) echo mix_9_7_4 ;;
    *) usage ;;
  esac
}

write_manifest() {
  local lane=${1^^}
  local output=$2
  : > "$output"
  case "$lane" in
    A) printf 'primary\t%s\n' "${T80_2023[@]}" >> "$output" ;;
    B) printf 'primary\t%s\n' "${STOCKFISH_NEW[@]}" >> "$output" ;;
    C) printf 'primary\t%s\n' "${T80_2024[@]}" >> "$output" ;;
    D)
      printf 'primary\t%s\n' "${T80_2023[@]}" >> "$output"
      printf 'secondary\t%s\n' "${STOCKFISH_NEW[@]}" >> "$output"
      printf 'tertiary\t%s\n' "${T80_2024[@]}" >> "$output"
      ;;
  esac
}

build_args() {
  local lane=${1^^}
  local root=$2
  local max_epochs=$3
  local epoch_size=${4:-$EPOCH_SIZE}
  local cycle_steps=${5:-$ONE_CYCLE_STEPS}
  TRAIN_ARGS=()
  case "$lane" in
    A) TRAIN_ARGS+=("${T80_2023[@]}") ;;
    B) TRAIN_ARGS+=("${STOCKFISH_NEW[@]}") ;;
    C) TRAIN_ARGS+=("${T80_2024[@]}") ;;
    D)
      TRAIN_ARGS+=("${T80_2023[@]}")
      local file
      for file in "${STOCKFISH_NEW[@]}"; do TRAIN_ARGS+=(--secondary-datasets "$file"); done
      for file in "${T80_2024[@]}"; do TRAIN_ARGS+=(--tertiary-datasets "$file"); done
      TRAIN_ARGS+=(--secondary-batches-per-cycle=7 --tertiary-batches-per-cycle=4 --mix-cycle-batches=20)
      ;;
  esac
  TRAIN_ARGS+=(
    --architecture=shayveri-direct --features=ShayveriKB16^
    --shayveri-factorizer --loss-function=stockfish
    --lambda=0.74 --start-lambda=0.74 --end-lambda=0.74
    --optimizer-name=rangerlite --lr="$MAX_LR"
    --one-cycle-steps="$cycle_steps"
    --one-cycle-warmup-pct=0.2 --one-cycle-start-div=25 --one-cycle-final-div=50
    --batch-size="$BATCH_SIZE" --epoch-size="$epoch_size" --max-epochs="$max_epochs"
    --validation-size=0 --num-workers=2 --no-wld-filtered --soft-early-fen-skipping=-1
    --accelerator=cuda --compile-backend=inductor
    --network-save-period=1 --save-top-k=-1 --swa-start-epoch=-1
    --seed="$SEED" --default-root-dir="$root"
  )
}

collect_gates() {
  local lane=${1^^}
  local root=$2
  local out="$root/gates"
  mkdir -p "$out"
  local epoch presentations checkpoint net target
  for epoch in 0 1; do
    presentations=$(((epoch + 1) * EPOCH_SIZE))
    checkpoint=$(find "$root" -path "*/checkpoints/epoch=$epoch-step=*.ckpt" -type f -printf '%T@ %p\n' |
      sort -nr | head -1 | cut -d' ' -f2-)
    [[ -n "$checkpoint" && -f "$checkpoint" ]] || {
      echo "missing epoch $epoch checkpoint for Net6$lane" >&2
      exit 1
    }
    net="${checkpoint%.ckpt}.nnue"
    [[ -f "$net" ]] || { echo "missing automatic NNUE: $net" >&2; exit 1; }
    target="$out/net6${lane}_p${presentations}"
    cp -- "$checkpoint" "$target.ckpt.partial"
    mv -- "$target.ckpt.partial" "$target.ckpt"
    cp -- "$net" "$target.nnue.partial"
    mv -- "$target.nnue.partial" "$target.nnue"
    sha256sum "$target.ckpt" "$target.nnue" > "$target.sha256"
  done
}

run_lane() {
  local lane=${1^^}
  local mode=$2
  local checkpoint=${3:-}
  local name root
  name=$(lane_name "$lane")
  root="$RUN_ROOT/net6${lane}_${name}"

  if [[ "$mode" == start ]]; then
    if [[ -f "$root/training_finished" ]]; then
      echo "Net6$lane already complete; skipping"
      return
    fi
    [[ ! -e "$root" ]] || { echo "refusing to overwrite incomplete lane: $root" >&2; exit 1; }
    mkdir -p "$root"
    write_manifest "$lane" "$root/train_files.tsv"
  else
    [[ -d "$root" && -f "$root/train_files.tsv" ]] || { echo "missing lane root: $root" >&2; exit 1; }
    [[ -f "$checkpoint" ]] || { echo "missing recovery checkpoint: $checkpoint" >&2; exit 1; }
    [[ ! -f "$root/training_finished" ]] || { echo "lane already complete: $root" >&2; exit 1; }
  fi

  build_args "$lane" "$root" "$EPOCHS"
  if [[ "$mode" == start ]]; then
    TRAIN_ARGS+=(--resume-from-model="$PARENT")
  else
    echo "WARNING: recovery cannot restore the exact multi-file loader cursor" >&2
    TRAIN_ARGS+=(--resume-from-checkpoint="$checkpoint")
  fi

  {
    echo "experiment=Net6$lane"
    echo "corpus=$name"
    echo "mode=$mode"
    echo "trainer_commit=$(git rev-parse HEAD)"
    echo "parent_sha256=$PARENT_SHA"
    echo "maximum_lr=$MAX_LR"
    echo "lambda=0.74"
    echo "epoch_size=$EPOCH_SIZE"
    echo "epochs=$EPOCHS"
    echo "one_cycle_steps=$ONE_CYCLE_STEPS"
    echo "accepted_presentations=$((EPOCH_SIZE * EPOCHS))"
    echo "seed=$SEED"
    [[ "$lane" != D ]] || echo "accepted_batch_mix=9/7/4"
    date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
  } >> "$root/run_history.txt"

  /usr/bin/time -v python -u train.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$root/train_$mode.log"
  collect_gates "$lane" "$root"
  date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ' >> "$root/run_history.txt"
  touch "$root/training_finished"
  echo "Net6$lane complete: $root/gates"
}

run_smoke() {
  [[ ! -e "$PREFLIGHT_ROOT" ]] || { echo "refusing to overwrite preflight: $PREFLIGHT_ROOT" >&2; exit 1; }
  mkdir -p "$PREFLIGHT_ROOT"
  local lane root checkpoint net roundtrip
  for lane in A D; do
    root="$PREFLIGHT_ROOT/net6$lane"
    build_args "$lane" "$root" 1 1048576 64
    TRAIN_ARGS+=(--resume-from-model="$PARENT")
    python -u train.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$PREFLIGHT_ROOT/net6$lane.log"
    checkpoint=$(find "$root" -path '*/checkpoints/last.ckpt' -type f -printf '%T@ %p\n' |
      sort -nr | head -1 | cut -d' ' -f2-)
    [[ -n "$checkpoint" && -f "$checkpoint" ]] || { echo "smoke checkpoint missing" >&2; exit 1; }
    net="${checkpoint%.ckpt}.nnue"
    [[ -f "$net" ]] || { echo "smoke NNUE missing" >&2; exit 1; }
    roundtrip="$PREFLIGHT_ROOT/net6${lane}_roundtrip.nnue"
    python serialize.py "$net" "$roundtrip" --architecture=shayveri-direct       --features=ShayveriKB16^ --shayveri-factorizer --ft-compression=none --device=cpu
    cmp -- "$net" "$roundtrip"
    [[ -x "$ENGINE" ]] || { echo "missing engine: $ENGINE" >&2; exit 1; }
    printf 'uci\nsetoption name EvalFile value %s\nisready\nposition startpos\ngo nodes 1000\nquit\n' "$net" |
      timeout 30s "$ENGINE" > "$PREFLIGHT_ROOT/net6${lane}_engine.log"
    grep -q '^bestmove ' "$PREFLIGHT_ROOT/net6${lane}_engine.log"
  done
  echo "Net6 preflight passed for single-family and 9/7/4 lanes"
}

activate_environment
verify_inputs
case "${1:-}" in
  preflight) [[ $# -eq 1 ]] || usage; run_smoke ;;
  run) [[ $# -eq 2 ]] || usage; run_lane "$2" start ;;
  recover) [[ $# -eq 3 ]] || usage; run_lane "$2" recover "$3" ;;
  all)
    [[ $# -eq 1 ]] || usage
    for lane in A B C D; do run_lane "$lane" start; done
    ;;
  *) usage ;;
esac
