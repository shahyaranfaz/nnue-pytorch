#!/usr/bin/env bash
set -euo pipefail

SCREEN="${1:-}"
case "$SCREEN" in
  A|a)
    readonly SCREEN=A
    readonly MAX_LR=0.000010
    ;;
  B|b)
    readonly SCREEN=B
    readonly MAX_LR=0.000020
    ;;
  C|c)
    readonly SCREEN=C
    readonly MAX_LR=0.000040
    ;;
  *)
    echo "usage: $0 A|B|C" >&2
    exit 2
    ;;
esac

readonly DATA_ROOT=/mnt/d/nnue/robotmoon
readonly ROOT=/mnt/c/bullet_data/v2.11/net1
readonly WARM="$ROOT/artifacts/v210_parent_factorized.pt"
readonly WARM_SHA=b1bbfa9463aa1970cbdc478dffc2cdf80875102b5bc8fafda8dc8d65efe8b610
readonly RUN="$ROOT/lr_screen/$SCREEN"
readonly LOG="$ROOT/lr_screen/${SCREEN}.log"
readonly EPOCH_SIZE=500006912
readonly EPOCHS=4
readonly ONE_CYCLE_STEPS=122072
readonly BATCH_SIZE=16384
readonly SEED=42

readonly -a T80_2023=(
  "$DATA_ROOT/t80_2023/test80-2023-01-jan-16tb7p.v6-sk20.min.binpack"
  "$DATA_ROOT/t80_2023/test80-2023-02-feb-16tb7p.v6-dd.min.binpack"
  "$DATA_ROOT/t80_2023/test80-2023-03-mar-2tb7p.v6-sk16.min.binpack"
  "$DATA_ROOT/t80_2023/test80-2023-04-apr-2tb7p.v6-sk16.min.binpack"
  "$DATA_ROOT/t80_2023/test80-2023-05-may-2tb7p.v6.min.binpack"
  "$DATA_ROOT/t80_2023/test80-2023-06-jun-2tb7p.min-v2.v6.binpack"
  "$DATA_ROOT/t80_2023/test80-2023-07-jul-2tb7p.min-v2.v6.binpack"
  "$DATA_ROOT/t80_2023/test80-2023-08-aug-2tb7p.v6.min.binpack"
  "$DATA_ROOT/t80_2023/test80-2023-09-sep-2tb7p.min-v2.v6.binpack"
  "$DATA_ROOT/t80_2023/test80-2023-10-oct-2tb7p.min-v2.v6.binpack"
  "$DATA_ROOT/t80_2023/test80-2023-11-nov-2tb7p.min-v2.v6.binpack"
  "$DATA_ROOT/t80_2023/test80-2023-12-dec-2tb7p.min-v2.v6.binpack"
)

readonly -a STOCKFISH_NEW=(
  "$DATA_ROOT/stockfish_new/data_pv-2_diff-100_nodes-5000.binpack"
  "$DATA_ROOT/stockfish_new/fishpack32.binpack"
  "$DATA_ROOT/stockfish_new/test80-2022-08-aug-16tb7p.v6-dd.min.binpack"
  "$DATA_ROOT/stockfish_new/training_data_pylon.binpack"
  "$DATA_ROOT/stockfish_new/wrongNNUE_02_d9.binpack"
)

readonly -a T80_2024=(
  "$DATA_ROOT/t80_2024/test80-2024-01-jan-2tb7p.min-v2.v6.binpack"
  "$DATA_ROOT/t80_2024/test80-2024-02-feb-2tb7p.min-v2.v6.binpack"
  "$DATA_ROOT/t80_2024/test80-2024-03-mar-2tb7p.min-v2.v6.binpack"
  "$DATA_ROOT/t80_2024/test80-2024-04-apr-2tb7p.min-v2.v6.binpack"
  "$DATA_ROOT/t80_2024/test80-2024-05-may-2tb7p.min-v2.v6.binpack"
  "$DATA_ROOT/t80_2024/test80-2024-06-jun-2tb7p.min-v2.v6.binpack"
)

[[ "$(git branch --show-current)" == master ]] || {
  echo "trainer must be on master" >&2
  exit 1
}
git diff --quiet && git diff --cached --quiet || {
  echo "trainer worktree must be clean" >&2
  exit 1
}

[[ "$(sha256sum "$WARM" | cut -d' ' -f1)" == "$WARM_SHA" ]] || {
  echo "warm-model hash mismatch" >&2
  exit 1
}

[[ "${#T80_2023[@]}" -eq 12 ]]
[[ "${#STOCKFISH_NEW[@]}" -eq 5 ]]
[[ "${#T80_2024[@]}" -eq 6 ]]

for file in "${T80_2023[@]}" "${STOCKFISH_NEW[@]}" "${T80_2024[@]}"; do
  [[ -f "$file" ]] || {
    echo "missing training file: $file" >&2
    exit 1
  }
done

[[ ! -e "$RUN" && ! -e "$LOG" ]] || {
  echo "refusing to overwrite screen artifacts: $RUN or $LOG" >&2
  exit 1
}

mkdir -p "$RUN"

{
  echo "screen=$SCREEN"
  echo "trainer_commit=$(git rev-parse HEAD)"
  echo "warm_sha256=$WARM_SHA"
  echo "maximum_lr=$MAX_LR"
  echo "one_cycle_steps=$ONE_CYCLE_STEPS"
  echo "one_cycle_warmup_pct=0.2"
  echo "one_cycle_start_div=25"
  echo "one_cycle_final_div=50"
  echo "optimizer=rangerlite"
  echo "lambda=0.74"
  echo "start_lambda=0.74"
  echo "end_lambda=0.74"
  echo "batch_size=$BATCH_SIZE"
  echo "epoch_size=$EPOCH_SIZE"
  echo "epochs=$EPOCHS"
  echo "accepted_presentations=$((EPOCH_SIZE * EPOCHS))"
  echo "seed=$SEED"
  echo "t80_2023_batches_per_cycle=9"
  echo "stockfish_new_batches_per_cycle=7"
  echo "t80_2024_batches_per_cycle=4"
  echo "mix_cycle_batches=20"
  echo "wld_filtered=false"
  echo "soft_early_fen_skipping=-1"
  date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
} > "$RUN/run_config.txt"

{
  for file in "${T80_2023[@]}"; do
    printf 't80_2023\t%s\t%s\n' "$(stat -c '%s' "$file")" "$file"
  done
  for file in "${STOCKFISH_NEW[@]}"; do
    printf 'stockfish_new\t%s\t%s\n' "$(stat -c '%s' "$file")" "$file"
  done
  for file in "${T80_2024[@]}"; do
    printf 't80_2024\t%s\t%s\n' "$(stat -c '%s' "$file")" "$file"
  done
} > "$RUN/train_files.tsv"

args=(
  "${T80_2023[@]}"
  --architecture=shayveri-direct
  --features=ShayveriKB16^
  --shayveri-factorizer
  --loss-function=stockfish
  --lambda=0.74
  --start-lambda=0.74
  --end-lambda=0.74
  --optimizer-name=rangerlite
  --lr="$MAX_LR"
  --one-cycle-steps="$ONE_CYCLE_STEPS"
  --one-cycle-warmup-pct=0.2
  --one-cycle-start-div=25
  --one-cycle-final-div=50
  --batch-size="$BATCH_SIZE"
  --epoch-size="$EPOCH_SIZE"
  --max-epochs="$EPOCHS"
  --validation-size=0
  --num-workers=2
  --no-wld-filtered
  --soft-early-fen-skipping=-1
  --accelerator=cuda
  --compile-backend=inductor
  --network-save-period=1
  --save-top-k=-1
  --swa-start-epoch=-1
  --seed="$SEED"
  --secondary-batches-per-cycle=7
  --tertiary-batches-per-cycle=4
  --mix-cycle-batches=20
  --default-root-dir="$RUN"
  --resume-from-model="$WARM"
)

for file in "${STOCKFISH_NEW[@]}"; do
  args+=(--secondary-datasets "$file")
done

for file in "${T80_2024[@]}"; do
  args+=(--tertiary-datasets "$file")
done

python train.py "${args[@]}" 2>&1 | tee "$LOG"

date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ' >> "$RUN/run_config.txt"
touch "$RUN/training_finished"
