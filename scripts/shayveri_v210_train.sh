#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
[[ "$MODE" == candidate || "$MODE" == control ]] || {
  echo "usage: $0 candidate|control" >&2
  exit 2
}

readonly ROOT=/mnt/c/bullet_data/v2.10
readonly WARM="$ROOT/net4/artifacts/net1_e40_factorized.pt"
readonly SELF="$ROOT/v210_self_100m.bullet"
readonly EXTERNAL_MANIFEST="$ROOT/net1/train_files.txt"
readonly RUN="$ROOT/net4/${MODE}_500m"
readonly LOG="$ROOT/net4/${MODE}_500m.log"

readonly WARM_SHA=19d586112906f1ce272a9f5de30165fc9c9da8337af661d134219e2d793fbdc0
readonly SELF_SHA=895c55c06312efa6fc0a9992d31015cb16911aa5d03b61483194f3199d775945
readonly EXTERNAL_MANIFEST_SHA=bb3ef74734d8a7e41542e6715e44b718d775608c32ce61308f7cc58a12322c56
readonly EPOCH_SIZE=100007936
readonly EPOCHS=5
readonly ONE_CYCLE_STEPS=30520

[[ "$(sha256sum "$WARM" | cut -d' ' -f1)" == "$WARM_SHA" ]] || {
  echo "warm-model hash mismatch" >&2
  exit 1
}
[[ "$(sha256sum "$EXTERNAL_MANIFEST" | cut -d' ' -f1)" == "$EXTERNAL_MANIFEST_SHA" ]] || {
  echo "external-manifest hash mismatch" >&2
  exit 1
}
if [[ "$MODE" == candidate ]]; then
  [[ "$(sha256sum "$SELF" | cut -d' ' -f1)" == "$SELF_SHA" ]] || {
    echo "self-corpus hash mismatch" >&2
    exit 1
  }
fi

mapfile -t EXTERNAL < "$EXTERNAL_MANIFEST"
[[ "${#EXTERNAL[@]}" -eq 20 ]] || {
  echo "external manifest must contain 20 files" >&2
  exit 1
}
for file in "${EXTERNAL[@]}"; do
  [[ -f "$file" ]] || { echo "missing external file: $file" >&2; exit 1; }
done
[[ ! -e "$RUN" && ! -e "$LOG" ]] || {
  echo "refusing to overwrite run artifacts: $RUN or $LOG" >&2
  exit 1
}

mkdir -p "$RUN"
cp "$EXTERNAL_MANIFEST" "$RUN/external_files.txt"
{
  echo "mode=$MODE"
  echo "trainer_commit=$(git rev-parse HEAD)"
  echo "warm_sha256=$WARM_SHA"
  echo "external_manifest_sha256=$EXTERNAL_MANIFEST_SHA"
  echo "self_sha256=$([[ "$MODE" == candidate ]] && echo "$SELF_SHA" || echo '<none>')"
  echo "epoch_size=$EPOCH_SIZE"
  echo "epochs=$EPOCHS"
  echo "accepted_presentations=$((EPOCH_SIZE * EPOCHS))"
  echo "one_cycle_steps=$ONE_CYCLE_STEPS"
  echo "lambda=0.74"
  echo "wld_filtered=false"
  echo "soft_early_fen_skipping=-1"
  if [[ "$MODE" == candidate ]]; then
    echo "self_batches_per_cycle=9"
    echo "mix_cycle_batches=20"
  fi
  date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
} > "$RUN/run_config.txt"

args=(
  "${EXTERNAL[@]}"
  --architecture=shayveri-direct
  --features=ShayveriKB16^
  --shayveri-factorizer
  --loss-function=stockfish
  --lambda=0.74
  --start-lambda=0.74
  --end-lambda=0.74
  --optimizer-name=rangerlite
  --lr=0.0004375
  --one-cycle-steps="$ONE_CYCLE_STEPS"
  --batch-size=16384
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
  --default-root-dir="$RUN"
  --resume-from-model="$WARM"
)
if [[ "$MODE" == candidate ]]; then
  args+=(
    --secondary-datasets "$SELF"
    --secondary-batches-per-cycle=9
    --mix-cycle-batches=20
  )
fi

python train.py "${args[@]}" 2>&1 | tee "$LOG"
date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ' >> "$RUN/run_config.txt"
