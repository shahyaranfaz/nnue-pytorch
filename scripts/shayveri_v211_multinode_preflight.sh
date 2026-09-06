#!/usr/bin/env bash
set -euo pipefail

NODE_RANK="${1:?usage: $0 NODE_RANK MASTER_ADDR}"
MASTER_ADDR="${2:?usage: $0 NODE_RANK MASTER_ADDR}"

case "$NODE_RANK" in
  0|1|2|3) ;;
  *) echo "NODE_RANK must be 0, 1, 2, or 3" >&2; exit 2 ;;
esac

readonly MASTER_PORT="${MASTER_PORT:-29611}"
readonly ROOT="${PREFLIGHT_ROOT:-$HOME/v211_multinode_preflight}"
readonly RUN="$ROOT/run"
readonly LOG="$ROOT/rank_${NODE_RANK}.log"
readonly SOURCE_DATA="$PWD/.pgo/small.binpack"
readonly DATA="$ROOT/four_rank.binpack"

[[ -f "$SOURCE_DATA" ]] || { echo "missing preflight data: $SOURCE_DATA" >&2; exit 1; }
mkdir -p "$ROOT"

if [[ "$NODE_RANK" == 0 ]]; then
  tmp="$DATA.tmp.$$"
  cat "$SOURCE_DATA" "$SOURCE_DATA" "$SOURCE_DATA" "$SOURCE_DATA" > "$tmp"
  mv "$tmp" "$DATA"
else
  for _ in $(seq 1 120); do
    [[ -f "$DATA" ]] && break
    sleep 1
  done
fi
[[ -f "$DATA" ]] || { echo "rank-zero preflight corpus was not created" >&2; exit 1; }

python3 -m torch.distributed.run \
  --nnodes=4 \
  --nproc-per-node=1 \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  ddp_launcher.py train.py \
  "$DATA" \
  --architecture=shayveri-direct \
  --features=ShayveriKB16^ \
  --shayveri-factorizer \
  --loss-function=stockfish \
  --lambda=0.74 \
  --start-lambda=0.74 \
  --end-lambda=0.74 \
  --optimizer-name=rangerlite \
  --lr=0.000020 \
  --one-cycle-steps=64 \
  --one-cycle-warmup-pct=0.2 \
  --one-cycle-start-div=25 \
  --one-cycle-final-div=50 \
  --batch-size=16384 \
  --epoch-size=1048576 \
  --max-epochs=1 \
  --num-nodes=4 \
  --validation-size=0 \
  --num-workers=2 \
  --no-wld-filtered \
  --soft-early-fen-skipping=-1 \
  --accelerator=cuda \
  --compile-backend=inductor \
  --network-save-period=1 \
  --save-top-k=1 \
  --swa-start-epoch=-1 \
  --seed=42 \
  --default-root-dir="$RUN" \
  2>&1 | tee "$LOG"
