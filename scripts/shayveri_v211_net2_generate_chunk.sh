#!/usr/bin/env bash
set -euo pipefail

WAVE="${1:?usage: $0 WAVE RANK}"
RANK="${2:?usage: $0 WAVE RANK}"

case "$WAVE" in 0|1) ;; *) echo "WAVE must be 0 or 1" >&2; exit 2 ;; esac
case "$RANK" in 0|1|2|3) ;; *) echo "RANK must be 0, 1, 2, or 3" >&2; exit 2 ;; esac

readonly REPO="${REPO:-$HOME/SHAYVERI}"
readonly ROOT="${ROOT:-$HOME/v211_net2}"
readonly PARENT="${PARENT:-$REPO/SHAYVERI2_10_4.nnue}"
readonly BOOK="${START_FILE:-$HOME/chess_arena/books/final_search_mix_shuf.epd}"

if [[ "$WAVE" == 1 && ! -f "$ROOT/state/wave_0_consumed" ]]; then
  echo "Wave 0 must be consumed before Wave 1 starts." >&2
  exit 1
fi

[[ -x "$REPO/SHAYVERI" ]] || { echo "missing engine" >&2; exit 1; }
[[ -f "$PARENT" ]] || { echo "missing parent" >&2; exit 1; }
[[ -f "$BOOK" ]] || { echo "missing start file" >&2; exit 1; }
[[ "$(sha256sum "$PARENT" | cut -d' ' -f1)" == \
  6498611441a773f586f66ff9d39f65ef3b5ed9287c5e4f468a5e323325aede48 ]]

mkdir -p "$ROOT/state"
readonly CLAIM="$ROOT/state/wave_${WAVE}_rank_${RANK}.claimed"
if ! ( set -o noclobber; hostname > "$CLAIM" ) 2>/dev/null; then
  echo "chunk already claimed: $CLAIM" >&2
  exit 1
fi

WORKER_ID="v211n2_w${WAVE}_r${RANK}" \
SHARD_PREFIX="v211n2_w${WAVE}_r${RANK}" \
RUN_ROOT="$ROOT/raw/wave_$WAVE" \
ENGINE_DIR="$REPO" \
ENGINE=./SHAYVERI \
EVAL_FILE="$PARENT" \
THREADS=23 \
SHARD_POSITIONS=25000000 \
NODES=2500 \
SHARDS=1 \
START_SHARD=0 \
SEED_BASE="$((910000 + WAVE * 100 + RANK))" \
START_FILE="$BOOK" \
START_FILE_PROB=0.25 \
INCLUDE_DUPLICATES=false \
ENABLE_ADJUDICATION=false \
MAX_SAMPLES_PER_GAME=16 \
MAX_UNTRANSFERRED_SHARDS=4 \
bash "$REPO/scripts/datagen/worker_datagen.sh"

touch "$ROOT/state/wave_${WAVE}_rank_${RANK}.ready"
