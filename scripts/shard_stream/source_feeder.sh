#!/usr/bin/env bash
set -euo pipefail

readonly REPO=${V211_REPO:-/mnt/d/nnue/nnue-pytorch}
readonly DATA=${V211_DATA:-/mnt/d/nnue/robotmoon}
readonly STREAM=${V211_STREAM:-/mnt/d/nnue/v211_stream}
readonly REMOTE=${REMOTE:-anfazsha@dh2020pc10.utm.utoronto.ca}
readonly REMOTE_ROOT=${REMOTE_ROOT:-/student/anfazsha/v2_11}
readonly POLL_SECONDS=${POLL_SECONDS:-10}
readonly MAX_IN_FLIGHT=1
readonly SCHEDULE=(v210 stockfish v210 stockfish t80 v210 stockfish v210 v210 stockfish t80 v210 stockfish v210 stockfish v210 t80 v210 stockfish v210)

started_agent=0
if ! ssh-add -l >/dev/null 2>&1; then
  if [[ -z "${SSH_AUTH_SOCK:-}" ]]; then
    eval "$(ssh-agent -s)" >/dev/null
    started_agent=1
  fi
  echo "Unlocking the UofT SSH key once for the persistent feeder."
  ssh-add
fi
cleanup() { if (( started_agent )); then ssh-agent -k >/dev/null; fi; }
trap cleanup EXIT

mkdir -p "$STREAM/shards" "$STREAM/state"
schedule_file="$STREAM/schedule_index"
schedule_index=0
[[ ! -f "$schedule_file" ]] || schedule_index=$(<"$schedule_file")

v210_base=("$DATA"/farseer_relabel/*.binpack "$DATA"/hard_relabel/*.binpack "$DATA"/leela96_relabel/*.binpack "$DATA"/t80_2024/*.binpack)
stockfish_base=("$DATA"/stockfish_new/*.binpack)
t80_base=("$DATA"/t80_2023/*.binpack "$DATA"/t80_2024/*.binpack)
v210=("${v210_base[@]}"); stockfish=(); t80=()
for _ in {1..10}; do stockfish+=("${stockfish_base[@]}"); t80+=("${t80_base[@]}"); done

if [[ ! -f "$STREAM/state/v210.json" && -f /mnt/d/nnue/v210_stream_state.json ]]; then
  cp -- /mnt/d/nnue/v210_stream_state.json "$STREAM/state/v210.json"
  if [[ ! -f "$schedule_file" ]]; then
    schedule_index=1
    printf '1\n' > "$schedule_file"
  fi
fi

echo "9070 feeder active; maximum remote in-flight shards=$MAX_IN_FLIGHT"
while true; do
  remote_count=$(ssh "$REMOTE" "find '$REMOTE_ROOT/ready' -maxdepth 1 -type f -name '*.binpack' | wc -l")
  for kind in v210 stockfish t80; do
    state="$STREAM/state/$kind.json"
    [[ -f "$state" ]] || continue
    mapfile -t pending_names < <(python3 - "$state" <<'PY'
import json, os, sys
with open(sys.argv[1], encoding="utf-8") as f: values=json.load(f).get("pending") or []
if isinstance(values, dict): values=[values]
for value in values: print(os.path.basename(value["path"]))
PY
)
    if (( ${#pending_names[@]} )); then
      name=${pending_names[0]}
      if ssh "$REMOTE" test -f "$REMOTE_ROOT/deleted/$name"; then
        python3 "$REPO/scripts/shard_stream/shard_binpacks.py" ack --state "$state"
        ssh "$REMOTE" rm -f -- "$REMOTE_ROOT/deleted/$name"
      elif ssh "$REMOTE" test -f "$REMOTE_ROOT/ready/$name"; then
        case "$kind" in
          v210) required=lane_a,lane_b,lane_c,lane_d ;;
          *) required=lane_d ;;
        esac
        remote_sequence=$(ssh "$REMOTE" "sed -n 's/^stream_sequence=//p' '$REMOTE_ROOT/ready/$name.meta' 2>/dev/null || true")
        [[ -n "$remote_sequence" ]] || remote_sequence=0
        STATE="$state" PENDING_INDEX=0 SHARD_KIND="$kind" STREAM_SEQUENCE="$remote_sequence" \
          REQUIRED_LANES="$required" REMOTE="$REMOTE" REMOTE_ROOT="$REMOTE_ROOT" \
          bash "$REPO/scripts/shard_stream/push_pending_shard.sh"
      else
        if (( remote_count >= MAX_IN_FLIGHT )); then
          continue
        fi
        case "$kind" in
          v210) required=lane_a,lane_b,lane_c,lane_d ;;
          *) required=lane_d ;;
        esac
        STATE="$state" PENDING_INDEX=0 SHARD_KIND="$kind" STREAM_SEQUENCE="$schedule_index" \
          REQUIRED_LANES="$required" REMOTE="$REMOTE" REMOTE_ROOT="$REMOTE_ROOT" \
          bash "$REPO/scripts/shard_stream/push_pending_shard.sh"
        remote_count=$((remote_count + 1))
      fi
    fi
  done

  remote_count=$(ssh "$REMOTE" "find '$REMOTE_ROOT/ready' -maxdepth 1 -type f -name '*.binpack' | wc -l")
  if (( remote_count >= MAX_IN_FLIGHT )); then sleep "$POLL_SECONDS"; continue; fi

  kind=${SCHEDULE[$((schedule_index % ${#SCHEDULE[@]}))]}
  case "$kind" in
    v210) inputs=("${v210[@]}"); required=lane_a,lane_b,lane_c,lane_d ;;
    stockfish) inputs=("${stockfish[@]}"); required=lane_d ;;
    t80) inputs=("${t80[@]}"); required=lane_d ;;
  esac
  state="$STREAM/state/$kind.json"
  output_dir="$STREAM/shards"
  [[ "$kind" != v210 || ! -d /mnt/d/nnue/v210_stream ]] || output_dir=/mnt/d/nnue/v210_stream
  before=$(find "$output_dir" -maxdepth 1 -type f -name "${kind}_*.binpack" | wc -l)
  python3 "$REPO/scripts/shard_stream/shard_binpacks.py" next \
    --output-dir "$output_dir" --state "$state" --prefix="$kind" \
    --target-size=2750M --max-pending=2 "${inputs[@]}"
  after=$(find "$output_dir" -maxdepth 1 -type f -name "${kind}_*.binpack" | wc -l)
  (( after > before )) || { echo "$kind source exhausted" >&2; exit 1; }

  STATE="$state" PENDING_INDEX=-1 SHARD_KIND="$kind" STREAM_SEQUENCE="$schedule_index" REQUIRED_LANES="$required" \
    REMOTE="$REMOTE" REMOTE_ROOT="$REMOTE_ROOT" \
    bash "$REPO/scripts/shard_stream/push_pending_shard.sh"
  schedule_index=$((schedule_index + 1))
  printf '%s\n' "$schedule_index" > "$schedule_file.partial"
  mv -- "$schedule_file.partial" "$schedule_file"
done
