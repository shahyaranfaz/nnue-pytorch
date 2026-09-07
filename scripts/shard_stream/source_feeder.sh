#!/usr/bin/env bash
set -euo pipefail

readonly REPO=${V211_REPO:-/mnt/d/nnue/nnue-pytorch}
readonly DATA=${V211_DATA:-/mnt/d/nnue/robotmoon}
readonly STREAM=${V211_STREAM:-/mnt/d/nnue/v211_lr_screen_stream}
readonly REMOTE=${REMOTE:-anfazsha@dh2020pc10.utm.utoronto.ca}
readonly REMOTE_ROOT=${REMOTE_ROOT:-/student/anfazsha/v2_11_lr_screen}
readonly POLL_SECONDS=${POLL_SECONDS:-10}
readonly MAX_IN_FLIGHT=1
readonly REQUIRED_LANES=lr_220,lr_310,lr_4375,lr_620
readonly SCHEDULE=(v210 v210 v210 v210 v210 v210 v210 v210 v210 v210)

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
prepared_file="$STREAM/prepared.env"
schedule_index=0
[[ ! -f "$schedule_file" ]] || schedule_index=$(<"$schedule_file")

v210=("$DATA"/farseer_relabel/*.binpack "$DATA"/hard_relabel/*.binpack "$DATA"/leela96_relabel/*.binpack "$DATA"/t80_2024/*.binpack)

echo "9070 feeder active; maximum remote in-flight shards=$MAX_IN_FLIGHT"
while true; do
  prepared_name=""
  [[ ! -f "$prepared_file" ]] || prepared_name=$(sed -n 's/^name=//p' "$prepared_file")
  if ! remote_count=$(ssh "$REMOTE" "find '$REMOTE_ROOT/ready' -maxdepth 1 -type f -name '*.binpack' | wc -l"); then
    echo "Remote inventory failed; retrying in $POLL_SECONDS seconds" >&2
    sleep "$POLL_SECONDS"
    continue
  fi
  for kind in v210; do
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
      if ! remote_status=$(ssh "$REMOTE" "
        if test -f '$REMOTE_ROOT/deleted/$name'; then echo deleted
        elif test -f '$REMOTE_ROOT/ready/$name.meta'; then echo ready-meta
        elif test -f '$REMOTE_ROOT/ready/$name'; then echo ready-no-meta
        else echo absent
        fi
      "); then
        echo "Remote status failed; retrying in $POLL_SECONDS seconds" >&2
        sleep "$POLL_SECONDS"
        continue 2
      fi
      if [[ "$remote_status" == deleted ]]; then
        python3 "$REPO/scripts/shard_stream/shard_binpacks.py" ack --state "$state"
        ssh "$REMOTE" rm -f -- "$REMOTE_ROOT/deleted/$name" || true
      elif [[ "$remote_status" == ready-no-meta ]]; then
        required=$REQUIRED_LANES
        remote_sequence=$(ssh "$REMOTE" "sed -n 's/^stream_sequence=//p' '$REMOTE_ROOT/ready/$name.meta' 2>/dev/null || true")
        [[ -n "$remote_sequence" ]] || remote_sequence=0
        env STATE="$state" PENDING_INDEX=0 SHARD_KIND="$kind" STREAM_SEQUENCE="$remote_sequence" \
          REQUIRED_LANES="$required" REMOTE="$REMOTE" REMOTE_ROOT="$REMOTE_ROOT" \
          bash "$REPO/scripts/shard_stream/push_pending_shard.sh"
      elif [[ "$remote_status" == absent ]]; then
        [[ "$name" != "$prepared_name" ]] || continue
        if (( remote_count >= MAX_IN_FLIGHT )); then
          continue
        fi
        required=$REQUIRED_LANES
        env STATE="$state" PENDING_INDEX=0 SHARD_KIND="$kind" STREAM_SEQUENCE="$schedule_index" \
          REQUIRED_LANES="$required" REMOTE="$REMOTE" REMOTE_ROOT="$REMOTE_ROOT" \
          bash "$REPO/scripts/shard_stream/push_pending_shard.sh"
        remote_count=$((remote_count + 1))
        schedule_index=$((schedule_index + 1))
        printf '%s\n' "$schedule_index" > "$schedule_file.partial"
        mv -- "$schedule_file.partial" "$schedule_file"
      fi
    fi
  done

  if ! remote_count=$(ssh "$REMOTE" "find '$REMOTE_ROOT/ready' -maxdepth 1 -type f -name '*.binpack' | wc -l"); then
    echo "Remote inventory failed; retrying in $POLL_SECONDS seconds" >&2
    sleep "$POLL_SECONDS"
    continue
  fi

  if (( remote_count < MAX_IN_FLIGHT )) && [[ -f "$prepared_file" ]]; then
    prepared_kind=$(sed -n 's/^kind=//p' "$prepared_file")
    prepared_state=$(sed -n 's/^state=//p' "$prepared_file")
    prepared_sequence=$(sed -n 's/^sequence=//p' "$prepared_file")
    prepared_required=$(sed -n 's/^required_lanes=//p' "$prepared_file")
    prepared_name=$(sed -n 's/^name=//p' "$prepared_file")
    echo "Publishing prepared successor $prepared_name"
    env STATE="$prepared_state" PENDING_INDEX=-1 SHARD_KIND="$prepared_kind" \
      STREAM_SEQUENCE="$prepared_sequence" REQUIRED_LANES="$prepared_required" \
      REMOTE="$REMOTE" REMOTE_ROOT="$REMOTE_ROOT" \
      bash "$REPO/scripts/shard_stream/push_pending_shard.sh"
    rm -f -- "$prepared_file"
    remote_count=$((remote_count + 1))
  fi

  if [[ ! -f "$prepared_file" ]] && (( schedule_index >= ${#SCHEDULE[@]} )); then
    pending_count=0
    state="$STREAM/state/v210.json"
    if [[ -f "$state" ]]; then
      pending_count=$(python3 - "$state" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as source:
    pending = json.load(source).get("pending") or []
print(1 if isinstance(pending, dict) else len(pending))
PY
      )
    fi
    if (( remote_count == 0 && pending_count == 0 )); then
      echo "LR-screen feeder complete: all ${#SCHEDULE[@]} shards acknowledged"
      exit 0
    fi
    sleep "$POLL_SECONDS"
    continue
  fi

  if [[ ! -f "$prepared_file" ]]; then
    kind=${SCHEDULE[$schedule_index]}
    inputs=("${v210[@]}")
    required=$REQUIRED_LANES
    state="$STREAM/state/$kind.json"
    output_dir="$STREAM/shards"
    echo "Preparing local successor for stream sequence $schedule_index ($kind)"
    python3 "$REPO/scripts/shard_stream/shard_binpacks.py" next \
      --output-dir "$output_dir" --state "$state" --prefix="$kind" \
      --target-size=2750M --max-pending=2 "${inputs[@]}"
    prepared_name=$(python3 - "$state" <<'PY'
import json, os, sys
with open(sys.argv[1], encoding="utf-8") as source:
    pending = json.load(source).get("pending") or []
if isinstance(pending, dict):
    pending = [pending]
if not pending:
    raise SystemExit("sharder produced no pending successor")
print(os.path.basename(pending[-1]["path"]))
PY
    )
    {
      printf 'name=%s\n' "$prepared_name"
      printf 'kind=%s\n' "$kind"
      printf 'state=%s\n' "$state"
      printf 'sequence=%s\n' "$schedule_index"
      printf 'required_lanes=%s\n' "$required"
    } > "$prepared_file.partial"
    mv -- "$prepared_file.partial" "$prepared_file"
    schedule_index=$((schedule_index + 1))
    printf '%s\n' "$schedule_index" > "$schedule_file.partial"
    mv -- "$schedule_file.partial" "$schedule_file"
    echo "Prepared local successor $prepared_name"
  fi

  if (( remote_count >= MAX_IN_FLIGHT )); then sleep "$POLL_SECONDS"; fi
done
