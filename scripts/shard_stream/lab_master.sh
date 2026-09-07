#!/usr/bin/env bash
set -euo pipefail

readonly ROOT=${V211_ROOT:-/student/anfazsha/v2_11_lr_screen}
readonly POLL_SECONDS=${POLL_SECONDS:-10}

[[ "$(hostname -s)" == dh2020pc10 ]] || {
  echo "lab_master.sh is configured to run on dh2020pc10" >&2
  exit 1
}

mkdir -p "$ROOT/ready" "$ROOT/acks" "$ROOT/deleted" "$ROOT/logs"
echo "Lab master watching $ROOT"

while true; do
  found=0
  for shard in "$ROOT"/ready/*.binpack; do
    [[ -f "$shard" ]] || continue
    found=1
    name=$(basename "$shard")
    meta="$ROOT/ready/$name.meta"
    [[ -f "$meta" ]] || continue
    required=$(sed -n 's/^required_lanes=//p' "$meta")
    [[ -n "$required" ]] || { echo "Missing required_lanes in $meta" >&2; continue; }

    complete=1
    IFS=',' read -r -a lanes <<< "$required"
    for lane in "${lanes[@]}"; do
      [[ -f "$ROOT/acks/$name/$lane.ack" ]] || complete=0
    done
    (( complete )) || continue

    echo "All consumers acknowledged $name; deleting shared payload"
    rm -f -- "$shard" "$ROOT/ready/$name.sha256" "$meta"
    touch "$ROOT/deleted/$name"
  done
  (( found )) || true
  sleep "$POLL_SECONDS"
done
