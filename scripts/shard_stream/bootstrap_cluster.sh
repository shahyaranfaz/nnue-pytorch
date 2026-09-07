#!/usr/bin/env bash
set -euo pipefail

readonly USER_NAME=${USER_NAME:-anfazsha}
readonly DOMAIN=${DOMAIN:-utm.utoronto.ca}
readonly REMOTE_REPO=${REMOTE_REPO:-/student/anfazsha/nnue-pytorch}
readonly SMOKE_SHARD=${SMOKE_SHARD:-/student/anfazsha/v2_11/ready/v210_00000.binpack}
readonly LOG_ROOT=${LOG_ROOT:-/tmp/v211-bootstrap-logs}
readonly WORKERS=(
  dh2010pc16
  dh2010pc19
  dh2010pc22
  dh2010pc25
)

mkdir -p "$LOG_ROOT"
pids=()

for worker in "${WORKERS[@]}"; do
  host="$USER_NAME@$worker.$DOMAIN"
  log="$LOG_ROOT/$worker.log"
  echo "Starting $host; log=$log"
  ssh "$host" \
    V211_REPO="$REMOTE_REPO" \
    bash "$REMOTE_REPO/scripts/shard_stream/bootstrap_worker.sh" \
    "$SMOKE_SHARD" >"$log" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!WORKERS[@]}"; do
  if wait "${pids[$index]}"; then
    echo "PASS ${WORKERS[$index]}"
  else
    echo "FAIL ${WORKERS[$index]} -- see $LOG_ROOT/${WORKERS[$index]}.log" >&2
    failed=1
  fi
done

echo
grep -H -E 'torch |cuda |gpu |Iter:|bootstrap complete|^[0-9.]+[A-Z].*/tmp' \
  "$LOG_ROOT"/*.log || true
exit "$failed"
