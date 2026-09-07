#!/usr/bin/env bash
set -euo pipefail

# Publish the one pending shard produced by shard_binpacks.py to a remote host.
# This does not acknowledge or delete the local shard. That is a separate step
# after every required trainer has checkpointed beyond the shard.

readonly STATE=${STATE:-/mnt/d/nnue/v210_stream_state.json}
readonly REMOTE=${REMOTE:-anfazsha@dh2020pc10.utm.utoronto.ca}
readonly REMOTE_ROOT=${REMOTE_ROOT:-/student/anfazsha/v2_11_lr_screen}
readonly SSH_KEY=${SSH_KEY:-}
readonly PENDING_INDEX=${PENDING_INDEX:-0}
readonly SHARD_KIND=${SHARD_KIND:-v210}
readonly REQUIRED_LANES=${REQUIRED_LANES:-lr_220,lr_310,lr_4375,lr_620}
readonly STREAM_SEQUENCE=${STREAM_SEQUENCE:-0}

started_agent=0
if ! ssh-add -l >/dev/null 2>&1; then
  if [[ -z "${SSH_AUTH_SOCK:-}" ]]; then
    eval "$(ssh-agent -s)" >/dev/null
    started_agent=1
  fi
  echo "Unlocking the SSH key once for this transfer run."
  if [[ -n "$SSH_KEY" ]]; then
    ssh-add "$SSH_KEY"
  else
    ssh-add
  fi
fi

cleanup() {
  if (( started_agent )); then
    ssh-agent -k >/dev/null
  fi
}
trap cleanup EXIT

[[ -f "$STATE" ]] || {
  echo "Missing sharder state: $STATE" >&2
  exit 1
}

mapfile -t pending < <(
  python3 - "$STATE" "$PENDING_INDEX" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    values = json.load(source).get("pending")
if isinstance(values, dict):
    values = [values]
if not values:
    raise SystemExit("State has no pending shard")
value = values[int(sys.argv[2])]
print(value["path"])
print(value["bytes"])
print(value["sha256"])
PY
)

(( ${#pending[@]} == 3 )) || {
  echo "Could not read pending shard metadata from $STATE" >&2
  exit 1
}

readonly shard=${pending[0]}
readonly expected_bytes=${pending[1]}
readonly expected_sha=${pending[2]}
readonly name=$(basename "$shard")

[[ -f "$shard" ]] || {
  echo "Missing pending shard: $shard" >&2
  exit 1
}
[[ "$name" =~ ^[A-Za-z0-9._-]+\.binpack$ ]] || {
  echo "Unsafe shard filename: $name" >&2
  exit 1
}
[[ "$expected_bytes" =~ ^[0-9]+$ ]] || {
  echo "Invalid expected byte count: $expected_bytes" >&2
  exit 1
}
[[ "$expected_sha" =~ ^[0-9a-f]{64}$ ]] || {
  echo "Invalid expected SHA-256: $expected_sha" >&2
  exit 1
}

actual_bytes=$(stat -c '%s' -- "$shard")
[[ "$actual_bytes" == "$expected_bytes" ]] || {
  echo "Local size mismatch: expected $expected_bytes, got $actual_bytes" >&2
  exit 1
}

echo "Preparing $REMOTE:$REMOTE_ROOT"
ssh "$REMOTE" bash -s -- "$REMOTE_ROOT" <<'REMOTE_PREPARE'
set -euo pipefail
root=$1
mkdir -p "$root/incoming" "$root/ready" "$root/consumed"
REMOTE_PREPARE

remote_partial="$REMOTE_ROOT/incoming/$name.partial"
remote_ready="$REMOTE_ROOT/ready/$name"

echo "Uploading $name ($expected_bytes bytes)"
if ssh "$REMOTE" test -f "$remote_ready"; then
  echo "Remote ready file already exists; verifying without re-upload"
else
  scp -p -- "$shard" "$REMOTE:$remote_partial"
fi

echo "Verifying and publishing $name"
ssh "$REMOTE" bash -s -- \
  "$REMOTE_ROOT" "$name" "$expected_bytes" "$expected_sha" \
  "$SHARD_KIND" "$REQUIRED_LANES" "$STREAM_SEQUENCE" <<'REMOTE_PUBLISH'
set -euo pipefail
root=$1
name=$2
expected_bytes=$3
expected_sha=$4
kind=$5
required_lanes=$6
stream_sequence=$7
partial="$root/incoming/$name.partial"
ready="$root/ready/$name"

if [[ -f "$ready" ]]; then
  actual_bytes=$(stat -c '%s' -- "$ready")
  actual_sha=$(sha256sum -- "$ready" | cut -d' ' -f1)
  [[ "$actual_bytes" == "$expected_bytes" && "$actual_sha" == "$expected_sha" ]] || {
    echo "Existing ready shard does not match: $ready" >&2
    exit 1
  }
  rm -f -- "$partial"
  {
    printf 'kind=%s\n' "$kind"
    printf 'required_lanes=%s\n' "$required_lanes"
    printf 'bytes=%s\n' "$expected_bytes"
    printf 'sha256=%s\n' "$expected_sha"
    printf 'stream_sequence=%s\n' "$stream_sequence"
  } > "$root/ready/$name.meta.partial"
  mv -- "$root/ready/$name.meta.partial" "$root/ready/$name.meta"
  echo "Already published and verified: $ready"
  exit 0
fi

[[ -f "$partial" ]] || {
  echo "Missing uploaded partial: $partial" >&2
  exit 1
}
actual_bytes=$(stat -c '%s' -- "$partial")
[[ "$actual_bytes" == "$expected_bytes" ]] || {
  echo "Remote size mismatch: expected $expected_bytes, got $actual_bytes" >&2
  exit 1
}
actual_sha=$(sha256sum -- "$partial" | cut -d' ' -f1)
[[ "$actual_sha" == "$expected_sha" ]] || {
  echo "Remote SHA-256 mismatch: expected $expected_sha, got $actual_sha" >&2
  exit 1
}

mv -- "$partial" "$ready"
printf '%s  %s\n' "$expected_sha" "$name" > "$root/ready/$name.sha256"
{
  printf 'kind=%s\n' "$kind"
  printf 'required_lanes=%s\n' "$required_lanes"
  printf 'bytes=%s\n' "$expected_bytes"
  printf 'sha256=%s\n' "$expected_sha"
  printf 'stream_sequence=%s\n' "$stream_sequence"
} > "$root/ready/$name.meta.partial"
mv -- "$root/ready/$name.meta.partial" "$root/ready/$name.meta"
echo "Published: $ready"
REMOTE_PUBLISH

echo
echo "Remote shard is ready. Do not run shard_binpacks.py ack until all"
echo "required trainers have consumed it and saved durable checkpoints."
