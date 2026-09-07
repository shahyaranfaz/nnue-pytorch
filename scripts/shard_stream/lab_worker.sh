#!/usr/bin/env bash
set -euo pipefail

readonly REPO=${V211_REPO:-/student/anfazsha/nnue-pytorch}
readonly ROOT=${V211_ROOT:-/student/anfazsha/v2_11}
readonly LOCAL_ROOT=${V211_LOCAL_ROOT:-/tmp/anfazsha-v211}
readonly POLL_SECONDS=${POLL_SECONDS:-10}
readonly BATCH_SIZE=16384
readonly FULL_STEPS=2441440

case "$(hostname -s)" in
  dh2010pc16) readonly LANE=lane_a; readonly LAMBDA=0.74; readonly LR=0.0004375; readonly EPOCH_SIZE=330579968; readonly MAX_SEGMENTS=121; readonly PARENT="$ROOT/parents/v2_5_factorized.pt"; readonly PARENT_SHA=c9b37e262cb917650b445e54d3bb6153d4698b84dcb094c657d75145cd242a79 ;;
  dh2010pc19) readonly LANE=lane_b; readonly LAMBDA=0.74; readonly LR=0.0000200; readonly EPOCH_SIZE=330579968; readonly MAX_SEGMENTS=121; readonly PARENT="$ROOT/parents/net1_35B_factorized.pt"; readonly PARENT_SHA=abe2ce14392ea07d0f2eb3279468299fc546bbd1a14f5367877d1c1adfe684e1 ;;
  dh2010pc22) readonly LANE=lane_c; readonly LAMBDA=0.90; readonly LR=0.0000200; readonly EPOCH_SIZE=330579968; readonly MAX_SEGMENTS=121; readonly PARENT="$ROOT/parents/net1_35B_factorized.pt"; readonly PARENT_SHA=abe2ce14392ea07d0f2eb3279468299fc546bbd1a14f5367877d1c1adfe684e1 ;;
  dh2010pc25) readonly LANE=lane_d; readonly LAMBDA=0.74; readonly LR=0.0000200; readonly EPOCH_SIZE=165289984; readonly MAX_SEGMENTS=242; readonly PARENT="$ROOT/parents/net1_35B_factorized.pt"; readonly PARENT_SHA=abe2ce14392ea07d0f2eb3279468299fc546bbd1a14f5367877d1c1adfe684e1 ;;
  *) echo "No lane is assigned to $(hostname -s)" >&2; exit 1 ;;
esac

[[ -f "$PARENT" ]] || { echo "Missing frozen parent: $PARENT" >&2; exit 1; }
actual_parent_sha=$(sha256sum "$PARENT" | cut -d' ' -f1)
[[ "$actual_parent_sha" == "$PARENT_SHA" ]] || {
  echo "Parent hash mismatch: expected $PARENT_SHA, got $actual_parent_sha" >&2
  exit 1
}
# shellcheck disable=SC1091
source "$REPO/scripts/shard_stream/worker_env.sh"

lane_root="$ROOT/lanes/$LANE"
checkpoint="$lane_root/checkpoints/current.ckpt"
segment_file="$lane_root/segment"
completion_file="$lane_root/last_completion.env"
mkdir -p "$lane_root/checkpoints" "$lane_root/logs" "$ROOT/acks"
segment=0
[[ ! -f "$segment_file" ]] || segment=$(<"$segment_file")
if [[ -f "$completion_file" ]]; then
  completion_segment=$(sed -n 's/^segment=//p' "$completion_file")
  if [[ "$completion_segment" =~ ^[0-9]+$ ]] && (( completion_segment > segment )); then
    segment=$completion_segment
    printf '%s\n' "$segment" > "$segment_file.partial"
    mv -- "$segment_file.partial" "$segment_file"
  fi
fi

echo "$LANE on $(hostname -s), starting at segment $segment/$MAX_SEGMENTS"
while (( segment < MAX_SEGMENTS )); do
  selected=""
  selected_sequence=999999999
  for meta in "$ROOT"/ready/*.binpack.meta; do
    [[ -f "$meta" ]] || continue
    name=$(basename "$meta" .meta)
    shard="$ROOT/ready/$name"
    [[ -f "$shard" ]] || continue
    required=$(sed -n 's/^required_lanes=//p' "$meta")
    [[ ",$required," == *",$LANE,"* ]] || continue
    [[ ! -f "$ROOT/acks/$name/$LANE.ack" ]] || continue
    sequence=$(sed -n 's/^stream_sequence=//p' "$meta")
    [[ "$sequence" =~ ^[0-9]+$ ]] || continue
    if (( sequence < selected_sequence )); then
      selected="$shard"
      selected_sequence=$sequence
    fi
  done
  if [[ -z "$selected" ]]; then sleep "$POLL_SECONDS"; continue; fi

  name=$(basename "$selected")
  if [[ -f "$completion_file" ]] && \
     [[ "$(sed -n 's/^shard=//p' "$completion_file")" == "$name" ]]; then
    echo "Republishing durable ACK for $LANE and $name"
    mkdir -p "$ROOT/acks/$name"
    cp -- "$completion_file" "$ROOT/acks/$name/$LANE.ack.partial"
    mv -- "$ROOT/acks/$name/$LANE.ack.partial" "$ROOT/acks/$name/$LANE.ack"
    continue
  fi
  next_segment=$((segment + 1))
  work="$LOCAL_ROOT/work/$LANE-segment-$next_segment"
  [[ "$work" == /tmp/anfazsha-v211/work/* ]]
  rm -rf -- "$work"
  mkdir -p "$work"
  log="$work/train.log"

  args=(
    "$selected"
    --architecture=shayveri-direct --features=ShayveriKB16^
    --shayveri-factorizer --loss-function=stockfish
    --lambda="$LAMBDA" --start-lambda="$LAMBDA" --end-lambda="$LAMBDA"
    --optimizer-name=rangerlite --lr="$LR" --one-cycle-steps="$FULL_STEPS"
    --batch-size="$BATCH_SIZE" --epoch-size="$EPOCH_SIZE"
    --max-epochs="$next_segment" --validation-size=0 --num-workers=2
    --accelerator=cuda --compile-backend=inductor
    --network-save-period=1 --save-top-k=1 --swa-start-epoch=-1
    --seed=42 --default-root-dir="$work"
  )
  if [[ "$LANE" != lane_a ]]; then
    args+=(--no-wld-filtered --soft-early-fen-skipping=-1)
  fi
  if [[ -f "$checkpoint" ]]; then
    args+=(--resume-from-checkpoint="$checkpoint")
  else
    args+=(--resume-from-model="$PARENT")
  fi

  echo "Training $LANE segment $next_segment from $name"
  python -u train.py "${args[@]}" 2>&1 | tee "$log"
  produced=$(find "$work" -path '*/checkpoints/last.ckpt' -type f -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
  [[ -n "$produced" && -f "$produced" ]] || { echo "No last.ckpt produced" >&2; exit 1; }

  cp -- "$produced" "$lane_root/checkpoints/current.ckpt.partial"
  checkpoint_sha=$(sha256sum "$lane_root/checkpoints/current.ckpt.partial" | cut -d' ' -f1)
  mv -- "$lane_root/checkpoints/current.ckpt.partial" "$checkpoint"
  gzip -c "$log" > "$lane_root/logs/segment_$(printf '%03d' "$next_segment").log.gz.partial"
  mv -- "$lane_root/logs/segment_$(printf '%03d' "$next_segment").log.gz.partial" \
    "$lane_root/logs/segment_$(printf '%03d' "$next_segment").log.gz"

  {
    echo "lane=$LANE"
    echo "host=$(hostname -s)"
    echo "shard=$name"
    echo "segment=$next_segment"
    echo "accepted_presentations=$((next_segment * EPOCH_SIZE))"
    echo "checkpoint_sha256=$checkpoint_sha"
    date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ'
  } > "$completion_file.partial"
  mv -- "$completion_file.partial" "$completion_file"
  printf '%s\n' "$next_segment" > "$segment_file.partial"
  mv -- "$segment_file.partial" "$segment_file"
  mkdir -p "$ROOT/acks/$name"
  cp -- "$completion_file" "$ROOT/acks/$name/$LANE.ack.partial"
  mv -- "$ROOT/acks/$name/$LANE.ack.partial" "$ROOT/acks/$name/$LANE.ack"
  segment=$next_segment
  rm -rf -- "$work"
done

echo "$LANE reached $((MAX_SEGMENTS * EPOCH_SIZE)) accepted presentations"
