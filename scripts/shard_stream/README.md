# SHAYVERI v2.11 Net4 compressed LR screen

This directory runs a four-lane learning-rate screen on the UofT lab machines.
The RX 9070 XT Net4 production run is separate and continues uninterrupted.
The laptop only opens terminals and edits repositories; it has no runtime role.

## Question and limitations

The screen asks whether `0.0004375` is a reasonable maximum LR after correcting
lambda from the historical effective value of 1.0 to 0.74. It is a directional
800M-presentation audition, not proof of the optimal LR for the full 40B
OneCycle. Every lane compresses a complete OneCycle into the same 48,820-step
horizon, exactly as Net3A did.

Only maximum LR varies:

| Host | Lane | Maximum LR |
|---|---|---:|
| `dh2010pc16` | `lr_220` | 0.0002200 |
| `dh2010pc19` | `lr_310` | 0.0003100 |
| `dh2010pc22` | `lr_4375` | 0.0004375 |
| `dh2010pc25` | `lr_620` | 0.0006200 |

All lanes use the frozen v2.5 factorized parent, reconstructed v2.10 corpus,
v2.10 filters, lambda 0.74, RangerLite, batch size 16,384, seed 42, ten segments
of 79,986,688 presentations, and 799,866,880 total presentations. Midpoint and
final NNUEs are exported near 400M and 800M.

## Isolated state

This screen deliberately does not reuse completed Net3 state:

- Shared NFS root: `/student/anfazsha/v2_11_lr_screen`
- Worker-local root: `/tmp/anfazsha-v211-lr-screen`
- 9070 sharder state: `/mnt/d/nnue/v211_lr_screen_stream`
- Frozen parent: `/student/anfazsha/v2_11/parents/v2_5_factorized.pt`

Expected parent SHA-256:

```text
c9b37e262cb917650b445e54d3bb6153d4698b84dcb094c657d75145cd242a79
```

The NFS root holds one 2.75GB shard at a time, four current checkpoints,
compressed logs, ACKs, and compact NNUE gates. Active training, venvs, and
caches remain under each machine's `/tmp`.

## Runtime roles

- RX 9070 XT: `source_feeder.sh` creates and transfers ten v2.10 shards. It
  keeps at most one remote shard and one locally prepared successor, and exits
  after all ten are acknowledged.
- `dh2020pc10`: `lab_master.sh` deletes the shared shard only after all four
  lanes publish durable checkpoint ACKs.
- `dh2010pc16`, `19`, `22`, `25`: `lab_worker.sh` trains the assigned LR lane
  until 799,866,880 presentations and then exits.

The durable pending/checkpoint/ACK recovery protocol remains enabled. A shard
is never deleted before all four current checkpoints cover it.

## Start from a clean screen namespace

The defaults are fresh, so do not set `V211_ROOT`, `V211_LOCAL_ROOT`, or
`V211_STREAM` to old Net3 locations. Confirm the parent once from a lab machine:

```bash
sha256sum /student/anfazsha/v2_11/parents/v2_5_factorized.pt
```

Pull the new code on pc10 and every trainer:

```bash
cd /student/anfazsha/nnue-pytorch
git pull
```

The existing lab venv need not be reinstalled. On each trainer, reuse it through
the new isolated local root:

```bash
mkdir -p /tmp/anfazsha-v211-lr-screen
ln -s /tmp/anfazsha-v211/venv /tmp/anfazsha-v211-lr-screen/venv
```

Alternatively, run `bootstrap_lab_pc.sh` on each trainer to build the new local
environment independently.

## Launch

On `dh2020pc10`:

```bash
cd /student/anfazsha/nnue-pytorch
bash scripts/shard_stream/lab_master.sh
```

On each trainer:

```bash
cd /student/anfazsha/nnue-pytorch
bash scripts/shard_stream/lab_worker.sh
```

On the RX 9070 XT:

```bash
cd /mnt/d/nnue/nnue-pytorch
bash scripts/shard_stream/source_feeder.sh
```

Start the master and workers before the feeder. Idle workers wait for the first
shard. The feeder transfers about 27.5GB total, shared by all four workers,
rather than one copy per worker.

## Completion and evaluation

Final nets appear under:

```text
/student/anfazsha/v2_11_lr_screen/nets/lr_220/
/student/anfazsha/v2_11_lr_screen/nets/lr_310/
/student/anfazsha/v2_11_lr_screen/nets/lr_4375/
/student/anfazsha/v2_11_lr_screen/nets/lr_620/
```

Evaluate the four 800M finals in one connected `5+0.05` pool, anchored by v2.5,
historical v2.10 Net1 5B, and current Net4 5B if practical. Compare direct W-D-L
edges, not training loss. The screen informs the ongoing production run; it
does not automatically replace or stop it.
