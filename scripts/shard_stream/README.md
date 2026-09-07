# SHAYVERI v2.11 Net3 shard stream

This directory implements a machine-resident pipeline. The laptop has no
runtime role.

## Roles

- RX 9070 XT: run `source_feeder.sh` continuously during the lab auditions. It
  creates and retains at most one remote 2.75 GB native binpack shard, transfers it through
  `dh2020pc10.utm.utoronto.ca`, and advances only after remote deletion.
- `dh2020pc10`: run `lab_master.sh` continuously. It deletes a shared-NFS shard
  only after every required lane has written a durable checkpoint ACK.
- `dh2010pc16`, `19`, `22`, and `25`: run `bootstrap_lab_pc.sh` once and then
  run the audition worker continuously. The final selected recipe trains on the
  RX 9070 XT against the complete local corpus.

The shared root is `/student/anfazsha/v2_11`. The account has about 4.5 GB free,
so only one shard may be ready at a time. Only source shards, current
checkpoints, compressed logs, ACKs, and deletion markers live on NFS. Venvs,
caches, compilation caches, and active training directories live under
`/tmp/anfazsha-v211` on each worker.

Each lane keeps one resumable checkpoint and exports compact `.nnue` files at
its midpoint and final gate. A durable pending record is written before every
segment. If a worker dies after publishing its checkpoint but before its ACK,
the next invocation verifies the checkpoint's global step and reconstructs the
completion without training the shard twice.

## Encoded experiments

| Host | Lane | Parent | Data | Lambda | LR | Presentations per shard |
|---|---|---|---|---:|---:|---:|
| pc16 | A | v2.5 factorized | v2.10 | 0.74 | 4.375e-4 | 79,986,688 |
| pc19 | B | v2.11 Net1 35B | v2.10 | 0.74 | 2e-5 | 79,986,688 |
| pc22 | C | v2.11 Net1 35B | v2.10 | 0.90 | 2e-5 | 79,986,688 |
| pc25 | D | v2.11 Net1 35B | 50/35/15 | 0.74 | 2e-5 | 39,993,344 |

The audition is one deterministic 20-shard cycle: ten v2.10, seven new
Stockfish, and three T80 shards. Lanes A-C consume the ten v2.10 shards at
79,986,688 presentations each. Lane D consumes every shard at half that size.
All four lanes finish at 799,866,880 accepted presentations. The shared
OneCycle horizon is 48,820 optimizer steps.

## Required parents

Before starting workers, place immutable factorized PyTorch models at:

```text
/student/anfazsha/v2_11/parents/v2_5_factorized.pt
/student/anfazsha/v2_11/parents/net1_35B_factorized.pt
```

Record their SHA-256 values separately before production. `train.py` loads
warm models with `torch.load`; exported `.nnue` files are not valid substitutes.

The worker-enforced hashes are:

```text
v2_5_factorized.pt     c9b37e262cb917650b445e54d3bb6153d4698b84dcb094c657d75145cd242a79
net1_35B_factorized.pt abe2ce14392ea07d0f2eb3279468299fc546bbd1a14f5367877d1c1adfe684e1
```

## Size the auditions

Before starting workers, count one complete non-cyclic pass through the first
v2.10 shard under both filtering recipes:

```bash
source scripts/shard_stream/worker_env.sh
python scripts/shard_stream/count_accepted_positions.py \
  --profile=both \
  /student/anfazsha/v2_11/ready/v210_00000.binpack
```

Use the reported complete-batch counts to set a segment budget that has only a
small, explicit amount of local replay. Do not start the retired 40B-per-lane
worker configuration.

## Startup

Run once on each of the four trainers:

```bash
cd /student/anfazsha/nnue-pytorch
bash scripts/shard_stream/bootstrap_lab_pc.sh
```

Before starting the audition pipeline, run the disposable 2-phase resume and
serialization smoke test on each trainer:

```bash
cd /student/anfazsha/nnue-pytorch
bash scripts/shard_stream/smoke_lab_worker.sh
```

The smoke test writes only under `/tmp/anfazsha-v211/smoke`, does not publish an
ACK, and does not alter the persistent audition state.

Run continuously on pc10:

```bash
cd /student/anfazsha/nnue-pytorch
bash scripts/shard_stream/lab_master.sh
```

Run continuously on each trainer:

```bash
cd /student/anfazsha/nnue-pytorch
bash scripts/shard_stream/lab_worker.sh
```

Run continuously on the 9070 XT:

```bash
cd /mnt/d/nnue/nnue-pytorch
bash scripts/shard_stream/source_feeder.sh
```
