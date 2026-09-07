# SHAYVERI v2.11 Net3 shard stream

This directory implements a machine-resident pipeline. The laptop has no
runtime role.

## Roles

- RX 9070 XT: run `source_feeder.sh` continuously. It creates and retains at
  most two 2.75 GB native binpack shards, transfers them through
  `dh2020pc10.utm.utoronto.ca`, and advances only after remote deletion.
- `dh2020pc10`: run `lab_master.sh` continuously. It deletes a shared-NFS shard
  only after every required lane has written a durable checkpoint ACK.
- `dh2010pc16`, `19`, `22`, and `25`: run `bootstrap_lab_pc.sh` once and then
  `lab_worker.sh` continuously.

The shared root is `/student/anfazsha/v2_11`. Only source shards, current
checkpoints, compressed logs, ACKs, and deletion markers live on NFS. Venvs,
caches, compilation caches, and active training directories live under
`/tmp/anfazsha-v211` on each worker.

## Encoded experiments

| Host | Lane | Parent | Data | Lambda | LR | Presentations per shard |
|---|---|---|---|---:|---:|---:|
| pc16 | A | v2.5 factorized | v2.10 | 0.74 | 4.375e-4 | 330,579,968 |
| pc19 | B | v2.11 Net1 35B | v2.10 | 0.74 | 2e-5 | 330,579,968 |
| pc22 | C | v2.11 Net1 35B | v2.10 | 0.90 | 2e-5 | 330,579,968 |
| pc25 | D | v2.11 Net1 35B | 50/35/15 | 0.74 | 2e-5 | 165,289,984 |

The feeder emits an interleaved deterministic 20-shard cycle: ten v2.10, seven
new Stockfish, and three T80 shards. Lanes A-C consume all 121 v2.10 shards at
330,579,968 presentations each. Lane D consumes every shard at half that size.
After 242 total shards all lanes have 40,000,176,128 presentations; Lane D has
consumed 121 v2.10, 85 Stockfish, and 36 T80 shards (50.0/35.1/14.9%).

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

## Startup

Run once on each of the four trainers:

```bash
cd /student/anfazsha/nnue-pytorch
bash scripts/shard_stream/bootstrap_lab_pc.sh
```

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
