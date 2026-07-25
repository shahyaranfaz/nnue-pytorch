#!/usr/bin/env python3
import argparse
import copy
import glob
import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_loader.config import DataloaderSkipConfig
from data_loader.dataset import SparseBatchProvider
from model.lightning_module import remap_tablebase_score


RUN_ROOT = "/mnt/c/bullet_data/v2.9/runs/net2"
WARM_MODEL = (
    "/mnt/c/bullet_data/v2.9/artifacts/net2/"
    "net1_e40_s1a_factorized.pt"
)
DIAGNOSTIC_FILE = (
    "/mnt/d/nnue/robotmoon/t80_2024/"
    "test80-2024-03-mar-2tb7p.min-v2.v6.binpack"
)
BATCH_SIZE = 16_384


def find_checkpoint(iteration: int) -> str:
    epoch = iteration - 1
    pattern = os.path.join(
        RUN_ROOT,
        "lightning_logs",
        "*",
        "checkpoints",
        f"epoch={epoch}-step=*.ckpt",
    )
    matches = glob.glob(pattern)
    if not matches:
        raise SystemExit(
            f"No Net2 checkpoint found for iteration {iteration}: {pattern}"
        )
    return max(matches, key=os.path.getmtime)


def weighted_sf_loss(
    scorenet: torch.Tensor,
    score: torch.Tensor,
    outcome: torch.Tensor,
    loss_params,
    lambda_: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    score = remap_tablebase_score(
        score,
        base=loss_params.tb_remap_base,
        scale=loss_params.tb_remap_scale,
        decay=loss_params.tb_remap_decay,
    )

    q = (scorenet - loss_params.in_offset) / loss_params.in_scaling
    qm = (-scorenet - loss_params.in_offset) / loss_params.in_scaling
    prediction = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())

    s = (score - loss_params.out_offset) / loss_params.out_scaling
    sm = (-score - loss_params.out_offset) / loss_params.out_scaling
    evaluation_target = 0.5 * (1.0 + s.sigmoid() - sm.sigmoid())
    target = evaluation_target * lambda_ + outcome * (1.0 - lambda_)

    loss = torch.abs(target - prediction).pow(loss_params.pow_exp)
    if loss_params.qp_asymmetry != 0.0:
        loss = loss * (
            (prediction > target) * loss_params.qp_asymmetry + 1
        )
    weights = 1 + (2.0**loss_params.w1 - 1) * torch.pow(
        (evaluation_target - 0.5).square()
        * evaluation_target
        * (1 - evaluation_target),
        loss_params.w2,
    )
    return loss.reshape(-1) * weights.reshape(-1), weights.reshape(-1)


def add_by_bucket(
    destination: torch.Tensor,
    bucket: torch.Tensor,
    values: torch.Tensor,
) -> None:
    destination.index_add_(0, bucket, values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Net2 with its Net1 warm parent on a fixed, bucketed "
            "diagnostic corpus prefix."
        )
    )
    parser.add_argument("iteration", type=int)
    parser.add_argument("--batches", type=int, default=64)
    args = parser.parse_args()
    if args.iteration <= 0 or args.batches <= 0:
        parser.error("iteration and batches must be positive")

    checkpoint_path = find_checkpoint(args.iteration)
    if not os.path.isfile(DIAGNOSTIC_FILE):
        raise SystemExit(f"Missing diagnostic corpus: {DIAGNOSTIC_FILE}")

    parent = torch.load(WARM_MODEL, weights_only=False, map_location="cpu")
    candidate = copy.deepcopy(parent)
    checkpoint = torch.load(
        checkpoint_path, weights_only=False, map_location="cpu"
    )
    candidate.load_state_dict(checkpoint["state_dict"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parent.model.to(device).eval()
    candidate.model.to(device).eval()
    loss_params = parent.config.loss_params
    lambda_ = 0.74

    provider = SparseBatchProvider(
        "ShayveriKB16^",
        [DIAGNOSTIC_FILE],
        BATCH_SIZE,
        cyclic=True,
        num_workers=1,
        config=DataloaderSkipConfig(),
        device=device,
    )

    counts = torch.zeros(8, dtype=torch.float64, device=device)
    weight_sums = torch.zeros_like(counts)
    parent_numerators = torch.zeros_like(counts)
    candidate_numerators = torch.zeros_like(counts)
    score_delta_squares = torch.zeros_like(counts)

    print(f"checkpoint={checkpoint_path}")
    print(f"diagnostic_file={DIAGNOSTIC_FILE}")
    print(f"positions={args.batches * BATCH_SIZE}")
    print(f"device={device}")

    with torch.inference_mode():
        for batch_index in range(args.batches):
            (
                us,
                them,
                white_indices,
                black_indices,
                outcome,
                score,
                piece_count,
            ) = next(provider)
            bucket = ((piece_count - 1) // 4).clamp(0, 7).long()

            parent_score = parent.model(
                us,
                them,
                white_indices,
                black_indices,
                piece_count,
                True,
                True,
            ).reshape(-1) * parent.model.quantization.nnue2score
            candidate_score = candidate.model(
                us,
                them,
                white_indices,
                black_indices,
                piece_count,
                True,
                True,
            ).reshape(-1) * candidate.model.quantization.nnue2score

            parent_loss, weights = weighted_sf_loss(
                parent_score, score, outcome, loss_params, lambda_
            )
            candidate_loss, _ = weighted_sf_loss(
                candidate_score, score, outcome, loss_params, lambda_
            )

            add_by_bucket(
                counts, bucket, torch.ones_like(bucket, dtype=torch.float64)
            )
            add_by_bucket(weight_sums, bucket, weights.double())
            add_by_bucket(parent_numerators, bucket, parent_loss.double())
            add_by_bucket(candidate_numerators, bucket, candidate_loss.double())
            add_by_bucket(
                score_delta_squares,
                bucket,
                candidate_score.sub(parent_score).square().double(),
            )

            if (batch_index + 1) % 16 == 0:
                print(f"processed_batches={batch_index + 1}/{args.batches}")

    counts = counts.cpu()
    weight_sums = weight_sums.cpu()
    parent_numerators = parent_numerators.cpu()
    candidate_numerators = candidate_numerators.cpu()
    score_delta_squares = score_delta_squares.cpu()

    print(
        "\nbucket pieces count parent_loss candidate_loss delta "
        "score_delta_rms"
    )
    for bucket in range(8):
        count = counts[bucket].item()
        if count == 0:
            print(f"{bucket} empty")
            continue
        parent_loss = (
            parent_numerators[bucket] / weight_sums[bucket]
        ).item()
        candidate_loss = (
            candidate_numerators[bucket] / weight_sums[bucket]
        ).item()
        score_delta_rms = (
            score_delta_squares[bucket] / counts[bucket]
        ).sqrt().item()
        low = bucket * 4 + 1
        high = min(low + 3, 32)
        print(
            f"{bucket:>6} {low:02d}-{high:02d} {int(count):>8} "
            f"{parent_loss:.9f} {candidate_loss:.9f} "
            f"{candidate_loss - parent_loss:+.9f} "
            f"{score_delta_rms:.3f}"
        )

    total_parent = (parent_numerators.sum() / weight_sums.sum()).item()
    total_candidate = (
        candidate_numerators.sum() / weight_sums.sum()
    ).item()
    print(
        f"\noverall parent={total_parent:.9f} "
        f"candidate={total_candidate:.9f} "
        f"delta={total_candidate - total_parent:+.9f}"
    )
    print(
        "Diagnostic only: this fixed prefix belongs to the training "
        "distribution and is not a promotion-quality held-out set."
    )


if __name__ == "__main__":
    main()
