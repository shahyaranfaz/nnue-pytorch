#!/usr/bin/env python3
import argparse
import glob
import math
import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RUN_ROOT = "/mnt/c/bullet_data/v2.9/runs/net2"
WARM_MODEL = (
    "/mnt/c/bullet_data/v2.9/artifacts/net2/"
    "net1_e40_s1a_factorized.pt"
)


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


def tensor_stats(name: str, current: torch.Tensor, initial: torch.Tensor) -> None:
    current = current.detach().float().cpu()
    initial = initial.detach().float().cpu()
    if current.shape != initial.shape:
        raise SystemExit(
            f"{name} shape mismatch: {tuple(current.shape)} != "
            f"{tuple(initial.shape)}"
        )
    if not bool(torch.isfinite(current).all()):
        raise SystemExit(f"{name} contains non-finite values")

    delta = current - initial
    rms = float(delta.square().mean().sqrt())
    maximum = float(delta.abs().max())
    changed = int(torch.count_nonzero(delta))
    print(
        f"{name}: shape={tuple(current.shape)} changed={changed}/{delta.numel()} "
        f"delta_rms={rms:.9g} delta_max={maximum:.9g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a Net2 checkpoint against its exact warm start."
    )
    parser.add_argument("iteration", type=int)
    args = parser.parse_args()
    if args.iteration <= 0:
        parser.error("iteration must be positive")

    checkpoint_path = find_checkpoint(args.iteration)
    checkpoint = torch.load(
        checkpoint_path, weights_only=False, map_location="cpu"
    )
    warm_nnue = torch.load(
        WARM_MODEL, weights_only=False, map_location="cpu"
    )
    state = checkpoint["state_dict"]
    warm = warm_nnue.state_dict()

    print(f"checkpoint={checkpoint_path}")
    print(f"epoch={checkpoint.get('epoch')}")
    print(f"global_step={checkpoint.get('global_step')}")

    output_weight = state["model.output.weight"]
    output_bias = state["model.output.bias"]
    warm_output_weight = warm["model.output.weight"]
    warm_output_bias = warm["model.output.bias"]
    if tuple(output_weight.shape) != (8, 1024):
        raise SystemExit(
            f"Unexpected output weight shape: {tuple(output_weight.shape)}"
        )
    if tuple(output_bias.shape) != (8,):
        raise SystemExit(
            f"Unexpected output bias shape: {tuple(output_bias.shape)}"
        )

    print("\nOutput-head movement from the common warm head:")
    for bucket in range(8):
        tensor_stats(
            f"bucket[{bucket}].weight",
            output_weight[bucket],
            warm_output_weight[bucket],
        )
        tensor_stats(
            f"bucket[{bucket}].bias",
            output_bias[bucket],
            warm_output_bias[bucket],
        )

    print("\nShared-transformer movement:")
    for name in (
        "model.input.weight",
        "model.input.bias",
        "model.input.virtual_weight",
    ):
        tensor_stats(name, state[name], warm[name])

    print("\nHead specialization:")
    centered = output_weight.float() - output_weight.float().mean(dim=0)
    print(
        "between_bucket_weight_rms="
        f"{float(centered.square().mean().sqrt()):.9g}"
    )
    print(
        "between_bucket_bias_std="
        f"{float(output_bias.float().std(unbiased=False)):.9g}"
    )

    scheduler_states = checkpoint.get("lr_schedulers", [])
    optimizer_states = checkpoint.get("optimizer_states", [])
    print("\nLearning rate:")
    if scheduler_states:
        print(f"scheduler_last_lr={scheduler_states[0].get('_last_lr')}")
    if optimizer_states:
        rates = [
            group.get("lr")
            for group in optimizer_states[0].get("param_groups", [])
        ]
        print(f"optimizer_group_lr={rates}")

    values = [
        float(output_weight[bucket].sub(warm_output_weight[bucket]).square().mean())
        for bucket in range(8)
    ]
    if any(not math.isfinite(value) or value == 0.0 for value in values):
        raise SystemExit("At least one output head did not receive finite updates")

    print("\nNet2 checkpoint structure and update activity OK")


if __name__ == "__main__":
    main()
