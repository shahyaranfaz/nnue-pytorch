#!/usr/bin/env python3
"""Count accepted full batches in binpacks without cycling the input."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read binpacks once with the native training loader and report how "
            "many accepted positions form complete training batches."
        )
    )
    parser.add_argument("binpacks", nargs="+", type=Path)
    parser.add_argument(
        "--profile",
        choices=("lane-a", "modern", "both"),
        default="both",
        help=(
            "lane-a preserves v2.10 Net1 filtering; modern disables WLD and "
            "soft early-FEN filtering like v2.11"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def count_profile(paths: list[Path], profile: str, batch_size: int, workers: int) -> dict:
    import data_loader

    config = data_loader.DataloaderSkipConfig()
    if profile == "modern":
        config.wld_filtered = False
        config.soft_early_fen_skipping = -1

    dataset = data_loader.SparseBatchDataset(
        "ShayveriKB16",
        [str(path) for path in paths],
        batch_size,
        cyclic=False,
        num_workers=workers,
        config=config,
    )
    batches = sum(1 for _ in dataset)
    return {
        "profile": profile,
        "complete_batches": batches,
        "accepted_positions": batches * batch_size,
        "batch_size": batch_size,
    }


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.num_workers <= 0:
        raise SystemExit("--num-workers must be positive")
    missing = [str(path) for path in args.binpacks if not path.is_file()]
    if missing:
        raise SystemExit(f"missing binpack: {missing[0]}")

    paths = [path.resolve() for path in args.binpacks]
    profiles = ("lane-a", "modern") if args.profile == "both" else (args.profile,)
    result = {
        "files": [str(path) for path in paths],
        "total_bytes": sum(path.stat().st_size for path in paths),
        "counts": [
            count_profile(paths, profile, args.batch_size, args.num_workers)
            for profile in profiles
        ],
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
