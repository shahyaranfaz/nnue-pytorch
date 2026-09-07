#!/usr/bin/env python3
"""Split Stockfish binpacks at native BINP chunk boundaries.

The output files remain directly readable by nnue-pytorch. A durable state file
allows an interrupted run to resume without rereading or duplicating input.
Only one completed shard is retained at a time: acknowledge it with --ack,
then run the producer again for the next shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path


HEADER_SIZE = 8
MAGIC = b"BINP"


def parse_size(value: str) -> int:
    units = {"": 1, "K": 1000, "M": 1000**2, "G": 1000**3,
             "KI": 1024, "MI": 1024**2, "GI": 1024**3}
    text = value.strip().upper()
    for suffix in sorted(units, key=len, reverse=True):
        if suffix and text.endswith(suffix):
            return int(float(text[:-len(suffix)]) * units[suffix])
    return int(text)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def initial_state(inputs: list[Path]) -> dict:
    return {
        "version": 1,
        "inputs": [str(path.resolve()) for path in inputs],
        "input_index": 0,
        "input_offset": 0,
        "next_shard": 0,
        "pending": [],
    }


def load_state(path: Path, inputs: list[Path]) -> dict:
    expected = [str(item.resolve()) for item in inputs]
    if not path.exists():
        return initial_state(inputs)
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("version") != 1 or state.get("inputs") != expected:
        raise ValueError("state file does not match this ordered input list")
    if isinstance(state.get("pending"), dict):
        state["pending"] = [state["pending"]]
    elif state.get("pending") is None:
        state["pending"] = []
    return state


def read_chunk(source, source_name: Path) -> bytes | None:
    header = source.read(HEADER_SIZE)
    if not header:
        return None
    if len(header) != HEADER_SIZE or header[:4] != MAGIC:
        raise ValueError(f"invalid BINP header in {source_name} at {source.tell() - len(header)}")
    payload_size = struct.unpack("<I", header[4:])[0]
    payload = source.read(payload_size)
    if len(payload) != payload_size:
        raise ValueError(f"truncated BINP chunk in {source_name} at {source.tell() - len(payload) - HEADER_SIZE}")
    return header + payload


def produce(args: argparse.Namespace) -> int:
    inputs = [Path(item) for item in args.inputs]
    for item in inputs:
        if not item.is_file():
            raise FileNotFoundError(item)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(args.state, inputs)
    pending = state["pending"]
    if len(pending) >= args.max_pending:
        print(f"Pending shard limit reached ({args.max_pending}).")
        return 2
    if state["input_index"] >= len(inputs):
        print("Corpus complete.")
        return 0

    shard_number = state["next_shard"]
    final = args.output_dir / f"{args.prefix}_{shard_number:05d}.binpack"
    partial = final.with_suffix(final.suffix + ".partial")
    if final.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {final} or {partial}")

    digest = hashlib.sha256()
    written = 0
    chunks = 0
    shard_full = False
    start_index = state["input_index"]
    start_offset = state["input_offset"]

    with partial.open("xb") as output:
        while state["input_index"] < len(inputs):
            source_path = inputs[state["input_index"]]
            with source_path.open("rb") as source:
                source.seek(state["input_offset"])
                while True:
                    chunk_start = source.tell()
                    chunk = read_chunk(source, source_path)
                    if chunk is None:
                        state["input_index"] += 1
                        state["input_offset"] = 0
                        break
                    if written and written + len(chunk) > args.target_bytes:
                        source.seek(chunk_start)
                        shard_full = True
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                    chunks += 1
                    state["input_offset"] = source.tell()
                    if written >= args.target_bytes:
                        break
            if shard_full or written >= args.target_bytes:
                break

        output.flush()
        os.fsync(output.fileno())

    if not written:
        partial.unlink()
        print("Corpus complete.")
        return 0

    os.replace(partial, final)
    state["next_shard"] += 1
    record = {
        "path": str(final.resolve()),
        "bytes": written,
        "chunks": chunks,
        "sha256": digest.hexdigest(),
        "start_input_index": start_index,
        "start_input_offset": start_offset,
        "end_input_index": state["input_index"],
        "end_input_offset": state["input_offset"],
    }
    state["pending"].append(record)
    atomic_json(args.state, state)
    print(json.dumps(record, indent=2))
    return 0


def acknowledge(args: argparse.Namespace) -> int:
    state = json.loads(args.state.read_text(encoding="utf-8"))
    pending = state.get("pending") or []
    if isinstance(pending, dict):
        pending = [pending]
    if not pending:
        print("No pending shard.")
        return 0
    record = pending[0]
    shard = Path(record["path"])
    if not args.keep and shard.exists():
        shard.unlink()
    state["pending"] = pending[1:]
    atomic_json(args.state, state)
    print(f"Acknowledged shard {shard.name}")
    return 0


def retry(args: argparse.Namespace) -> int:
    state = json.loads(args.state.read_text(encoding="utf-8"))
    pending = state.get("pending") or []
    if isinstance(pending, dict):
        pending = [pending]
    if not pending:
        print("No pending shard.")
        return 0
    if len(pending) != 1:
        raise ValueError("retry is only safe when exactly one shard is pending")
    record = pending[0]
    shard = Path(record["path"])
    shard.unlink(missing_ok=True)
    shard.with_suffix(shard.suffix + ".partial").unlink(missing_ok=True)
    state["input_index"] = record["start_input_index"]
    state["input_offset"] = record["start_input_offset"]
    state["next_shard"] -= 1
    state["pending"] = []
    atomic_json(args.state, state)
    print(f"Rewound shard {shard.name}; run next to regenerate it")
    return 0


def status(args: argparse.Namespace) -> int:
    if not args.state.exists():
        print("No state file.")
        return 0
    print(args.state.read_text(encoding="utf-8"), end="")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("next", help="produce one resumable shard")
    create.add_argument("inputs", nargs="+", help="ordered source .binpack files")
    create.add_argument("--output-dir", type=Path, required=True)
    create.add_argument("--state", type=Path, required=True)
    create.add_argument("--prefix", default="v210")
    create.add_argument("--target-size", default="2750M")
    create.add_argument("--max-pending", type=int, default=1)
    create.set_defaults(func=produce)

    ack = subparsers.add_parser("ack", help="delete and advance past the pending shard")
    ack.add_argument("--state", type=Path, required=True)
    ack.add_argument("--keep", action="store_true")
    ack.set_defaults(func=acknowledge)

    rewind = subparsers.add_parser("retry", help="rewind and regenerate the pending shard")
    rewind.add_argument("--state", type=Path, required=True)
    rewind.set_defaults(func=retry)

    show = subparsers.add_parser("status")
    show.add_argument("--state", type=Path, required=True)
    show.set_defaults(func=status)

    args = parser.parse_args()
    if hasattr(args, "target_size"):
        args.target_bytes = parse_size(args.target_size)
        if args.target_bytes <= HEADER_SIZE:
            parser.error("--target-size is too small")
        if args.max_pending <= 0:
            parser.error("--max-pending must be positive")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
