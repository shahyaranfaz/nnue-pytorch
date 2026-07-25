import struct

import numpy as np
import torch

from ..shayveri_model import ShayveriBucketedModel, ShayveriDirectModel


NNUE_MAGIC = 0x4E4E5545
NNUE_VERSION_KB = 3
NNUE_VERSION_BUCKETED = 4
KING_BUCKETS = 16
OUTPUT_BUCKETS = 8
FLAG_SCRELU = 1
QUANT = 255


def _quantize(
    value: torch.Tensor, scale: int, dtype: torch.dtype, name: str
) -> torch.Tensor:
    rounded = value.detach().cpu().mul(scale).round()
    info = torch.iinfo(dtype)
    if bool(((rounded < info.min) | (rounded > info.max)).any()):
        raise RuntimeError(
            f"{name} is outside {dtype} after scaling by {scale}: "
            f"[{rounded.min().item()}, {rounded.max().item()}]"
        )
    return rounded.to(dtype).contiguous()


class ShayveriNNUEWriter:
    """Serialize SHAYVERI S0 (v3) or material-bucketed S1a (v4)."""

    def __init__(self, model: ShayveriDirectModel):
        if not isinstance(model, ShayveriDirectModel):
            raise TypeError("ShayveriNNUEWriter requires ShayveriDirectModel")

        feature_weights = _quantize(
            model.input.merged_weight(), QUANT, torch.int16, "feature weights"
        )
        feature_bias = _quantize(
            model.input.bias, QUANT, torch.int16, "feature bias"
        )
        output_weights = _quantize(
            model.merged_output_weight().reshape(-1),
            QUANT,
            torch.int16,
            "output weights",
        )
        output_bias = _quantize(
            model.merged_output_bias(),
            QUANT * QUANT,
            torch.int32,
            "output bias",
        )

        feature_weights_bytes = feature_weights.numpy().tobytes()
        feature_bias_bytes = feature_bias.numpy().tobytes()
        output_weights_bytes = output_weights.numpy().tobytes()
        output_bias_bytes = output_bias.numpy().tobytes()

        if model.output_buckets == 1:
            header = struct.pack(
                "<III", NNUE_MAGIC, NNUE_VERSION_KB, KING_BUCKETS
            )
        elif model.output_buckets == OUTPUT_BUCKETS:
            header = struct.pack(
                "<10I",
                NNUE_MAGIC,
                NNUE_VERSION_BUCKETED,
                KING_BUCKETS,
                model.L1,
                OUTPUT_BUCKETS,
                FLAG_SCRELU,
                len(feature_weights_bytes),
                len(feature_bias_bytes),
                len(output_weights_bytes),
                len(output_bias_bytes),
            )
        else:
            raise ValueError(
                f"Unsupported SHAYVERI output bucket count: {model.output_buckets}"
            )

        self.buf = b"".join(
            (
                header,
                feature_weights_bytes,
                feature_bias_bytes,
                output_weights_bytes,
                output_bias_bytes,
            )
        )


class ShayveriNNUEReader:
    """Read SHAYVERI S0/S1a networks, optionally expanding S0 to S1a."""

    def __init__(
        self,
        stream,
        use_factorizer: bool = False,
        output_buckets: int | None = None,
    ):
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated SHAYVERI NNUE header")

        magic, version = struct.unpack("<II", prefix)
        if magic != NNUE_MAGIC:
            raise ValueError(
                f"Invalid SHAYVERI NNUE magic: {magic:#x}"
            )

        if version == NNUE_VERSION_KB:
            tail = stream.read(4)
            if len(tail) != 4:
                raise ValueError("Truncated SHAYVERI v3 header")
            (king_buckets,) = struct.unpack("<I", tail)
            if king_buckets != KING_BUCKETS:
                raise ValueError(
                    f"Invalid SHAYVERI king bucket count: {king_buckets}"
                )
            file_output_buckets = 1
            section_lengths = None
        elif version == NNUE_VERSION_BUCKETED:
            tail = stream.read(32)
            if len(tail) != 32:
                raise ValueError("Truncated SHAYVERI v4 header")
            (
                king_buckets,
                hidden_size,
                file_output_buckets,
                flags,
                feature_weights_bytes,
                feature_bias_bytes,
                output_weights_bytes,
                output_bias_bytes,
            ) = struct.unpack("<8I", tail)
            expected = (KING_BUCKETS, 512, OUTPUT_BUCKETS, FLAG_SCRELU)
            actual = (
                king_buckets,
                hidden_size,
                file_output_buckets,
                flags,
            )
            if actual != expected:
                raise ValueError(
                    f"Invalid SHAYVERI v4 architecture: {actual}, expected {expected}"
                )
            section_lengths = (
                feature_weights_bytes,
                feature_bias_bytes,
                output_weights_bytes,
                output_bias_bytes,
            )
        else:
            raise ValueError(f"Unsupported SHAYVERI NNUE version: {version}")

        target_output_buckets = (
            file_output_buckets if output_buckets is None else output_buckets
        )
        if target_output_buckets not in (1, OUTPUT_BUCKETS):
            raise ValueError(
                f"Unsupported requested output bucket count: {target_output_buckets}"
            )
        if file_output_buckets == OUTPUT_BUCKETS and target_output_buckets == 1:
            raise ValueError("Cannot collapse a bucketed SHAYVERI network to one head")

        model_type = (
            ShayveriBucketedModel
            if target_output_buckets == OUTPUT_BUCKETS
            else ShayveriDirectModel
        )
        self.model = model_type(use_factorizer=use_factorizer)
        feature_count = self.model.input.NUM_INPUTS * self.model.L1
        file_output_count = file_output_buckets * self.model.L1 * 2

        expected_section_lengths = (
            feature_count * 2,
            self.model.L1 * 2,
            file_output_count * 2,
            file_output_buckets * 4,
        )
        if section_lengths is not None and section_lengths != expected_section_lengths:
            raise ValueError(
                "Invalid SHAYVERI v4 section lengths: "
                f"{section_lengths}, expected {expected_section_lengths}"
            )

        feature_weights = self._read_tensor(stream, feature_count, "<i2")
        feature_bias = self._read_tensor(stream, self.model.L1, "<i2")
        output_weights = self._read_tensor(stream, file_output_count, "<i2")
        output_bias = self._read_tensor(stream, file_output_buckets, "<i4")

        if stream.read(1):
            raise ValueError("Trailing bytes in SHAYVERI NNUE file")

        with torch.no_grad():
            self.model.input.load_export_weights(
                torch.from_numpy(feature_weights)
                .reshape(self.model.input.NUM_INPUTS, self.model.L1)
                .float()
                .div_(QUANT)
            )
            self.model.input.bias.copy_(
                torch.from_numpy(feature_bias).float().div_(QUANT)
            )
            loaded_weights = (
                torch.from_numpy(output_weights)
                .reshape(file_output_buckets, self.model.L1 * 2)
                .float()
                .div_(QUANT)
            )
            loaded_bias = (
                torch.from_numpy(output_bias).float().div_(QUANT * QUANT)
            )
            if file_output_buckets == 1 and target_output_buckets == OUTPUT_BUCKETS:
                loaded_weights = loaded_weights.repeat(OUTPUT_BUCKETS, 1)
                loaded_bias = loaded_bias.repeat(OUTPUT_BUCKETS)
            self.model.output.weight.copy_(loaded_weights)
            self.model.output.bias.copy_(loaded_bias)

    @staticmethod
    def _read_tensor(stream, count: int, dtype: str) -> np.ndarray:
        itemsize = np.dtype(dtype).itemsize
        data = stream.read(count * itemsize)
        if len(data) != count * itemsize:
            raise ValueError("Truncated SHAYVERI NNUE tensor data")
        return np.frombuffer(data, dtype=dtype, count=count).copy()
