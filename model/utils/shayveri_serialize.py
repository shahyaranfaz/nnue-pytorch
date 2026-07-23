import struct

import numpy as np
import torch

from ..shayveri_model import ShayveriDirectModel


NNUE_MAGIC = 0x4E4E5545
NNUE_VERSION_KB = 3
KING_BUCKETS = 16
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
    """Serialize a direct model in SHAYVERI's version-3 KB format."""

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
            model.output.weight.reshape(-1),
            QUANT,
            torch.int16,
            "output weights",
        )
        output_bias = _quantize(
            model.output.bias, QUANT * QUANT, torch.int32, "output bias"
        )

        self.buf = b"".join(
            (
                struct.pack("<III", NNUE_MAGIC, NNUE_VERSION_KB, KING_BUCKETS),
                feature_weights.numpy().tobytes(),
                feature_bias.numpy().tobytes(),
                output_weights.numpy().tobytes(),
                output_bias.numpy().tobytes(),
            )
        )


class ShayveriNNUEReader:
    """Read SHAYVERI's version-3 KB format into the direct training model."""

    def __init__(self, stream, use_factorizer: bool = False):
        header = stream.read(12)
        if len(header) != 12:
            raise ValueError("Truncated SHAYVERI NNUE header")

        magic, version, king_buckets = struct.unpack("<III", header)
        expected = (NNUE_MAGIC, NNUE_VERSION_KB, KING_BUCKETS)
        if (magic, version, king_buckets) != expected:
            raise ValueError(
                "Invalid SHAYVERI NNUE header: "
                f"{(magic, version, king_buckets)!r}, expected {expected!r}"
            )

        self.model = ShayveriDirectModel(use_factorizer=use_factorizer)
        feature_count = self.model.input.NUM_INPUTS * self.model.L1
        output_count = self.model.output.weight.numel()

        feature_weights = self._read_tensor(stream, feature_count, "<i2")
        feature_bias = self._read_tensor(stream, self.model.L1, "<i2")
        output_weights = self._read_tensor(stream, output_count, "<i2")
        output_bias = self._read_tensor(stream, 1, "<i4")

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
            self.model.output.weight.copy_(
                torch.from_numpy(output_weights)
                .reshape_as(self.model.output.weight)
                .float()
                .div_(QUANT)
            )
            self.model.output.bias.copy_(
                torch.from_numpy(output_bias).float().div_(QUANT * QUANT)
            )

    @staticmethod
    def _read_tensor(stream, count: int, dtype: str) -> np.ndarray:
        itemsize = np.dtype(dtype).itemsize
        data = stream.read(count * itemsize)
        if len(data) != count * itemsize:
            raise ValueError("Truncated SHAYVERI NNUE tensor data")
        return np.frombuffer(data, dtype=dtype, count=count).copy()
