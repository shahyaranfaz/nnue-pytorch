import struct

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
