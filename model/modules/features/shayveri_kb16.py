import torch
from torch import nn

from .input_feature import InputFeature


def shayveri_kb16_index(
    perspective: int,
    king_sq: int,
    piece_sq: int,
    piece_type: int,
    piece_colour: int,
) -> int:
    """Match SHAYVERI's Chess768xKB16 feature index exactly.

    Squares use python-chess numbering (a1=0). Piece types are zero based and
    colours use 0 for white and 1 for black, matching SHAYVERI's runtime.
    """
    effective_sq = piece_sq ^ (56 if perspective else 0)
    effective_colour = piece_colour ^ perspective
    base = (effective_colour * 6 + piece_type) * 64 + effective_sq

    effective_king_sq = king_sq ^ (56 if perspective else 0)
    king_file = king_sq & 7
    file_bucket = (0, 1, 2, 3, 3, 2, 1, 0)[king_file]
    rank_bucket = (effective_king_sq >> 3) // 2
    bucket = rank_bucket * 4 + file_bucket

    horizontal_flip = 7 if king_file > 3 else 0
    return bucket * 768 + (base ^ horizontal_flip)


class ShayveriKB16(InputFeature):
    HASH = 0x53484B16
    FEATURE_NAME = "ShayveriKB16^"
    INPUT_FEATURE_NAME = "ShayveriKB16"
    MAX_ACTIVE_FEATURES = 32

    NUM_PLANES = 768
    NUM_BUCKETS = 16
    NUM_INPUTS = NUM_PLANES * NUM_BUCKETS
    NUM_REAL_FEATURES = NUM_INPUTS
    NUM_INPUTS_VIRTUAL = NUM_PLANES

    def __init__(self, num_outputs: int):
        super().__init__()
        self.num_outputs = num_outputs
        self.weight = nn.Parameter(
            torch.empty(self.NUM_INPUTS, num_outputs, dtype=torch.float32)
        )
        self.virtual_weight = nn.Parameter(
            torch.zeros(self.NUM_INPUTS_VIRTUAL, num_outputs, dtype=torch.float32)
        )
        self.reset_parameters()

    def merged_weight(self) -> torch.Tensor:
        return self.weight + self.virtual_weight.repeat(self.NUM_BUCKETS, 1)

    @torch.no_grad()
    def coalesce(self) -> None:
        self.weight.add_(self.virtual_weight.repeat(self.NUM_BUCKETS, 1))
        self.zero_virtual_weights()

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        self.virtual_weight.zero_()

    @torch.no_grad()
    def init_weights(self, num_psqt_buckets: int, nnue2score: float) -> None:
        self.zero_virtual_weights()
        if num_psqt_buckets:
            self.weight[:, -num_psqt_buckets:] = 0.0

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        return self.merged_weight().clone()

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        self.weight.data.copy_(export_weight)
        self.zero_virtual_weights()
