import torch
from torch import nn

from .modules.features.shayveri_kb16 import ShayveriKB16
from .modules.feature_transformer.sparse_linear_functions import SparseLinearFunction


class _ShayveriQuantization:
    nnue2score = 400.0


def _fake_quantize(value: torch.Tensor, scale: int) -> torch.Tensor:
    hard = value.mul(scale).round().div(scale).detach()
    return hard + value - value.detach()


def _fake_quantize_activation(value: torch.Tensor) -> torch.Tensor:
    hard = value.mul(255).add(1e-5).floor().div(255).detach()
    return hard + value - value.detach()


def _screlu(value: torch.Tensor, fake_quantize: bool) -> torch.Tensor:
    clipped = torch.clamp(value, 0.0, 1.0)
    squared = clipped.square()
    if not fake_quantize:
        return squared

    # SHAYVERI computes floor(clipped_i16^2 / 255). Recreate that integer
    # grid in the forward pass while retaining the gradient of x^2.
    hard = squared.mul(255).add(1e-5).floor().div(255).detach()
    return hard + squared - squared.detach()


class ShayveriDirectModel(nn.Module):
    """Exact floating-point training counterpart of SHAYVERI KB16x512."""

    L1 = 512
    L2 = 0
    L3 = 0
    num_psqt_buckets = 0
    num_ls_buckets = 1
    feature_name = "ShayveriKB16^"
    input_feature_name = "ShayveriKB16"

    def __init__(self, use_factorizer: bool = False, output_buckets: int = 1):
        super().__init__()
        if output_buckets not in (1, 8):
            raise ValueError("SHAYVERI output_buckets must be 1 or 8")
        self.output_buckets = output_buckets
        self.num_ls_buckets = output_buckets
        self.quantization = _ShayveriQuantization()
        self.input = ShayveriKB16(self.L1)
        self.input.bias = nn.Parameter(torch.zeros(self.L1, dtype=torch.float32))
        self.input.init_weights(0, self.quantization.nnue2score)
        self.input.virtual_weight.requires_grad_(use_factorizer)
        self.output = nn.Linear(self.L1 * 2, output_buckets)
        self.output_factorizer = (
            nn.Linear(self.L1 * 2, 1)
            if use_factorizer and output_buckets > 1
            else None
        )
        if self.output_factorizer is not None:
            with torch.no_grad():
                self.output_factorizer.weight.zero_()
                self.output_factorizer.bias.zero_()
        self.feature_hash = self.input.HASH

    def merged_output_weight(self) -> torch.Tensor:
        if self.output_factorizer is None:
            return self.output.weight
        return self.output.weight + self.output_factorizer.weight

    def merged_output_bias(self) -> torch.Tensor:
        if self.output_factorizer is None:
            return self.output.bias
        return self.output.bias + self.output_factorizer.bias

    @torch.no_grad()
    def clip_weights(self, include_input: bool):
        if include_input:
            virtual = self.input.virtual_weight.repeat(
                self.input.NUM_BUCKETS, 1
            )
            self.input.weight.clamp_(-0.99 - virtual, 0.99 - virtual)

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        self.input.zero_virtual_weights()

    def optimizer_param_groups(self, optimizer_config):
        adamw = optimizer_config.optimizer_name == "adamw"
        ft_decay = (
            optimizer_config.adamw_weight_decay
            if adamw else optimizer_config.ft_weight_decay
        )
        dense_decay = (
            optimizer_config.adamw_weight_decay
            if adamw else optimizer_config.dense_weight_decay
        )
        input_weights = [self.input.weight]
        if self.input.virtual_weight.requires_grad:
            input_weights.append(self.input.virtual_weight)
        groups = [
            {
                "params": input_weights,
                "lr": optimizer_config.lr,
                "weight_decay": ft_decay,
            },
            {
                "params": [self.input.bias],
                "lr": optimizer_config.lr,
                "weight_decay": optimizer_config.adamw_weight_decay if adamw else 0.0,
            },
            {
                "params": [self.output.weight],
                "lr": optimizer_config.lr,
                "weight_decay": dense_decay,
            },
            {
                "params": [self.output.bias],
                "lr": optimizer_config.lr,
                "weight_decay": optimizer_config.adamw_weight_decay if adamw else 0.0,
            },
        ]
        if self.output_factorizer is not None:
            groups.extend(
                (
                    {
                        "params": [self.output_factorizer.weight],
                        "lr": optimizer_config.lr,
                        "weight_decay": (
                            optimizer_config.adamw_weight_decay
                            if adamw
                            else optimizer_config.factorized_weight_decay
                        ),
                    },
                    {
                        "params": [self.output_factorizer.bias],
                        "lr": optimizer_config.lr,
                        "weight_decay": (
                            optimizer_config.adamw_weight_decay if adamw else 0.0
                        ),
                    },
                )
            )
        return groups

    def forward(
        self,
        us: torch.Tensor,
        them: torch.Tensor,
        white_indices: torch.Tensor,
        black_indices: torch.Tensor,
        piece_count: torch.Tensor,
        fake_quantize_acts: bool = True,
        fake_quantize_weights: bool = True,
    ) -> torch.Tensor:
        weight = self.input.merged_weight()
        bias = self.input.bias
        output_weight = self.merged_output_weight()
        output_bias = self.merged_output_bias()

        if fake_quantize_weights:
            weight = _fake_quantize(weight, 255)
            bias = _fake_quantize(bias, 255)
            output_weight = _fake_quantize(output_weight, 255)
            output_bias = _fake_quantize(output_bias, 255 * 255)

        white = SparseLinearFunction.apply(
            white_indices, weight, bias, backend="auto"
        )
        black = SparseLinearFunction.apply(
            black_indices, weight, bias, backend="auto"
        )
        stm = us * white + them * black
        nstm = us * black + them * white

        if fake_quantize_acts:
            stm = _fake_quantize_activation(stm)
            nstm = _fake_quantize_activation(nstm)

        stm = _screlu(stm, fake_quantize_acts)
        nstm = _screlu(nstm, fake_quantize_acts)
        hidden = torch.cat((stm, nstm), dim=1)
        outputs = nn.functional.linear(hidden, output_weight, output_bias)
        if self.output_buckets == 1:
            return outputs

        bucket = ((piece_count - 1) // 4).clamp(0, self.output_buckets - 1)
        return outputs.gather(1, bucket.reshape(-1, 1))


class ShayveriBucketedModel(ShayveriDirectModel):
    """S1a model with eight material heads and an optional shared factorizer."""

    num_ls_buckets = 8

    def __init__(self, use_factorizer: bool = False):
        super().__init__(use_factorizer=use_factorizer, output_buckets=8)
