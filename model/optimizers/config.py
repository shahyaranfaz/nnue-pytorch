from dataclasses import dataclass
from typing import Literal

from .rangerlite_wrapper import RangerLiteConfig, RangerLiteWrapper
from .schedulefree_wrapper import ScheduleFreeConfig, ScheduleFreeWrapper
from .adamw_wrapper import AdamWWrapper


@dataclass(kw_only=True)
class OptimizerConfig(RangerLiteConfig, ScheduleFreeConfig):
    optimizer_name: Literal["schedulefree", "ranger21", "rangerlite", "adamw"] = "rangerlite"
    """Which optimizer to use. Note that ranger21 is a specific configuration of rangerlite emulating ranger21 behaviour with legacy_mode=True."""

    ft_weight_decay: float = 0.0
    """Weight decay to apply to the feature transformer parameters."""

    dense_weight_decay: float = 0.0
    """Weight decay to apply to the dense layer parameters."""

    factorized_weight_decay: float = 0.0
    """Weight decay to apply to the factorized dense layer parameters."""

    lr: float = 8.75e-4
    """Initial learning rate."""

    adamw_final_lr: float = 1.0e-5
    """Final learning rate for AdamW cosine decay."""

    adamw_epochs: int = 20
    """Number of epochs in the AdamW cosine schedule."""

    adamw_weight_decay: float = 0.01
    """Decoupled weight decay used by the Bullet-compatible AdamW."""

    def get_optimizer_wrapper(self):
        optimizer_name = self.optimizer_name.lower().strip()
        if optimizer_name == "schedulefree":
            wrapper = ScheduleFreeWrapper(self)
        elif optimizer_name == "ranger21":
            wrapper = RangerLiteWrapper(self, legacy_mode=True)
        elif optimizer_name == "rangerlite":
            wrapper = RangerLiteWrapper(self, legacy_mode=False)
        elif optimizer_name == "adamw":
            wrapper = AdamWWrapper(self)
        else:
            raise ValueError(
                f"Unknown optimizer_name: '{optimizer_name}'."
            )

        info_str = f"[OptimizerConfig] Using {optimizer_name} optimizer with lr: {self.lr}"
        if self.dense_weight_decay > 0.0 or self.ft_weight_decay > 0.0:
            info_str += f" and ft_weight_decay: {self.ft_weight_decay}, dense_weight_decay: {self.dense_weight_decay}"
        print(info_str + ".")
        return wrapper
