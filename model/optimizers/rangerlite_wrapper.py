import math

import lightning as L
import torch

from dataclasses import dataclass
from .ranger_lite import RangerLite

@dataclass(kw_only=True, frozen=False)
class RangerLiteConfig:
    gamma: float = 0.992
    """Multiplicative factor applied to the learning rate after every epoch."""

    one_cycle_steps: int = 0
    """Number of steps for the One Cycle LR scheduler. If set to a positive value, One Cycle LR scheduler will be used. If set to 0 or a negative value, StepLR with step_size=1 will be used."""

    one_cycle_warmup_pct: float = 0.2
    """Fraction of the cycle to spend increasing the learning rate in the One Cycle LR scheduler."""

    one_cycle_start_div: float = 25
    """Initial lr div factor when using One Cycle LR scheduler."""

    one_cycle_final_div: float = 50
    """Final lr div factor when using One Cycle LR scheduler."""

    one_cycle_refinement_steps: int = 0
    """Optional cosine-refinement steps after OneCycle, using the same optimizer."""

    one_cycle_refinement_lr: float = 0.0
    """Learning rate applied once at the OneCycle-to-refinement boundary."""

    one_cycle_refinement_final_lr: float = 0.0
    """Final learning rate of the optional cosine-refinement phase."""

    pnm_active: bool = True
    """Whether to activate Positive Negative Momentum."""

    pnm_momentum: float = 1.0
    """Positive Negative Momentum parameter. Value of 1.0 corresponds to ranger21 behaviour. Note: `pnm_momentum` was hardcoded to 1.0 in ranger21. The argument was unused."""

    lookahead_alpha: float = 0.5
    """Lookahead alpha parameter. Value of 0.5 corresponds to ranger21 behaviour."""

    lookahead_steps: int = 5
    """Lookahead steps parameter. Value of 5 corresponds to ranger21 behaviour."""

class SafeOneCycleLR(torch.optim.lr_scheduler.OneCycleLR):
    def step(self, epoch=None):
        if self.last_epoch < self.total_steps - 1:
            super().step(epoch)


class OneCycleCosineRefinementLR(SafeOneCycleLR):
    """OneCycle followed by a low-LR cosine refinement without an optimizer reset.

    The OneCycle prefix deliberately uses PyTorch's implementation unchanged.
    After its final scheduled update, the next batch starts at ``refinement_lr``
    and cosine-anneals to ``refinement_final_lr`` over ``refinement_steps``.
    """

    def __init__(
        self,
        optimizer,
        *,
        max_lr,
        one_cycle_steps,
        refinement_steps,
        refinement_lr,
        refinement_final_lr,
        div_factor,
        final_div_factor,
        pct_start,
    ):
        if one_cycle_steps <= 0:
            raise ValueError("one_cycle_steps must be positive")
        if refinement_steps < 2:
            raise ValueError("refinement_steps must be at least 2")
        if refinement_lr <= 0.0:
            raise ValueError("refinement_lr must be positive")
        if not 0.0 <= refinement_final_lr <= refinement_lr:
            raise ValueError(
                "refinement_final_lr must be between 0 and refinement_lr"
            )

        self.one_cycle_steps = one_cycle_steps
        self.refinement_steps = refinement_steps
        self.refinement_lr = refinement_lr
        self.refinement_final_lr = refinement_final_lr
        super().__init__(
            optimizer,
            max_lr=max_lr,
            total_steps=one_cycle_steps,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            pct_start=pct_start,
            cycle_momentum=False,
        )

    def step(self, epoch=None):
        if self.last_epoch < self.one_cycle_steps - 1:
            return super().step(epoch)

        if epoch is not None:
            raise ValueError(
                "OneCycleCosineRefinementLR does not support an explicit epoch"
            )

        final_epoch = self.one_cycle_steps + self.refinement_steps - 1
        if self.last_epoch >= final_epoch:
            return

        self._step_count += 1
        self.last_epoch += 1
        refinement_step = self.last_epoch - self.one_cycle_steps
        progress = refinement_step / (self.refinement_steps - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = self.refinement_final_lr + (
            self.refinement_lr - self.refinement_final_lr
        ) * cosine

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self._last_lr = [lr] * len(self.optimizer.param_groups)


class RangerLiteWrapper:
    def __init__(
        self,
        config,
        legacy_mode,
    ):
        self.gamma = config.gamma
        self.pnm_active = config.pnm_active
        self.pnm_momentum = config.pnm_momentum
        self.lookahead_alpha = config.lookahead_alpha
        self.lookahead_steps = config.lookahead_steps
        self.cycle_steps = config.one_cycle_steps
        self.one_cycle_warmup_pct = config.one_cycle_warmup_pct
        self.one_cycle_start_div = config.one_cycle_start_div
        self.one_cycle_final_div = config.one_cycle_final_div
        self.one_cycle_refinement_steps = config.one_cycle_refinement_steps
        self.one_cycle_refinement_lr = config.one_cycle_refinement_lr
        self.one_cycle_refinement_final_lr = config.one_cycle_refinement_final_lr
        self.legacy_mode = legacy_mode
        self.needs_train_flip = True

        self.optimizer = None

    def configure_optimizers(self, train_params):
        # train_params is expected to be a list of dicts: [{'params': ..., 'lr': ..., 'weight_decay': ...}]
        self.optimizer = RangerLite(
            train_params,
            # Global defaults acting as fallbacks if not defined in param groups
            lr=1.0,
            weight_decay=0.0,
            use_legacy_scoping_bug=self.legacy_mode,
            normloss_active=self.legacy_mode,
            pnm_activate=self.pnm_active,
            pnm_momentum=self.pnm_momentum,
            lookahead_blending_alpha=self.lookahead_alpha,
            lookahead_mergetime=self.lookahead_steps,
        )

        if self.cycle_steps <= 0:
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=1, gamma=self.gamma
            )

        else:
            LRs = [group["lr"] for group in train_params]
            if self.one_cycle_refinement_steps > 0:
                one_cycle_scheduler = OneCycleCosineRefinementLR(
                    self.optimizer,
                    max_lr=LRs,
                    one_cycle_steps=self.cycle_steps,
                    refinement_steps=self.one_cycle_refinement_steps,
                    refinement_lr=self.one_cycle_refinement_lr,
                    refinement_final_lr=self.one_cycle_refinement_final_lr,
                    div_factor=self.one_cycle_start_div,
                    final_div_factor=self.one_cycle_final_div,
                    pct_start=self.one_cycle_warmup_pct,
                )
            else:
                one_cycle_scheduler = SafeOneCycleLR(
                    self.optimizer,
                    max_lr=LRs,
                    total_steps=self.cycle_steps,
                    div_factor=self.one_cycle_start_div,
                    final_div_factor=self.one_cycle_final_div,
                    pct_start=self.one_cycle_warmup_pct,
                    cycle_momentum=False,
                )
            scheduler = {"scheduler": one_cycle_scheduler, "interval": "step"}

        return [self.optimizer], [scheduler]

    def switch_to_train(self, force=False):
        if (force or self.needs_train_flip) and not self.legacy_mode:
            self.optimizer.train()
            self.needs_train_flip = False

    def switch_to_eval(self):
        if not self.legacy_mode:
            self.optimizer.eval()
            self.needs_train_flip = True

    def on_train_epoch_start(self, pl_module: L.LightningModule):
        self.switch_to_train(True)

    def on_train_batch_start(self, pl_module: L.LightningModule, batch, batch_idx):
        self.switch_to_train()

    def on_validation_epoch_start(self, pl_module: L.LightningModule):
        self.switch_to_eval()

    def on_test_epoch_start(self, pl_module: L.LightningModule):
        self.switch_to_eval()

    def on_train_epoch_end(self, pl_module: L.LightningModule):
        self.switch_to_eval()

    def on_save_checkpoint(self, pl_module: L.LightningModule, checkpoint):
        self.switch_to_eval()
