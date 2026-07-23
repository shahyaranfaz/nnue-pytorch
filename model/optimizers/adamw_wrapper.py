import lightning as L
import torch


class AdamWWrapper:
    def __init__(self, config):
        self.epochs = config.adamw_epochs
        self.final_lr = config.adamw_final_lr
        self.optimizer = None

    def configure_optimizers(self, train_params):
        self.optimizer = torch.optim.AdamW(
            train_params,
            lr=1.0,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=self.final_lr,
        )
        return [self.optimizer], [scheduler]

    def switch_to_train(self, force=False):
        pass

    def switch_to_eval(self):
        pass

    def on_train_epoch_start(self, pl_module: L.LightningModule):
        pass

    def on_train_batch_start(self, pl_module, batch, batch_idx):
        pass

    def on_validation_epoch_start(self, pl_module):
        pass

    def on_test_epoch_start(self, pl_module):
        pass

    def on_train_epoch_end(self, pl_module):
        pass

    def on_save_checkpoint(self, pl_module, checkpoint):
        pass
