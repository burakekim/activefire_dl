import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision

from model import UNetWithSkipv2


class UNetTrainer(pl.LightningModule):
    """Pytorch Lightning triner class for the pixel-wise classification
    task of active fire detection."""

    def __init__(
        self,
        input_ch=10,
        use_act=None,
        enc_ch=(32, 64, 128, 256, 512, 1024),
        lr=1e-4,
        tb_log_pred_gt=False,
    ):
        """Initialize the UNetTrainer class.

        Args:
            input_ch: number of input channels
            enc_ch: encoder channels
            use_act: activation function
            lr: learning rate
            tb_log_pred_gt: whether to plot predictions and annotations in tensorboard

        """
        super().__init__()
        self.mse_err = torchmetrics.MeanSquaredError()
        self.mae_err = torchmetrics.MeanAbsoluteError()
        self.iou_err = torchmetrics.JaccardIndex(task="binary")
        self.acc = torchmetrics.classification.Accuracy(task="binary")
        self.lr = lr
        self.tb_log_pred_gt = tb_log_pred_gt
        self.loss = nn.BCEWithLogitsLoss()

        self.model = UNetWithSkipv2(
            input_ch=input_ch,
            use_act=use_act,
            encoder_channels=enc_ch,
        )

        self.validation_preds = []
        self.validation_targets = []

    def forward(self, x):
        """Run forward pass."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Define the training step."""
        x, y, _ = batch
        x = x.to(memory_format=torch.channels_last)
        x = x.float()
        y = y.float()

        preds = self.model.forward(x)
        loss = self.loss(preds, y)

        pred_mask = (preds > 0.5).float()
        mse_error = self.mse_err(pred_mask, y)
        iou_error = self.iou_err(pred_mask, y)
        accuracy = self.acc(pred_mask, y)
        metrics = {
            "train_loss": loss,
            "train_mse_err": mse_error,
            "train_iou_err": iou_error,
            "train_accuracy": accuracy,
        }
        self.log_dict(metrics, logger=True, prog_bar=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """Define the validation step."""
        x, y, _ = batch
        x = x.to(memory_format=torch.channels_last)
        x = x.float()
        y = y.float()

        preds = self.model.forward(x)
        loss = self.loss(preds, y)

        pred_mask = (preds > 0.5).float()
        mse_error = self.mse_err(pred_mask, y)
        iou_error = self.iou_err(pred_mask, y)
        accuracy = self.acc(pred_mask, y)

        self.validation_preds.append(pred_mask)
        self.validation_targets.append(y)

        metrics = {
            "val_loss": loss,
            "val_mse_err": mse_error,
            "val_iou_err": iou_error,
            "val_accuracy": accuracy,
        }
        self.log_dict(metrics, logger=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        """Define the prediction and annotation plotting step after validation."""
        if self.tb_log_pred_gt:
            preds = torch.cat(self.validation_preds, dim=0)
            targets = torch.cat(self.validation_targets, dim=0)

            grid_preds = torchvision.utils.make_grid(preds)
            grid_targets = torchvision.utils.make_grid(targets)

            self.logger.experiment.add_image(
                "predictions", grid_preds, self.current_epoch
            )
            self.logger.experiment.add_image(
                "targets", grid_targets, self.current_epoch
            )

            self.validation_preds.clear()
            self.validation_targets.clear()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        """Define the test step."""
        x, y, _ = batch
        x = x.to(memory_format=torch.channels_last)
        x = x.float()
        y = y.float()

        preds = self.model.forward(x)
        loss = self.loss(preds, y)

        pred_mask = (preds > 0.5).float()
        mse_error = self.mse_err(pred_mask, y)
        iou_error = self.iou_err(pred_mask, y)
        accuracy = self.acc(pred_mask, y)
        metrics = {
            "test_mse_err": mse_error,
            "test_iou_err": iou_error,
            "test_loss": loss,
            "test_accuracy": accuracy,
        }
        self.log_dict(metrics, logger=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self, use_lr_scheduler=True):
        """Configure the optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if use_lr_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=3,
                T_mult=1,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "epoch",
                },
            }
        else:
            return optimizer
