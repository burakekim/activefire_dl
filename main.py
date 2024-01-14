import argparse
import os

import kornia.augmentation as K
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchgeo.transforms import AugmentationSequential

from dataset import ActiveFire
from trainer import UNetTrainer


def main(args):
    """Run train and tests loops after defining the environmetal variables,
    datasets, dataloaders, and model."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    pl.seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_transforms_gpu = AugmentationSequential(
        K.RandomHorizontalFlip(p=0.3, keepdim=True),
        K.RandomVerticalFlip(p=0.3, keepdim=True),
        K.RandomSharpness(sharpness=0.5, p=0.2, keepdim=True),
        K.RandomErasing(
            scale=(0.02, 0.33),
            ratio=(0.3, 3.3),
            value=0.0,
            same_on_batch=False,
            p=0.4,
            keepdim=True,
        ),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.2, keepdim=True),
        data_keys=["image", "mask"],
    ).to(device)

    train_dataset = ActiveFire(
        img_dir=args.fire_path,
        mask_dir=args.mask_path,
        split="train",
        transforms=train_transforms_gpu,
    )
    val_dataset = ActiveFire(
        img_dir=args.fire_path,
        mask_dir=args.mask_path,
        split="val",
        transforms=None,
    )
    test_dataset = ActiveFire(
        img_dir=args.fire_path,
        mask_dir=args.mask_path,
        split="test",
        transforms=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
        num_workers=os.cpu_count() - 1,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
        num_workers=os.cpu_count() - 1,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
        num_workers=os.cpu_count() - 1,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=args.patience,
        verbose=True,
        mode="min",
    )

    logname = (
        f"{args.exp_name}_{args.batch_size}" + "{epoch:02d}-{val_loss:.2f}"
    )  # noqa: E501

    tb_logger = TensorBoardLogger(save_dir=os.getcwd(), name="lightning_logs")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="fire_unetsimple",
        filename=logname,
        save_top_k=1,
        mode="min",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval=None)

    model = UNetTrainer(
        input_ch=args.input_ch,
        enc_ch=args.encoder_ch,
        use_act=args.use_act,
        lr=args.lr,
        tb_log_pred_gt=args.tb_log_pred_gt,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision="16-mixed",
        devices=[0],
        enable_progress_bar=True,
        logger=tb_logger,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    print("Best model path: ", checkpoint_callback.best_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")

    # Add your arguments
    parser.add_argument(
        "--cuda-visible-devices",
        default="2",
        type=str,
        help="CUDA Visible Devices",
    )
    parser.add_argument(
        "--fire-path",
        default="/data/active_fire_dataset/fire_images",
        type=str,
        help="Path to image root",
    )
    parser.add_argument(
        "--mask-path",
        default="/data/active_fire_dataset/fire_masks",
        type=str,
        help="Path to mask root",
    )
    parser.add_argument(
        "--input-ch", default=10, type=int, help="Number of input channels"
    )
    parser.add_argument(
        "--encoder-ch",
        default=(32, 64, 128, 256, 512, 1024, 2048),
        type=int,
        help="Encoder channels",
    )
    parser.add_argument("--use-act", default=None, type=int, help="Activation function")
    parser.add_argument("--lr", default=1e-4, type=int, help="Learning rate")
    parser.add_argument(
        "--tb-log-pred-gt", default=True, type=int, help="Viz pred and gt with TB"
    )
    parser.add_argument("--seed", default=42, type=int, help="Seed")
    parser.add_argument(
        "--batch-size", default=512, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "--max-epochs", default=100, type=int, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--patience", default=10, type=int, help="Patience for early stopping"
    )
    parser.add_argument(
        "--exp-name", default="06_activefire", type=str, help="Experiment name"
    )

    args = parser.parse_args()
    main(args)
