"""Training loop for 3D medical image segmentation."""

import logging
import time
from pathlib import Path

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import get_training_datalists
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.unet3d import build_model, count_parameters
from src.training.losses import build_loss
from src.utils.config import get_foreground_label_map
from src.utils.device import get_best_available_device

logger = logging.getLogger(__name__)


class Trainer:
    """End-to-end trainer for 3D U-Net segmentation."""

    def __init__(self, config: dict, fold: int = 0):
        self.config = config
        self.fold = fold
        self.device = get_best_available_device()
        self.label_map = get_foreground_label_map(config)

        # Paths
        self.checkpoint_dir = Path(config["output"]["checkpoint_dir"]) / f"fold_{fold}"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir / "logs"))

        # Post-processing for metrics
        self.post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=config["data"]["num_classes"])])
        self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=config["data"]["num_classes"])])

        # Dice metric (per-class, batch-first)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

        self._setup()

    def _setup(self):
        """Initialize data loaders, model, loss, optimizer, and scheduler."""
        cfg = self.config
        train_cfg = cfg["training"]

        # Data
        train_root = cfg["data"].get("train_dir", cfg["data"].get("processed_dir"))
        if train_root is None:
            raise ValueError("Config must define data.train_dir or data.processed_dir")

        train_files, val_files = get_training_datalists(
            train_root,
            fold=self.fold,
            num_folds=train_cfg["num_folds"],
            seed=train_cfg["seed"],
        )
        logger.info(f"Fold {self.fold}: {len(train_files)} train, {len(val_files)} val patients")

        train_transforms = get_train_transforms(cfg)
        val_transforms = get_val_transforms(cfg)

        train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=cfg["data"]["cache_rate"],
            num_workers=train_cfg["num_workers"],
        )
        val_ds = CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_rate=cfg["data"]["cache_rate"],
            num_workers=train_cfg["num_workers"],
        )

        self.train_loader = DataLoader(
            train_ds,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=train_cfg["num_workers"],
            pin_memory=self.device.type == "cuda",
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=train_cfg["num_workers"],
            pin_memory=self.device.type == "cuda",
        )

        # Model
        self.model = build_model(cfg).to(self.device)
        num_params = count_parameters(self.model)
        logger.info(f"Model parameters: {num_params:,}")

        # Loss
        self.loss_fn = build_loss(cfg).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )

        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=train_cfg["num_epochs"]
        )

        # AMP scaler
        self.scaler = torch.amp.GradScaler("cuda") if train_cfg.get("amp", True) and self.device.type == "cuda" else None
        self.use_amp = self.scaler is not None

        self.best_dice = 0.0
        self.start_epoch = 0

    def _to_device_tensor(self, data, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Convert MONAI MetaTensor/plain tensors to a plain torch tensor on the active device."""
        tensor = data.as_tensor() if hasattr(data, "as_tensor") else data
        return tensor.to(device=self.device, dtype=dtype)

    def train(self):
        """Run the full training loop."""
        cfg = self.config
        num_epochs = cfg["training"]["num_epochs"]
        val_interval = cfg["training"]["val_interval"]

        logger.info(f"Starting training for {num_epochs} epochs on {self.device}")

        for epoch in range(self.start_epoch, num_epochs):
            epoch_start = time.time()

            # --- Train ---
            train_loss = self._train_epoch(epoch)

            # --- LR schedule ---
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # --- Log ---
            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("train/lr", current_lr, epoch)

            elapsed = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # --- Validate ---
            if (epoch + 1) % val_interval == 0:
                mean_dice, per_organ_dice = self._validate(epoch)

                self.writer.add_scalar("val/mean_dice", mean_dice, epoch)
                for i, (_, name) in enumerate(self.label_map.items()):
                    metric_name = name.lower().replace(" ", "_")
                    self.writer.add_scalar(f"val/dice_{metric_name}", per_organ_dice[i], epoch)

                logger.info(
                    f"  Validation | Mean Dice: {mean_dice:.4f} | "
                    + " | ".join(
                        f"{name}: {per_organ_dice[i]:.4f}"
                        for i, (_, name) in enumerate(self.label_map.items())
                    )
                )

                # Save best
                if mean_dice > self.best_dice:
                    self.best_dice = mean_dice
                    self._save_checkpoint(epoch, is_best=True)
                    logger.info(f"  New best model! Dice: {mean_dice:.4f}")

            # Save periodic checkpoint
            if (epoch + 1) % 50 == 0:
                self._save_checkpoint(epoch, is_best=False)

        self.writer.close()
        logger.info(f"Training complete. Best mean Dice: {self.best_dice:.4f}")

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        epoch_loss = 0.0
        num_steps = 0

        for batch in self.train_loader:
            images = self._to_device_tensor(batch["image"], dtype=torch.float32)
            labels = self._to_device_tensor(batch["label"], dtype=torch.long)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item()
            num_steps += 1

        return epoch_loss / max(num_steps, 1)

    @torch.no_grad()
    def _validate(self, epoch: int) -> tuple[float, np.ndarray]:
        """
        Validate using sliding window inference on full volumes.

        Returns:
            Tuple of (mean_dice, per_organ_dice_array).
        """
        self.model.eval()
        self.dice_metric.reset()

        eval_cfg = self.config["evaluation"]

        for batch in self.val_loader:
            images = self._to_device_tensor(batch["image"], dtype=torch.float32)
            labels = self._to_device_tensor(batch["label"], dtype=torch.long)

            # Sliding window inference for full-volume prediction
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = sliding_window_inference(
                        inputs=images,
                        roi_size=eval_cfg["sliding_window_size"],
                        sw_batch_size=eval_cfg["sw_batch_size"],
                        predictor=self.model,
                        overlap=eval_cfg["overlap"],
                    )
            else:
                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=eval_cfg["sliding_window_size"],
                    sw_batch_size=eval_cfg["sw_batch_size"],
                    predictor=self.model,
                    overlap=eval_cfg["overlap"],
                )

            # Post-process predictions and labels
            outputs_list = decollate_batch(outputs)
            labels_list = decollate_batch(labels)
            outputs_post = [self.post_pred(o) for o in outputs_list]
            labels_post = [self.post_label(l) for l in labels_list]

            self.dice_metric(y_pred=outputs_post, y=labels_post)

        # Aggregate: shape [num_classes - 1] (excludes background)
        per_organ_dice = self.dice_metric.aggregate().cpu().numpy()
        mean_dice = per_organ_dice.mean()

        return float(mean_dice), per_organ_dice

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_dice": self.best_dice,
            "config": self.config,
        }
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        if is_best:
            path = self.checkpoint_dir / "best_model.pth"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"

        torch.save(checkpoint, str(path))
        logger.info(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, checkpoint_path: str | Path):
        """Resume training from a checkpoint."""
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_dice = checkpoint["best_dice"]
        self.start_epoch = checkpoint["epoch"] + 1

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(
            f"Resumed from epoch {self.start_epoch} "
            f"(best Dice: {self.best_dice:.4f})"
        )
