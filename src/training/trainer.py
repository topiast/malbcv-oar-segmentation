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
from src.models import build_model, count_parameters
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
            split_file=cfg["data"].get("split_file"),
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
        self.train_iterations_per_epoch = train_cfg.get("train_iterations_per_epoch", len(self.train_loader))
        self._train_iterator = None

        # Model
        self.model = build_model(cfg).to(self.device)
        num_params = count_parameters(self.model)
        logger.info(f"Model parameters: {num_params:,}")

        # Loss
        self.loss_fn = build_loss(cfg).to(self.device)

        # Optimizer
        base_lr = train_cfg["learning_rate"]
        transformer_lr_ratio = train_cfg.get("transformer_lr_ratio")
        if transformer_lr_ratio is not None and hasattr(self.model, "get_param_groups"):
            parameters = self.model.get_param_groups(base_lr, transformer_lr_ratio)
        else:
            parameters = self.model.parameters()

        optimizer_name = train_cfg.get("optimizer", "adamw").lower()
        if optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(
                parameters,
                lr=base_lr,
                momentum=train_cfg.get("momentum", 0.9),
                weight_decay=train_cfg["weight_decay"],
                nesterov=train_cfg.get("nesterov", False),
            )
        else:
            self.optimizer = torch.optim.AdamW(
                parameters,
                lr=base_lr,
                weight_decay=train_cfg["weight_decay"],
            )

        # LR scheduler
        scheduler_name = train_cfg.get("scheduler", "cosine").lower()
        if scheduler_name == "polynomial":
            power = train_cfg.get("poly_power", 0.9)

            def _poly_lambda(epoch: int) -> float:
                progress = min(epoch / max(train_cfg["num_epochs"], 1), 1.0)
                return max((1.0 - progress) ** power, 0.0)

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=_poly_lambda
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=train_cfg["num_epochs"]
            )

        # AMP scaler
        self.scaler = torch.amp.GradScaler("cuda") if train_cfg.get("amp", True) and self.device.type == "cuda" else None
        self.use_amp = self.scaler is not None

        self.best_dice = 0.0
        self.start_epoch = 0

    def _next_train_batch(self):
        """Cycle over the training loader to support a fixed iteration budget per epoch."""
        if self._train_iterator is None:
            self._train_iterator = iter(self.train_loader)
        try:
            return next(self._train_iterator)
        except StopIteration:
            self._train_iterator = iter(self.train_loader)
            return next(self._train_iterator)

    def _to_device_tensor(self, data, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Convert MONAI MetaTensor/plain tensors to a plain torch tensor on the active device."""
        tensor = data.as_tensor() if hasattr(data, "as_tensor") else data
        return tensor.to(device=self.device, dtype=dtype)

    @staticmethod
    def _extract_logits(outputs):
        """Return dense semantic logits from either a tensor or a structured model output."""
        if isinstance(outputs, dict):
            return outputs["logits"]
        return outputs

    def _predict_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Tensor-only model wrapper for sliding-window inference."""
        return self._extract_logits(self.model(images))

    def train(self):
        """Run the full training loop."""
        cfg = self.config
        num_epochs = cfg["training"]["num_epochs"]
        val_interval = cfg["training"]["val_interval"]

        logger.info(f"Starting training for {num_epochs} epochs on {self.device}")

        for epoch in range(self.start_epoch, num_epochs):
            epoch_start = time.time()

            # --- Train ---
            train_metrics = self._train_epoch(epoch)
            train_loss = train_metrics["loss"]

            # --- LR schedule ---
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # --- Log ---
            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("train/lr", current_lr, epoch)
            for name, value in train_metrics.items():
                if name == "loss":
                    continue
                self.writer.add_scalar(f"train/{name}", value, epoch)

            elapsed = time.time() - epoch_start
            component_summary = ""
            if "class_ce" in train_metrics:
                component_summary = (
                    f" | Class CE: {train_metrics['class_ce']:.4f}"
                    f" | Mask BCE: {train_metrics['mask_bce']:.4f}"
                    f" | Mask Dice: {train_metrics['mask_dice']:.4f}"
                )
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
                f"{component_summary}"
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

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Train for one epoch. Returns average training metrics."""
        self.model.train()
        epoch_loss = 0.0
        num_steps = 0
        component_sums: dict[str, float] = {}

        for _ in range(self.train_iterations_per_epoch):
            batch = self._next_train_batch()
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
            if hasattr(self.loss_fn, "get_last_components"):
                for name, value in self.loss_fn.get_last_components().items():
                    component_sums[name] = component_sums.get(name, 0.0) + float(value)
            num_steps += 1

        metrics = {"loss": epoch_loss / max(num_steps, 1)}
        for name, total in component_sums.items():
            metrics[name] = total / max(num_steps, 1)
        return metrics

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
                        predictor=self._predict_logits,
                        overlap=eval_cfg["overlap"],
                    )
            else:
                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=eval_cfg["sliding_window_size"],
                    sw_batch_size=eval_cfg["sw_batch_size"],
                    predictor=self._predict_logits,
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
