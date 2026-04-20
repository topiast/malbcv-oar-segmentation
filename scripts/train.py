#!/usr/bin/env python
"""
Training entry point for 3D segmentation.

Usage:
    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/baseline.yaml --fold 0
    python scripts/train.py --config configs/baseline.yaml --fold 0 --resume results/checkpoints/fold_0/best_model.pth
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.trainer import Trainer
from src.utils.device import get_best_available_device, get_device_name
from src.utils.config import resolve_config_paths


def main():
    parser = argparse.ArgumentParser(description="Train 3D U-Net for volumetric segmentation")
    parser.add_argument(
        "--config", type=str, default="configs/baseline.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--fold", type=int, default=0,
        help="Cross-validation fold (0 to num_folds-1). Use -1 to train all folds.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training.log"),
        ],
    )
    logger = logging.getLogger(__name__)

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    with config_path.open() as f:
        config = resolve_config_paths(yaml.safe_load(f), project_root)

    logger.info(f"Config loaded from {config_path}")
    logger.info(f"Device: {get_device_name(get_best_available_device())}")

    # Determine folds to train
    num_folds = config["training"]["num_folds"]
    if args.fold == -1:
        folds = list(range(num_folds))
    else:
        folds = [args.fold]

    for fold in folds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training fold {fold}/{num_folds - 1}")
        logger.info(f"{'='*60}")

        trainer = Trainer(config, fold=fold)

        if args.resume:
            trainer.load_checkpoint(args.resume)

        trainer.train()

    logger.info("All folds complete!")


if __name__ == "__main__":
    main()
