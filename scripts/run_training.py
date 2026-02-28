"""
scripts/run_training.py
========================
Step 2: Train MobileECG on PTB-XL.

Usage:
    python scripts/run_training.py
    python scripts/run_training.py --config configs/config.yaml --arch ResNet1D
"""

import os
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.model import build_model
from src.dataset import build_dataloaders
from src.train import train
from src.utils import set_seed, load_config, get_logger, count_parameters, model_size_mb

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--arch",   default=None, help="Override architecture")
    parser.add_argument("--data_dir", default=None, help="Override data path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.arch:
        cfg["model"]["architecture"] = args.arch
    data_path = args.data_dir or cfg["data"]["ptbxl_path"]

    set_seed(cfg["data"]["random_seed"])
    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["training"]["best_model_path"]), exist_ok=True)

    # Build DataLoaders
    logger.info("=" * 60)
    logger.info("  PTB-XL Edge ECG Challenge — Training")
    logger.info("=" * 60)

    train_loader, val_loader, test_loader, pos_weights = build_dataloaders(
        ptbxl_path=data_path,
        sampling_rate=cfg["data"]["sampling_rate"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=min(os.cpu_count(), 8),
        pin_memory=torch.cuda.is_available(),
        aug_config=cfg.get("augmentation"),
        preproc_config=cfg.get("preprocessing"),
    )

    # Build Model
    model = build_model(cfg["model"])
    params = count_parameters(model)
    logger.info(f"🏗️  Model: {cfg['model']['architecture']}")
    logger.info(f"   Parameters: {params:,}")
    logger.info(f"   Size (fp32): {model_size_mb(model):.2f} MB")

    # Train
    result = train(model, train_loader, val_loader, cfg["training"], pos_weights)
    logger.info(f"\n🏆 Training complete! Best Val AUC: {result['best_auc']:.4f}")
    logger.info(f"   Best model → {result['best_model_path']}")


if __name__ == "__main__":
    main()