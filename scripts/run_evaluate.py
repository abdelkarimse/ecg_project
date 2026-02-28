"""
scripts/run_evaluate.py
========================
Step 4: Evaluate the trained model on the test set and print a full report.

Usage:
    python scripts/run_evaluate.py
    python scripts/run_evaluate.py --model models/best_model.pth
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.model import build_model
from src.dataset import build_dataloaders
from src.evaluate import evaluate
from src.utils import set_seed, load_config, get_logger, load_checkpoint, count_parameters, model_size_mb

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument("--model",     default=None, help="Path to .pth checkpoint")
    parser.add_argument("--data_dir",  default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["random_seed"])
    data_path = args.data_dir or cfg["data"]["ptbxl_path"]
    model_path = args.model or cfg["training"]["best_model_path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("  PTB-XL Edge ECG Challenge — Evaluation")
    logger.info("=" * 60)

    # Load model
    model = build_model(cfg["model"])
    ckpt = load_checkpoint(model, model_path, device)
    logger.info(f"Model: {cfg['model']['architecture']}  |  {count_parameters(model):,} params  |  {model_size_mb(model):.2f} MB")
    logger.info(f"Checkpoint epoch: {ckpt.get('epoch', '?')}  |  Val AUC: {ckpt.get('val_auc', '?'):.4f}")

    # Build test loader
    _, _, test_loader, _ = build_dataloaders(
        ptbxl_path=data_path,
        sampling_rate=cfg["data"]["sampling_rate"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        preproc_config=cfg.get("preprocessing"),
    )

    # Evaluate
    metrics = evaluate(
        model,
        test_loader,
        threshold=args.threshold,
        report_path=cfg["evaluation"]["report_path"],
        device=device,
    )

    # Challenge summary
    macro_auc = metrics["macro_auc"]
    print("\n🎯 Challenge Scorecard")
    print("-" * 40)
    print(f"  Macro-AUC : {macro_auc:.4f}  {'✅' if macro_auc >= 0.90 else '⚠️ (target: ≥0.90)'}")
    print(f"  Model size: {model_size_mb(model):.2f} MB")
    print("  Latency   : Run scripts/rpi_inference.py on Raspberry Pi 5")
    print("-" * 40)


if __name__ == "__main__":
    main()