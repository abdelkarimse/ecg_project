"""
scripts/run_compress.py
========================
Step 3: Compress the best model (pruning + quantization + ONNX export).

Usage:
    python scripts/run_compress.py
    python scripts/run_compress.py --config configs/config.yaml --sparsity 0.5
"""

import os
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.model import build_model
from src.compress import run_compression, export_onnx, benchmark_onnx
from src.dataset import build_dataloaders
from src.evaluate import evaluate
from src.utils import set_seed, load_config, get_logger, load_checkpoint, model_size_mb

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/config.yaml")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--sparsity", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["random_seed"])
    data_path = args.data_dir or cfg["data"]["ptbxl_path"]

    if args.sparsity:
        cfg["compression"]["pruning"]["sparsity"] = args.sparsity

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load best trained model
    logger.info("Loading best model checkpoint...")
    model = build_model(cfg["model"])
    ckpt = load_checkpoint(model, cfg["training"]["best_model_path"], device)
    logger.info(f"  Loaded epoch {ckpt.get('epoch', '?')} | Val AUC={ckpt.get('val_auc', '?'):.4f}")
    logger.info(f"  Original size: {model_size_mb(model):.2f} MB")

    # Build calibration / test loader
    _, val_loader, test_loader, _ = build_dataloaders(
        ptbxl_path=data_path,
        sampling_rate=cfg["data"]["sampling_rate"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=4,
        pin_memory=False,
        preproc_config=cfg.get("preprocessing"),
    )

    # ── ONNX Export (from float model) ────────────────────────────────────────
    logger.info("\n📤 Exporting float model to ONNX...")
    float_onnx = "models/ecg_model_float.onnx"
    export_onnx(model, float_onnx)

    # ── Pruning ────────────────────────────────────────────────────────────────
    if cfg["compression"]["pruning"]["enabled"]:
        logger.info("\n🔪 Pruning model...")
        from src.compress import prune_model
        pruned = prune_model(
            model,
            sparsity=cfg["compression"]["pruning"]["sparsity"],
            iterative_steps=cfg["compression"]["pruning"]["iterative_steps"],
        )
        pruned_path = "models/ecg_model_pruned.pth"
        torch.save(pruned.state_dict(), pruned_path)
        logger.info(f"  Pruned model size: {model_size_mb(pruned):.2f} MB")

        logger.info("  Evaluating pruned model...")
        evaluate(pruned, test_loader, report_path="models/eval_pruned.json")

        logger.info("  Exporting pruned ONNX...")
        export_onnx(pruned, "models/ecg_model_pruned.onnx")

    # ── INT8 Quantization ──────────────────────────────────────────────────────
    if cfg["compression"]["quantization"]["enabled"]:
        logger.info("\n📦 Quantizing model...")
        from src.compress import quantize_model
        quantized = quantize_model(
            model,
            calib_loader=val_loader,
            backend=cfg["compression"]["quantization"]["backend"],
            calibration_batches=cfg["compression"]["quantization"]["calibration_batches"],
        )
        quant_path = "models/ecg_model_int8.pth"
        torch.save(quantized.state_dict(), quant_path)
        logger.info(f"  Quantized model size: {model_size_mb(quantized):.2f} MB")

    # ── ONNX Benchmark ─────────────────────────────────────────────────────────
    logger.info("\n⏱  Benchmarking ONNX inference latency...")
    results = benchmark_onnx(float_onnx, n_runs=50)
    if results:
        print(f"\n  {'Metric':<15} {'Value':>10}")
        print("  " + "-" * 26)
        for k, v in results.items():
            print(f"  {k:<15} {v:>9.2f} ms")
        target = "✅ PASS" if results.get("p95_ms", 999) < 200 else "❌ FAIL"
        print(f"\n  Edge target (<200ms): {target}")

    logger.info("\n✅ Compression pipeline complete!")


if __name__ == "__main__":
    main()