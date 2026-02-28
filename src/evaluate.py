"""
src/evaluate.py
===============
Comprehensive evaluation: Macro-AUC, per-class AUC, F1, accuracy,
confusion matrices, and a printable classification report.
"""

import json
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    classification_report,
    multilabel_confusion_matrix,
)
from typing import Dict, Any, List, Optional

from src.dataset import SUPERCLASSES
from src.utils import get_logger

logger = get_logger(__name__)


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    """Run full pass over loader, return (probs, labels) as numpy arrays."""
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).float().cpu()
        probs = torch.sigmoid(logits)
        all_probs.append(probs.numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    class_names: List[str] = SUPERCLASSES,
) -> Dict[str, Any]:
    preds = (probs >= threshold).astype(int)

    macro_auc = roc_auc_score(labels, probs, average="macro")
    per_class_auc = roc_auc_score(labels, probs, average=None).tolist()

    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_per   = f1_score(labels, preds, average=None, zero_division=0).tolist()

    acc = accuracy_score(labels.flatten(), preds.flatten())

    report = classification_report(
        labels, preds,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    return {
        "macro_auc": macro_auc,
        "per_class_auc": dict(zip(class_names, per_class_auc)),
        "f1_macro": f1_macro,
        "f1_per_class": dict(zip(class_names, f1_per)),
        "accuracy": acc,
        "classification_report": report,
    }


def print_metrics(metrics: Dict[str, Any]):
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Macro AUC  : {metrics['macro_auc']:.4f}")
    print(f"  Macro F1   : {metrics['f1_macro']:.4f}")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print("\n  Per-Class AUC:")
    for cls, auc in metrics["per_class_auc"].items():
        print(f"    {cls:<8}: {auc:.4f}")
    print("\n  Per-Class F1:")
    for cls, f1 in metrics["f1_per_class"].items():
        print(f"    {cls:<8}: {f1:.4f}")
    print("=" * 60 + "\n")


# ── Full Evaluation Pipeline ───────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    threshold: float = 0.5,
    report_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    logger.info("Running evaluation on test set...")
    probs, labels = collect_predictions(model, test_loader, device)
    metrics = compute_metrics(probs, labels, threshold)
    print_metrics(metrics)

    if report_path:
        import os
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Report saved to {report_path}")

    return metrics