"""
src/train.py
============
Training loop with:
  - Mixed-precision (AMP)
  - Cosine annealing LR with warmup
  - BCEWithLogitsLoss + auto class weighting
  - Early stopping on validation Macro-AUC
  - Checkpoint saving
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any

import numpy as np
from sklearn.metrics import roc_auc_score

from src.utils import AverageMeter, set_seed, get_logger

logger = get_logger(__name__)


# ── LR Scheduler with Warmup ──────────────────────────────────────────────────

def build_scheduler(optimizer, epochs, warmup_epochs, steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch

    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    return scheduler


# ── One Epoch ─────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion,
    scaler: Optional[GradScaler],
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    all_logits, all_labels = [], []
    t0 = time.time()

    for step, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        loss_meter.update(loss.item(), x.size(0))
        all_logits.append(logits.detach().float().cpu())
        all_labels.append(y.detach().float().cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-all_logits))
    try:
        auc = roc_auc_score(all_labels, probs, average="macro")
    except Exception:
        auc = float("nan")

    return {
        "loss": loss_meter.avg,
        "macro_auc": auc,
        "time": time.time() - t0,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    all_logits, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_meter.update(loss.item(), x.size(0))
        all_logits.append(logits.float().cpu())
        all_labels.append(y.float().cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-all_logits))
    try:
        auc = roc_auc_score(all_labels, probs, average="macro")
        per_class = roc_auc_score(all_labels, probs, average=None)
    except Exception:
        auc = float("nan")
        per_class = [float("nan")] * 5

    return {
        "loss": loss_meter.avg,
        "macro_auc": auc,
        "per_class_auc": per_class.tolist() if hasattr(per_class, "tolist") else per_class,
    }


# ── Full Training ─────────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    pos_weights: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Full training procedure.

    Returns:
        dict with training history and best checkpoint path
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    model = model.to(device)

    # Loss
    if pos_weights is not None and config.get("pos_weight_auto", True):
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Scheduler
    scheduler = build_scheduler(
        optimizer,
        epochs=config["epochs"],
        warmup_epochs=config.get("warmup_epochs", 5),
        steps_per_epoch=len(train_loader),
    )

    # AMP
    scaler = GradScaler() if config.get("mixed_precision", True) and device.type == "cuda" else None

    # Setup
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    best_auc = 0.0
    patience = config.get("early_stopping_patience", 15)
    no_improve = 0
    history = {"train": [], "val": []}

    logger.info(f"Starting training for {config['epochs']} epochs...")
    for epoch in range(1, config["epochs"] + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch)
        scheduler.step()
        val_metrics = validate(model, val_loader, criterion, device)

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        auc = val_metrics["macro_auc"]
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch:03d}/{config['epochs']} | "
            f"Train loss={train_metrics['loss']:.4f} AUC={train_metrics['macro_auc']:.4f} | "
            f"Val loss={val_metrics['loss']:.4f} AUC={auc:.4f} | "
            f"LR={lr:.2e} | time={train_metrics['time']:.1f}s"
        )

        # Checkpoint
        if auc > best_auc:
            best_auc = auc
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": auc,
                    "val_metrics": val_metrics,
                    "config": config,
                },
                config["best_model_path"],
            )
            logger.info(f"  ✅ New best model saved (AUC={best_auc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"  ⏹ Early stopping triggered after {epoch} epochs.")
                break

    logger.info(f"\n🏆 Best validation Macro-AUC: {best_auc:.4f}")

    # Save history
    hist_path = os.path.join(config["checkpoint_dir"], "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    return {"best_auc": best_auc, "history": history, "best_model_path": config["best_model_path"]}