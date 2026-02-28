"""
src/utils.py
============
Shared utilities: logging, seeding, metrics helpers, YAML loading.
"""

import os
import random
import logging
import numpy as np
import torch
import yaml
from typing import Any, Dict


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


# ── Seeding ───────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── Meters ────────────────────────────────────────────────────────────────────

class AverageMeter:
    """Running mean/count tracker."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


# ── Model IO ──────────────────────────────────────────────────────────────────

def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device) -> Dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: torch.nn.Module) -> float:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(model.state_dict(), f.name)
        return os.path.getsize(f.name) / 1e6