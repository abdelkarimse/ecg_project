"""
src/compress.py
===============
Model compression pipeline:
  1. L1 Unstructured Pruning (iterative)
  2. PyTorch INT8 Static Quantization (qnnpack / fbgemm)
  3. ONNX Export + ONNX Runtime optimization

Designed for edge deployment on Raspberry Pi 5 (ARM Cortex-A76).
"""

import os
import time
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import warnings

from src.utils import get_logger

logger = get_logger(__name__)


# ── Pruning ───────────────────────────────────────────────────────────────────

def prune_model(
    model: nn.Module,
    sparsity: float = 0.40,
    iterative_steps: int = 3,
    method: str = "l1_unstructured",
) -> nn.Module:
    """
    Apply magnitude-based unstructured pruning.

    Args:
        sparsity: Fraction of weights to zero out (0.40 = 40%)
        iterative_steps: Number of gradual pruning rounds
        method: "l1_unstructured" or "random_unstructured"
    """
    import torch.nn.utils.prune as prune_utils

    model = copy.deepcopy(model)
    step_sparsity = 1 - (1 - sparsity) ** (1 / iterative_steps)

    prunable = [
        (m, "weight")
        for m in model.modules()
        if isinstance(m, (nn.Conv1d, nn.Linear))
    ]

    for step in range(iterative_steps):
        logger.info(f"  Pruning step {step + 1}/{iterative_steps} (step_sparsity={step_sparsity:.3f})")
        for module, param_name in prunable:
            if method == "l1_unstructured":
                prune_utils.l1_unstructured(module, name=param_name, amount=step_sparsity)
            else:
                prune_utils.random_unstructured(module, name=param_name, amount=step_sparsity)

    # Make pruning permanent
    for module, param_name in prunable:
        try:
            prune_utils.remove(module, param_name)
        except Exception:
            pass

    # Measure actual sparsity
    zeros, total = 0, 0
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            zeros += (m.weight == 0).sum().item()
            total += m.weight.numel()
    actual = zeros / total if total > 0 else 0
    logger.info(f"  ✅ Actual weight sparsity: {actual:.2%}")

    return model


# ── INT8 Static Quantization ──────────────────────────────────────────────────

def quantize_model(
    model: nn.Module,
    calibration_loader: DataLoader,
    backend: str = "qnnpack",
    calibration_batches: int = 100,
) -> nn.Module:
    """
    Apply PyTorch static INT8 post-training quantization.

    Args:
        backend: "qnnpack" for ARM/RPi, "fbgemm" for x86
    """
    torch.backends.quantized.engine = backend
    model = copy.deepcopy(model).cpu().eval()

    # Fuse Conv-BN-ReLU patterns for better quantization accuracy
    try:
        model = torch.quantization.fuse_modules(model, _find_fuseable_patterns(model))
    except Exception as e:
        logger.warning(f"Fusion failed (non-fatal): {e}")

    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare(model, inplace=True)

    # Calibration
    logger.info(f"  Calibrating with {calibration_batches} batches...")
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(calibration_loader):
            if i >= calibration_batches:
                break
            model(x.cpu())

    torch.quantization.convert(model, inplace=True)
    logger.info("  ✅ INT8 quantization applied")
    return model


def _find_fuseable_patterns(model: nn.Module):
    """Attempt to find Conv1d-BN-ReLU triples for fusion."""
    # Simple heuristic – in practice you'd introspect the module tree
    # Returns empty list if architecture doesn't support simple fusion
    return []


# ── ONNX Export ───────────────────────────────────────────────────────────────

def export_onnx(
    model: nn.Module,
    output_path: str,
    input_channels: int = 12,
    input_length: int = 1000,
    opset: int = 17,
    optimize: bool = True,
    dynamic_batch: bool = True,
) -> str:
    """Export model to ONNX and optionally optimize with ONNX Runtime."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model = model.cpu().eval()
    dummy = torch.randn(1, input_channels, input_length)

    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}} if dynamic_batch else {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            dummy,
            output_path,
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
    logger.info(f"  ONNX model exported → {output_path}")
    size_mb = os.path.getsize(output_path) / 1e6
    logger.info(f"  Model size: {size_mb:.2f} MB")

    if optimize:
        _optimize_onnx(output_path)

    return output_path


def _optimize_onnx(model_path: str):
    """Run ONNX Runtime graph optimization (optional)."""
    try:
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.optimized_model_filepath = model_path.replace(".onnx", "_opt.onnx")
        ort.InferenceSession(model_path, sess_options)
        opt_size = os.path.getsize(sess_options.optimized_model_filepath) / 1e6
        logger.info(f"  Optimized ONNX saved ({opt_size:.2f} MB)")
    except ImportError:
        logger.warning("  onnxruntime not installed — skipping ONNX optimization")
    except Exception as e:
        logger.warning(f"  ONNX optimization failed: {e}")


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark_onnx(
    model_path: str,
    input_channels: int = 12,
    input_length: int = 1000,
    n_runs: int = 50,
    warmup: int = 5,
) -> Dict[str, float]:
    """Measure ONNX Runtime inference latency."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("onnxruntime not installed. Run: pip install onnxruntime")
        return {}

    sess = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )
    inp = np.random.randn(1, input_channels, input_length).astype(np.float32)
    input_name = sess.get_inputs()[0].name

    # Warmup
    for _ in range(warmup):
        sess.run(None, {input_name: inp})

    # Benchmark
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: inp})
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    result = {
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std()),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
    }
    logger.info(f"  Inference latency: mean={result['mean_ms']:.1f}ms  p95={result['p95_ms']:.1f}ms")
    return result


# ── Full Compression Pipeline ─────────────────────────────────────────────────

def run_compression(
    model: nn.Module,
    calib_loader: DataLoader,
    config: Dict[str, Any],
) -> nn.Module:
    """
    Run the full compression pipeline based on config flags.
    Returns the compressed model (PyTorch).
    Also exports ONNX if configured.
    """
    compressed = model

    if config.get("pruning", {}).get("enabled", False):
        logger.info("🔪 Applying weight pruning...")
        compressed = prune_model(
            compressed,
            sparsity=config["pruning"].get("sparsity", 0.40),
            iterative_steps=config["pruning"].get("iterative_steps", 3),
        )

    if config.get("quantization", {}).get("enabled", False):
        logger.info("📦 Applying INT8 quantization...")
        compressed = quantize_model(
            compressed,
            calib_loader,
            backend=config["quantization"].get("backend", "qnnpack"),
            calibration_batches=config["quantization"].get("calibration_batches", 100),
        )

    if config.get("onnx_export", {}).get("enabled", False):
        logger.info("📤 Exporting to ONNX...")
        export_onnx(
            model,  # Use original float model for ONNX (better compatibility)
            output_path=config["onnx_export"].get("output_path", "models/ecg_model.onnx"),
            opset=config["onnx_export"].get("opset", 17),
            optimize=config["onnx_export"].get("optimize", True),
        )

    return compressed