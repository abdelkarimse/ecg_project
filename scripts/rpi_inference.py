"""
scripts/rpi_inference.py
=========================
Real-time ECG inference demo for Raspberry Pi 5.

Loads a WFDB record (or random signal), runs inference with ONNX Runtime,
and displays:
  - Predicted diagnostic classes with confidence scores
  - Measured latency per inference

Usage (on Raspberry Pi 5):
    python scripts/rpi_inference.py --model models/ecg_model_float.onnx --record /path/to/record
    python scripts/rpi_inference.py --model models/ecg_model_float.onnx --demo   # synthetic input
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
DESCRIPTIONS = {
    "NORM": "Normal ECG",
    "MI":   "Myocardial Infarction",
    "STTC": "ST/T-wave Change",
    "CD":   "Conduction Disturbance",
    "HYP":  "Hypertrophy",
}

BANNER = """
╔══════════════════════════════════════════════════════╗
║     PTB-XL Edge ECG Classifier — Raspberry Pi 5     ║
╚══════════════════════════════════════════════════════╝
"""


def load_onnx_session(model_path: str):
    """Load ONNX Runtime inference session (CPU, ARM-optimized)."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("❌ onnxruntime not installed. Run: pip install onnxruntime")
        sys.exit(1)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4        # RPi 5 has 4 cores
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    sess = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
    print(f"✅ Model loaded: {model_path}")
    print(f"   Input  : {sess.get_inputs()[0].shape}")
    print(f"   Output : {sess.get_outputs()[0].shape}\n")
    return sess


def load_ecg_record(record_path: str, target_len: int = 1000) -> np.ndarray:
    """Load a WFDB record and preprocess it."""
    import wfdb
    from src.preprocessing import ECGPreprocessor

    record = wfdb.rdrecord(record_path)
    fs_raw = record.fs
    signal = record.p_signal.T.astype(np.float32)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    prep = ECGPreprocessor(
        fs_in=float(fs_raw),
        fs_out=100.0,
        target_len=target_len,
    )
    return prep(signal)


def make_demo_signal(channels: int = 12, length: int = 1000) -> np.ndarray:
    """Generate a synthetic ECG-like signal for demo purposes."""
    t = np.linspace(0, 10, length)
    sig = np.zeros((channels, length), dtype=np.float32)
    for i in range(channels):
        hr = 70 + i * 2  # slightly different per lead
        f_hr = hr / 60.0
        # Simplified QRS approximation
        qrs = 1.5 * np.sin(2 * np.pi * f_hr * t) * np.exp(-((t % (1 / f_hr) - 0.05) ** 2) / 0.001)
        noise = np.random.randn(length).astype(np.float32) * 0.05
        sig[i] = qrs + noise
    return sig


def run_inference(sess, signal: np.ndarray, warmup: int = 3, n_runs: int = 10):
    """
    Run inference and return predictions + latency stats.

    Returns:
        probs     : (5,) float32
        latency   : dict with timing stats (ms)
    """
    inp = signal[np.newaxis, :, :].astype(np.float32)  # (1, C, L)
    input_name = sess.get_inputs()[0].name

    # Warmup
    for _ in range(warmup):
        sess.run(None, {input_name: inp})

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        outputs = sess.run(None, {input_name: inp})
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    logits = outputs[0][0]
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    times = np.array(times)
    latency = {
        "mean_ms":   float(times.mean()),
        "p50_ms":    float(np.median(times)),
        "p95_ms":    float(np.percentile(times, 95)),
        "min_ms":    float(times.min()),
    }
    return probs, latency


def display_results(probs: np.ndarray, latency: dict, threshold: float = 0.5):
    """Pretty-print inference results."""
    print("┌─────────────────────────────────────────────────────┐")
    print("│                 DIAGNOSTIC RESULTS                  │")
    print("├────────┬──────────────────────────────┬─────────────┤")
    print("│ Class  │ Description                  │ Confidence  │")
    print("├────────┼──────────────────────────────┼─────────────┤")

    detected = []
    for cls, prob in zip(SUPERCLASSES, probs):
        bar_len = int(prob * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        flag = " ◄" if prob >= threshold else ""
        desc = DESCRIPTIONS[cls]
        print(f"│ {cls:<6} │ {desc:<28} │ {prob:.4f} {flag:<3}│")
        if prob >= threshold:
            detected.append(cls)

    print("├────────┴──────────────────────────────┴─────────────┤")
    if detected:
        print(f"│  ⚠️  Detected: {', '.join(detected):<37}│")
    else:
        print(f"│  ℹ️  No abnormality detected (all below threshold)   │")
    print("└─────────────────────────────────────────────────────┘")

    print("\n┌─────────────────────────────┐")
    print("│     INFERENCE LATENCY       │")
    print("├─────────────────────────────┤")
    print(f"│  Mean  : {latency['mean_ms']:>8.2f} ms         │")
    print(f"│  P50   : {latency['p50_ms']:>8.2f} ms         │")
    print(f"│  P95   : {latency['p95_ms']:>8.2f} ms         │")
    print(f"│  Min   : {latency['min_ms']:>8.2f} ms         │")
    target_ok = latency["p95_ms"] < 200
    status = "✅ PASS (<200ms)" if target_ok else "❌ FAIL (>200ms)"
    print(f"│  Edge target: {status:<16}│")
    print("└─────────────────────────────┘")


def main():
    parser = argparse.ArgumentParser(description="PTB-XL Edge ECG Inference — Raspberry Pi 5")
    parser.add_argument("--model",     required=True, help="Path to ONNX model file")
    parser.add_argument("--record",    default=None,  help="Path to WFDB record (without extension)")
    parser.add_argument("--demo",      action="store_true", help="Use synthetic demo signal")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--warmup",    type=int,   default=3)
    parser.add_argument("--runs",      type=int,   default=10)
    args = parser.parse_args()

    print(BANNER)

    # Load model
    sess = load_onnx_session(args.model)

    # Load signal
    if args.demo or args.record is None:
        print("🎭 Using synthetic demo signal (12-lead, 10s)\n")
        signal = make_demo_signal()
    else:
        print(f"📂 Loading ECG record: {args.record}")
        signal = load_ecg_record(args.record)
        print(f"   Shape: {signal.shape}\n")

    # Run inference
    print(f"⚡ Running inference ({args.warmup} warmup + {args.runs} timed runs)...\n")
    probs, latency = run_inference(sess, signal, args.warmup, args.runs)

    # Display results
    display_results(probs, latency, args.threshold)

    print("\n✅ Inference complete.")


if __name__ == "__main__":
    main()