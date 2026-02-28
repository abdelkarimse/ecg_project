"""
src/preprocessing.py
====================
ECG signal preprocessing pipeline for PTB-XL dataset.

Steps:
  1. Bandpass filter (0.5 – 40 Hz)
  2. Notch filter (50/60 Hz powerline)
  3. Resample to target length
  4. Normalization (z-score or min-max per lead)
  5. Lead selection
"""

import numpy as np
from scipy.signal import butter, sosfilt, iirnotch, filtfilt, resample_poly
from scipy.interpolate import interp1d
from typing import Literal, Optional
import warnings


# ── Filter Design ─────────────────────────────────────────────────────────────

def _butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype="band", output="sos")
    return sos


def _notch_filter_coeffs(notch_freq: float, fs: float, quality: float = 30.0):
    b, a = iirnotch(notch_freq / (fs / 2), quality)
    return b, a


# ── Core Preprocessing ────────────────────────────────────────────────────────

def bandpass_filter(signal: np.ndarray, fs: float, low: float = 0.5, high: float = 40.0) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter.
    signal shape: (leads, samples) or (samples,)
    """
    sos = _butter_bandpass(low, high, fs)
    if signal.ndim == 1:
        return sosfilt(sos, signal).astype(np.float32)
    return np.stack([sosfilt(sos, lead) for lead in signal], axis=0).astype(np.float32)


def notch_filter(signal: np.ndarray, fs: float, freq: float = 50.0) -> np.ndarray:
    """Apply zero-phase notch filter for powerline interference."""
    b, a = _notch_filter_coeffs(freq, fs)
    if signal.ndim == 1:
        return filtfilt(b, a, signal).astype(np.float32)
    return np.stack([filtfilt(b, a, lead) for lead in signal], axis=0).astype(np.float32)


def resample_signal(signal: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """Resample using rational fraction (high quality).
    signal shape: (leads, samples)
    """
    if orig_fs == target_fs:
        return signal
    from math import gcd
    g = gcd(int(target_fs), int(orig_fs))
    up = int(target_fs) // g
    down = int(orig_fs) // g
    if signal.ndim == 1:
        return resample_poly(signal, up, down).astype(np.float32)
    return np.stack([resample_poly(lead, up, down) for lead in signal], axis=0).astype(np.float32)


def pad_or_crop(signal: np.ndarray, target_len: int) -> np.ndarray:
    """Ensure fixed-length output (pad with zeros or center-crop).
    signal shape: (leads, samples)
    """
    n = signal.shape[-1]
    if n == target_len:
        return signal
    if n > target_len:
        # Center crop
        start = (n - target_len) // 2
        if signal.ndim == 1:
            return signal[start:start + target_len]
        return signal[:, start:start + target_len]
    # Zero-pad
    pad = target_len - n
    left = pad // 2
    right = pad - left
    if signal.ndim == 1:
        return np.pad(signal, (left, right), mode="constant")
    return np.pad(signal, ((0, 0), (left, right)), mode="constant")


def normalize(
    signal: np.ndarray,
    method: Literal["zscore", "minmax", "none"] = "zscore",
    eps: float = 1e-8,
) -> np.ndarray:
    """Per-lead normalization.
    signal shape: (leads, samples)
    """
    if method == "none":
        return signal.astype(np.float32)

    out = signal.copy().astype(np.float32)
    for i in range(out.shape[0]):
        lead = out[i]
        if method == "zscore":
            mu, sigma = lead.mean(), lead.std()
            out[i] = (lead - mu) / (sigma + eps)
        elif method == "minmax":
            lo, hi = lead.min(), lead.max()
            out[i] = (lead - lo) / (hi - lo + eps)
    return out


# ── Lead Selection ────────────────────────────────────────────────────────────

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_SETS = {
    "all":        list(range(12)),
    "limb":       [0, 1, 2, 3, 4, 5],
    "precordial": [6, 7, 8, 9, 10, 11],
}


def select_leads(signal: np.ndarray, leads: str = "all") -> np.ndarray:
    idx = LEAD_SETS.get(leads, list(range(12)))
    return signal[idx]


# ── Full Pipeline ─────────────────────────────────────────────────────────────

class ECGPreprocessor:
    """
    End-to-end preprocessing for PTB-XL ECG signals.

    Usage:
        prep = ECGPreprocessor(fs_in=500, fs_out=100, target_len=1000)
        x = prep(raw_signal)   # shape: (12, 1000) float32
    """

    def __init__(
        self,
        fs_in: float = 500.0,
        fs_out: float = 100.0,
        target_len: int = 1000,
        bandpass_low: float = 0.5,
        bandpass_high: float = 40.0,
        notch_freq: Optional[float] = 50.0,
        normalize_method: Literal["zscore", "minmax", "none"] = "zscore",
        leads: str = "all",
    ):
        self.fs_in = fs_in
        self.fs_out = fs_out
        self.target_len = target_len
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.notch_freq = notch_freq
        self.normalize_method = normalize_method
        self.leads = leads

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: (12, N) float array in mV
        Returns:
            (C, target_len) float32 — ready for the model
        """
        # 1. Lead selection
        x = select_leads(signal, self.leads)

        # 2. Bandpass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = bandpass_filter(x, self.fs_in, self.bandpass_low, self.bandpass_high)

        # 3. Notch
        if self.notch_freq is not None:
            x = notch_filter(x, self.fs_in, self.notch_freq)

        # 4. Resample
        x = resample_signal(x, self.fs_in, self.fs_out)

        # 5. Pad / crop
        x = pad_or_crop(x, self.target_len)

        # 6. Normalize
        x = normalize(x, self.normalize_method)

        return x.astype(np.float32)

    def __repr__(self):
        return (
            f"ECGPreprocessor(fs_in={self.fs_in}, fs_out={self.fs_out}, "
            f"len={self.target_len}, norm={self.normalize_method})"
        )