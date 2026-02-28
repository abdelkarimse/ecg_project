"""
src/dataset.py
==============
PTB-XL dataset loading with stratified multi-label splits.

PTB-XL provides:
  - ptbxl_database.csv  : metadata + diagnostic labels
  - records100/         : 100Hz WFDB records
  - records500/         : 500Hz WFDB records
  - scp_statements.csv  : SCP code → superclass mapping

Superclasses: NORM, MI, STTC, CD, HYP
"""

import os
import ast
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
import torch

from src.preprocessing import ECGPreprocessor
from src.augmentation import ECGAugmenter

# ── Constants ─────────────────────────────────────────────────────────────────

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
RANDOM_SEED = 42


# ── Label Processing ──────────────────────────────────────────────────────────

def load_scp_mapping(ptbxl_path: str) -> Dict[str, str]:
    """Returns {scp_code: superclass} mapping."""
    scp_df = pd.read_csv(os.path.join(ptbxl_path, "scp_statements.csv"), index_col=0)
    mapping = {}
    for code, row in scp_df.iterrows():
        sc = str(row.get("diagnostic_class", "")).strip().upper()
        if sc in SUPERCLASSES:
            mapping[str(code).strip()] = sc
    return mapping


def parse_labels(scp_codes_str: str, scp_mapping: Dict[str, str], min_confidence: float = 50.0) -> List[str]:
    """
    Parse SCP code string → list of superclass labels.
    Only include codes with confidence ≥ min_confidence.
    """
    try:
        codes = ast.literal_eval(scp_codes_str)
    except Exception:
        return []

    labels = set()
    for code, conf in codes.items():
        if float(conf) >= min_confidence:
            sc = scp_mapping.get(str(code).strip())
            if sc:
                labels.add(sc)
    return sorted(labels)


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_ptbxl_metadata(ptbxl_path: str, sampling_rate: int = 100) -> pd.DataFrame:
    """Load and preprocess the metadata CSV."""
    df = pd.read_csv(os.path.join(ptbxl_path, "ptbxl_database.csv"), index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    # Build filename column
    rate_folder = "records100" if sampling_rate == 100 else "records500"
    if sampling_rate == 100:
        df["filename"] = df["filename_lr"]
    else:
        df["filename"] = df["filename_hr"]

    return df


def build_label_matrix(
    df: pd.DataFrame,
    ptbxl_path: str,
    min_confidence: float = 50.0,
) -> Tuple[pd.DataFrame, MultiLabelBinarizer]:
    """Compute multi-label binary matrix for the 5 superclasses."""
    scp_mapping = load_scp_mapping(ptbxl_path)
    df = df.copy()
    df["label_list"] = df["scp_codes"].apply(
        lambda codes: sorted({
            scp_mapping[k]
            for k, v in codes.items()
            if float(v) >= min_confidence and k in scp_mapping
        })
    )
    # Drop samples with no valid label
    df = df[df["label_list"].map(len) > 0].copy()

    mlb = MultiLabelBinarizer(classes=SUPERCLASSES)
    label_matrix = mlb.fit_transform(df["label_list"])
    df[[f"lbl_{c}" for c in SUPERCLASSES]] = label_matrix

    return df, mlb


# ── Stratified Split ──────────────────────────────────────────────────────────

def _make_combined_key(df: pd.DataFrame) -> pd.Series:
    """Encode multi-label vector as string for stratification."""
    cols = [f"lbl_{c}" for c in SUPERCLASSES]
    return df[cols].astype(str).agg("".join, axis=1)


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Produce stratified train/val/test splits.
    PTB-XL recommends using the official `strat_fold` column for reproducibility.
    We respect that if available, otherwise fall back to sklearn splitting.
    """
    if "strat_fold" in df.columns:
        # PTB-XL official: folds 1-8 = train, 9 = val, 10 = test
        train_df = df[df["strat_fold"] <= 8].copy()
        val_df   = df[df["strat_fold"] == 9].copy()
        test_df  = df[df["strat_fold"] == 10].copy()
        print(f"  [Official folds] train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        return train_df, val_df, test_df

    # Fallback: stratified sklearn split
    key = _make_combined_key(df)
    train_idx, temp_idx = train_test_split(
        df.index, test_size=1 - train_ratio, stratify=key, random_state=seed
    )
    temp_df = df.loc[temp_idx]
    temp_key = _make_combined_key(temp_df)
    val_size = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_df.index, test_size=1 - val_size, stratify=temp_key, random_state=seed
    )
    print(f"  [Custom split] train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return df.loc[train_idx].copy(), df.loc[val_idx].copy(), df.loc[test_idx].copy()


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class PTBXLDataset(Dataset):
    """
    PyTorch Dataset for PTB-XL multi-label ECG classification.

    Each __getitem__ returns:
        signal : (C, L) float32 tensor
        label  : (5,)  float32 tensor (multi-hot)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        ptbxl_path: str,
        preprocessor: ECGPreprocessor,
        augmenter: Optional["ECGAugmenter"] = None,
        fs_raw: int = 100,
        cache: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.ptbxl_path = ptbxl_path
        self.preprocessor = preprocessor
        self.augmenter = augmenter
        self.fs_raw = fs_raw
        self.label_cols = [f"lbl_{c}" for c in SUPERCLASSES]
        self._cache: Dict[int, np.ndarray] = {}
        self.use_cache = cache

    def __len__(self) -> int:
        return len(self.df)

    def _load_raw(self, idx: int) -> np.ndarray:
        if self.use_cache and idx in self._cache:
            return self._cache[idx]

        row = self.df.iloc[idx]
        record_path = os.path.join(self.ptbxl_path, row["filename"])
        record = wfdb.rdrecord(record_path)
        # signal: (samples, leads) → transpose to (leads, samples)
        signal = record.p_signal.T.astype(np.float32)
        # Replace NaN/Inf
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

        if self.use_cache:
            self._cache[idx] = signal
        return signal

    def __getitem__(self, idx: int):
        raw = self._load_raw(idx)
        x = self.preprocessor(raw)          # (C, L)

        if self.augmenter is not None:
            x = self.augmenter(x)

        label = self.df.iloc[idx][self.label_cols].values.astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(label)

    @property
    def class_weights(self) -> torch.Tensor:
        """Compute positive class weights for BCEWithLogitsLoss."""
        labels = self.df[self.label_cols].values.astype(np.float32)
        pos = labels.sum(0)
        neg = len(labels) - pos
        weights = neg / (pos + 1e-8)
        return torch.tensor(weights, dtype=torch.float32)

    def summary(self) -> pd.DataFrame:
        counts = self.df[self.label_cols].sum()
        counts.index = SUPERCLASSES
        return counts.rename("count").to_frame()


# ── DataLoader Factory ────────────────────────────────────────────────────────

def build_dataloaders(
    ptbxl_path: str,
    sampling_rate: int = 100,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    aug_config: Optional[dict] = None,
    preproc_config: Optional[dict] = None,
    cache_train: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Build train/val/test DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, pos_weights
    """
    print("📦 Loading PTB-XL metadata...")
    df = load_ptbxl_metadata(ptbxl_path, sampling_rate)
    df, mlb = build_label_matrix(df, ptbxl_path)

    print("✂️  Splitting dataset...")
    train_df, val_df, test_df = stratified_split(df)

    print("🔧 Building preprocessor...")
    pc = preproc_config or {}
    preprocessor = ECGPreprocessor(
        fs_in=float(sampling_rate),
        fs_out=float(pc.get("fs_out", 100)),
        target_len=int(pc.get("target_len", 1000)),
        bandpass_low=float(pc.get("bandpass_low", 0.5)),
        bandpass_high=float(pc.get("bandpass_high", 40.0)),
        notch_freq=float(pc.get("notch_freq", 50.0)),
        normalize_method=pc.get("normalize", "zscore"),
    )

    from src.augmentation import ECGAugmenter
    augmenter = None
    if aug_config and aug_config.get("enabled", True):
        augmenter = ECGAugmenter(**{k: v for k, v in aug_config.items() if k != "enabled"})

    train_ds = PTBXLDataset(train_df, ptbxl_path, preprocessor, augmenter, sampling_rate, cache_train)
    val_ds   = PTBXLDataset(val_df,   ptbxl_path, preprocessor, None,      sampling_rate, False)
    test_ds  = PTBXLDataset(test_df,  ptbxl_path, preprocessor, None,      sampling_rate, False)

    pos_weights = train_ds.class_weights

    print(f"\n📊 Dataset Summary:")
    print(f"  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}  |  Test: {len(test_ds):,}")
    print("\n  Class distribution (train):")
    print(train_ds.summary().to_string())
    print()

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, pos_weights


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()
    train_l, val_l, test_l, pw = build_dataloaders(args.data_dir)
    x, y = next(iter(train_l))
    print(f"Batch signal shape: {x.shape}")
    print(f"Batch label shape:  {y.shape}")
    print(f"Pos weights:        {pw}")