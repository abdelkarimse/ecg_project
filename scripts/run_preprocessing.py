"""
scripts/run_preprocessing.py
=============================
Step 1: Preprocess PTB-XL signals and save as .npy arrays.

Usage:
    python scripts/run_preprocessing.py --data_dir /path/to/ptb-xl --output_dir data/processed
"""

import os
import argparse
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import ECGPreprocessor
from src.dataset import load_ptbxl_metadata, build_label_matrix, stratified_split, SUPERCLASSES
from src.utils import set_seed, load_config, get_logger

logger = get_logger(__name__)


def preprocess_and_save(
    df: pd.DataFrame,
    split_name: str,
    ptbxl_path: str,
    output_dir: str,
    preprocessor: ECGPreprocessor,
    fs_raw: int = 100,
):
    """Save preprocessed signals as a single .npz file."""
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    label_cols = [f"lbl_{c}" for c in SUPERCLASSES]
    n = len(df)
    C = 12 if preprocessor.leads == "all" else 6
    L = preprocessor.target_len

    signals = np.zeros((n, C, L), dtype=np.float32)
    labels  = np.zeros((n, 5),    dtype=np.float32)

    failed = 0
    for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=n, desc=f"  {split_name}")):
        try:
            record_path = os.path.join(ptbxl_path, row["filename"])
            record = wfdb.rdrecord(record_path)
            raw = record.p_signal.T.astype(np.float32)
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            signals[i] = preprocessor(raw)
            labels[i]  = row[label_cols].values.astype(np.float32)
        except Exception as e:
            logger.warning(f"Failed record {idx}: {e}")
            failed += 1

    out_path = os.path.join(split_dir, "data.npz")
    np.savez_compressed(out_path, signals=signals, labels=labels)
    logger.info(f"  Saved {n - failed}/{n} samples → {out_path}")
    if failed:
        logger.warning(f"  {failed} records failed to load")


def main():
    parser = argparse.ArgumentParser(description="PTB-XL preprocessing")
    parser.add_argument("--data_dir",   required=True,  help="Path to ptb-xl root folder")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--fs",         type=int, default=100, help="Sampling rate (100 or 500)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["random_seed"])
    pc = cfg["preprocessing"]

    preprocessor = ECGPreprocessor(
        fs_in=float(args.fs),
        fs_out=float(pc.get("target_fs", 100)),
        target_len=int(pc.get("target_length", 1000)),
        bandpass_low=float(pc.get("bandpass_low", 0.5)),
        bandpass_high=float(pc.get("bandpass_high", 40.0)),
        notch_freq=float(pc.get("notch_freq", 50.0)),
        normalize_method=pc.get("normalize", "zscore"),
    )

    logger.info("📥 Loading metadata...")
    df = load_ptbxl_metadata(args.data_dir, args.fs)
    df, mlb = build_label_matrix(df, args.data_dir)

    logger.info("✂️  Splitting dataset...")
    train_df, val_df, test_df = stratified_split(df)

    logger.info(f"💾 Saving preprocessed splits to {args.output_dir}")
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        preprocess_and_save(split_df, split_name, args.data_dir, args.output_dir, preprocessor, args.fs)

    logger.info("✅ Preprocessing complete!")


if __name__ == "__main__":
    main()