#!/usr/bin/env python3
"""Reference vs fast extractor on the SAME pcap, compared feature-by-feature and
through the deployed model (per-packet anomaly flags must be identical).

Runs both extractors in one process so the comparison is apples-to-apples.

Usage (pipeline image):
  python3 verify_fast_vs_reference.py \
      --pcaps output/device1_normal_capture.pcap output/device1_attack_hard_capture.pcap \
      --model /model/autoencoder_allinone.keras --scaler /model/scaler_allinone.pkl \
      --threshold 8.555540
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcap_csv import feature_columns                 # noqa: E402
import pcap_csv                                     # noqa: E402
import pcap_csv_fast                                # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--pcaps", nargs="+", required=True)
parser.add_argument("--model", default="/model/autoencoder_allinone.keras")
parser.add_argument("--scaler", default="/model/scaler_allinone.pkl")
parser.add_argument("--threshold", type=float, default=8.555540)
parser.add_argument("--out", default="output/verify_fast_vs_reference.csv")
args = parser.parse_args()

import joblib
from tensorflow import keras

model = keras.models.load_model(args.model)
scaler = joblib.load(args.scaler)


def score(frame):
    X = frame[feature_columns].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = scaler.transform(X)
    rec = model.predict(Xs, batch_size=256, verbose=0)
    mse = np.mean(np.power(Xs - rec, 2), axis=1)
    return mse, mse > args.threshold


rows = []
for pcap in args.pcaps:
    print(f"\n=== {pcap} ===", flush=True)
    t0 = time.monotonic()
    df_ref = pcap_csv.extract_features_from_pcap(pcap)
    t_ref = time.monotonic() - t0

    t0 = time.monotonic()
    df_fast = pcap_csv_fast.extract_features_from_pcap(pcap)
    t_fast = time.monotonic() - t0

    if len(df_ref) != len(df_fast):
        print(f"[FAIL] row mismatch {len(df_ref)} vs {len(df_fast)}")
        continue

    worst_abs, worst_rel, worst_col, exact_cols = 0.0, 0.0, None, 0
    for col in feature_columns:
        a = df_fast[col].to_numpy(dtype=np.float64)
        b = df_ref[col].to_numpy(dtype=np.float64)
        d = np.abs(a - b)
        r = d / np.maximum(np.abs(b), 1e-12)
        exact_cols += int((a == b).all())
        if d.size:
            if d.max() > worst_abs:
                worst_abs = float(d.max())
            if r.max() > worst_rel:
                worst_rel, worst_col = float(r.max()), col

    mse_ref, fl_ref = score(df_ref)
    mse_fast, fl_fast = score(df_fast)
    mse_diff = float(np.max(np.abs(mse_fast - mse_ref)))
    same = bool((fl_ref == fl_fast).all())

    print(f"    reference : {t_ref:7.2f}s   fast : {t_fast:7.2f}s   speedup {t_ref/max(t_fast,1e-9):5.1f}x")
    print(f"    bit-identical columns : {exact_cols}/115")
    print(f"    worst abs/rel diff    : {worst_abs:.3e} / {worst_rel:.3e}  ({worst_col})")
    print(f"    max |MSE diff|        : {mse_diff:.3e}")
    print(f"    anomaly flags identical: {same}  (ref anomalies {int(fl_ref.sum())}/{len(fl_ref)})")

    rows.append({
        "pcap": pcap, "rows": len(df_ref),
        "reference_seconds": round(t_ref, 3), "fast_seconds": round(t_fast, 3),
        "speedup": round(t_ref / max(t_fast, 1e-9), 1),
        "bit_identical_columns": exact_cols,
        "max_abs_diff": worst_abs, "max_rel_diff": worst_rel, "worst_column": worst_col,
        "mse_max_diff": mse_diff, "anomaly_flags_identical": same,
    })
    pd.DataFrame(rows).to_csv(args.out, index=False)

print("\n" + "=" * 96)
print(pd.DataFrame(rows).to_string(index=False))
print(f"\n-> {args.out}")
print("=" * 96)
