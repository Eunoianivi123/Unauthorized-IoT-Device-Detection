#!/usr/bin/env python3
"""Verify pcap_csv_fast.py against pcap_csv.py (feature values + model decisions).

Compares the fast extractor against the reference feature CSV produced by the
reference extractor on the SAME pcap, then scores both with the deployed
autoencoder and checks that the per-packet anomaly flags are unchanged.

Usage (inside the pipeline image):
  python3 verify_fast_extractor.py \
      --pcap output_overhead/device1_attack_capture.pcap \
      --ref  output_overhead/device1_attack_capture_features.csv \
      --model /model/autoencoder_allinone.keras \
      --scaler /model/scaler_allinone.pkl \
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
from pcap_csv import feature_columns                      # noqa: E402
from pcap_csv_fast import extract_features_from_pcap       # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--pcap", default="output_overhead/device1_attack_capture.pcap")
parser.add_argument("--ref", default="output_overhead/device1_attack_capture_features.csv")
parser.add_argument("--model", default="/model/autoencoder_allinone.keras")
parser.add_argument("--scaler", default="/model/scaler_allinone.pkl")
parser.add_argument("--threshold", type=float, default=8.555540)
parser.add_argument("--out-csv", default="/model/verify_fast_extractor.csv")
parser.add_argument("--skip-model", action="store_true")
args = parser.parse_args()

out_csv = args.out_csv
print("=" * 70)
print("  FAST EXTRACTOR VERIFICATION (pcap_csv_fast.py vs pcap_csv.py)")
print("=" * 70)

# ── 1. run the fast extractor (timed, monotonic clock) ─────────────────────
t0 = time.monotonic()
df_fast = extract_features_from_pcap(args.pcap)
t_fast = time.monotonic() - t0
print(f"\n[1] fast extractor : {len(df_fast)} rows in {t_fast:.1f}s "
      f"({len(df_fast)/max(t_fast,1e-9):.0f} rows/s)")

df_ref = pd.read_csv(args.ref)
print(f"[2] reference CSV  : {len(df_ref)} rows from {args.ref}")

if len(df_fast) != len(df_ref):
    print(f"[FAIL] row count mismatch: {len(df_fast)} vs {len(df_ref)}")
    sys.exit(1)

# ── 2. metadata columns must match exactly ─────────────────────────────────
meta_ok = True
for col in ("timestamp", "src_ip", "dst_ip"):
    a = df_fast[col].to_numpy()
    b = df_ref[col].to_numpy()
    same = (a == b).all()
    meta_ok &= bool(same)
    print(f"    {col:10s} exact match: {same}")
if not meta_ok:
    print("[FAIL] metadata columns differ")
    sys.exit(1)

# ── 3. feature-by-feature comparison ───────────────────────────────────────
print("\n[3] per-feature difference (fast vs reference):")
rows = []
worst_rel = 0.0
worst_col = None
max_abs_overall = 0.0
exact_cols = 0
for col in feature_columns:
    a = df_fast[col].to_numpy(dtype=np.float64)
    b = df_ref[col].to_numpy(dtype=np.float64)
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(b), 1e-12)
    rel = diff / denom
    max_abs = float(diff.max()) if diff.size else 0.0
    max_rel = float(rel.max()) if rel.size else 0.0
    exact = bool((a == b).all())
    exact_cols += int(exact)
    max_abs_overall = max(max_abs_overall, max_abs)
    if max_rel > worst_rel:
        worst_rel, worst_col = max_rel, col
    rows.append({"feature": col, "max_abs_diff": max_abs, "max_rel_diff": max_rel,
                 "bit_identical": exact})

print(f"    columns bit-identical : {exact_cols}/115")
print(f"    worst abs diff         : {max_abs_overall:.3e}")
print(f"    worst rel diff         : {worst_rel:.3e}  (column: {worst_col})")

# ── 4. model decisions must be unchanged ───────────────────────────────────
flags_equal = None
mse_max_diff = None
if not args.skip_model:
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

    mse_fast, flags_fast = score(df_fast)
    mse_ref, flags_ref = score(df_ref)
    mse_max_diff = float(np.max(np.abs(mse_fast - mse_ref)))
    flags_equal = bool((flags_fast == flags_ref).all())
    n_diff = int((flags_fast != flags_ref).sum())
    print("\n[4] model scoring (threshold %.6f):" % args.threshold)
    print(f"    anomalies fast/ref      : {int(flags_fast.sum())} / {int(flags_ref.sum())}")
    print(f"    max |MSE_fast - MSE_ref|: {mse_max_diff:.3e}")
    print(f"    anomaly flags identical : {flags_equal}  (differing rows: {n_diff})")

pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f"\n[5] per-feature diff table -> {out_csv}")

verdict = (exact_cols == 115) or (flags_equal is True)
print("\n" + "=" * 70)
print(f"  VERDICT: {'PASS' if verdict else 'FAIL'}  "
      f"(bit-identical columns: {exact_cols}/115, "
      f"anomaly flags identical: {flags_equal})")
print("=" * 70)

summary = {
    "pcap": args.pcap, "rows": len(df_fast),
    "fast_seconds": round(t_fast, 2),
    "fast_rows_per_s": round(len(df_fast) / max(t_fast, 1e-9), 1),
    "columns_bit_identical": exact_cols,
    "max_abs_diff": max_abs_overall,
    "max_rel_diff": worst_rel,
    "worst_column": worst_col,
    "mse_max_diff": mse_max_diff,
    "anomaly_flags_identical": flags_equal,
}
with open(os.path.splitext(out_csv)[0] + ".json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
