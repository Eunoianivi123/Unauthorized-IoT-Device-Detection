#!/usr/bin/env python3
"""Score N-BaIoT features with the all-in-one autoencoder model.

Usage:
  python testmodel.py --features capture_features.csv \
                      --model autoencoder_allinone.keras \
                      --scaler scaler_allinone.pkl \
                      --threshold <T> --device-name <NAME> --output results.csv
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

# ── Arguments ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--features",    default="attack_capture_features.csv", help="Extracted features CSV")
parser.add_argument("--model",       default="autoencoder_allinone.keras",  help="Trained global autoencoder")
parser.add_argument("--scaler",      default="scaler_allinone.pkl",         help="Fitted MinMaxScaler")
parser.add_argument("--threshold",   type=float, default=None,              help="MSE threshold (auto if unset)")
parser.add_argument("--device-name", default="Unknown Device",              help="Device name for display")
parser.add_argument("--output",      default="results.csv",                 help="Results CSV output path")
args = parser.parse_args()

# ── Exact 115 N-BaIoT feature names in correct order ──────
FEATURE_NAMES = [
    # MI_dir (15)
    "MI_dir_L5_weight","MI_dir_L5_mean","MI_dir_L5_variance",
    "MI_dir_L3_weight","MI_dir_L3_mean","MI_dir_L3_variance",
    "MI_dir_L1_weight","MI_dir_L1_mean","MI_dir_L1_variance",
    "MI_dir_L0.1_weight","MI_dir_L0.1_mean","MI_dir_L0.1_variance",
    "MI_dir_L0.01_weight","MI_dir_L0.01_mean","MI_dir_L0.01_variance",
    # H (15)
    "H_L5_weight","H_L5_mean","H_L5_variance",
    "H_L3_weight","H_L3_mean","H_L3_variance",
    "H_L1_weight","H_L1_mean","H_L1_variance",
    "H_L0.1_weight","H_L0.1_mean","H_L0.1_variance",
    "H_L0.01_weight","H_L0.01_mean","H_L0.01_variance",
    # HH (35)
    "HH_L5_weight","HH_L5_mean","HH_L5_std","HH_L5_magnitude","HH_L5_radius","HH_L5_covariance","HH_L5_pcc",
    "HH_L3_weight","HH_L3_mean","HH_L3_std","HH_L3_magnitude","HH_L3_radius","HH_L3_covariance","HH_L3_pcc",
    "HH_L1_weight","HH_L1_mean","HH_L1_std","HH_L1_magnitude","HH_L1_radius","HH_L1_covariance","HH_L1_pcc",
    "HH_L0.1_weight","HH_L0.1_mean","HH_L0.1_std","HH_L0.1_magnitude","HH_L0.1_radius","HH_L0.1_covariance","HH_L0.1_pcc",
    "HH_L0.01_weight","HH_L0.01_mean","HH_L0.01_std","HH_L0.01_magnitude","HH_L0.01_radius","HH_L0.01_covariance","HH_L0.01_pcc",
    # HH_jit (15)
    "HH_jit_L5_weight","HH_jit_L5_mean","HH_jit_L5_variance",
    "HH_jit_L3_weight","HH_jit_L3_mean","HH_jit_L3_variance",
    "HH_jit_L1_weight","HH_jit_L1_mean","HH_jit_L1_variance",
    "HH_jit_L0.1_weight","HH_jit_L0.1_mean","HH_jit_L0.1_variance",
    "HH_jit_L0.01_weight","HH_jit_L0.01_mean","HH_jit_L0.01_variance",
    # HpHp (35)
    "HpHp_L5_weight","HpHp_L5_mean","HpHp_L5_std","HpHp_L5_magnitude","HpHp_L5_radius","HpHp_L5_covariance","HpHp_L5_pcc",
    "HpHp_L3_weight","HpHp_L3_mean","HpHp_L3_std","HpHp_L3_magnitude","HpHp_L3_radius","HpHp_L3_covariance","HpHp_L3_pcc",
    "HpHp_L1_weight","HpHp_L1_mean","HpHp_L1_std","HpHp_L1_magnitude","HpHp_L1_radius","HpHp_L1_covariance","HpHp_L1_pcc",
    "HpHp_L0.1_weight","HpHp_L0.1_mean","HpHp_L0.1_std","HpHp_L0.1_magnitude","HpHp_L0.1_radius","HpHp_L0.1_covariance","HpHp_L0.1_pcc",
    "HpHp_L0.01_weight","HpHp_L0.01_mean","HpHp_L0.01_std","HpHp_L0.01_magnitude","HpHp_L0.01_radius","HpHp_L0.01_covariance","HpHp_L0.01_pcc",
]

assert len(FEATURE_NAMES) == 115

print("=" * 55)
print(f"  N-BaIoT Model Tester -- {args.device_name}")
print("=" * 55)

# ── Step 1: Load features CSV ─────────────────────────────
print(f"\n[1/4] Loading features from {args.features}...")
if not os.path.exists(args.features):
    print(f"[ERROR] File not found: {args.features}")
    sys.exit(1)

df = pd.read_csv(args.features)
print(f"      Rows loaded    : {len(df)}")
print(f"      Columns found  : {len(df.columns)}")

missing = [f for f in FEATURE_NAMES if f not in df.columns]
if missing:
    print(f"[ERROR] Missing {len(missing)} feature columns:")
    for m in missing[:10]:
        print(f"        - {m}")
    sys.exit(1)

X = df[FEATURE_NAMES].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
print(f"      Feature matrix : {X.shape}  (all 115 features found)")

# ── Step 2: Scale features ────────────────────────────────
print(f"\n[2/4] Scaling features using {args.scaler}...")
if not os.path.exists(args.scaler):
    print(f"[ERROR] Scaler not found: {args.scaler}")
    sys.exit(1)

scaler = joblib.load(args.scaler)
X_scaled = scaler.transform(X)
print(f"      Scaler type  : {type(scaler).__name__}")
print(f"      Scaled range : [{X_scaled.min():.4f}, {X_scaled.max():.4f}]")

# ── Step 3: Load model + reconstruct ──────────────────────
print(f"\n[3/4] Running autoencoder inference...")
if not os.path.exists(args.model):
    print(f"[ERROR] Model not found: {args.model}")
    sys.exit(1)

try:
    from tensorflow import keras
    model = keras.models.load_model(args.model)
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    sys.exit(1)

print(f"      Model input  : {model.input_shape}")
print(f"      Model output : {model.output_shape}")

X_reconstructed = model.predict(X_scaled, batch_size=256, verbose=0)

# MSE reconstruction error per sample
mse = np.mean(np.power(X_scaled - X_reconstructed, 2), axis=1)
print(f"      Done. {len(mse)} samples processed.")

# ── Step 4: Anomaly detection ─────────────────────────────
print(f"\n[4/4] Detecting anomalies...")

if args.threshold:
    threshold = args.threshold
    print(f"      Using provided threshold  : {threshold:.6f}")
else:
    threshold = np.mean(mse) + 2 * np.std(mse)
    print(f"      Auto threshold (mean+2std): {threshold:.6f}")

anomalies = mse > threshold
n_total = len(mse)
n_anomaly = int(anomalies.sum())
n_benign = n_total - n_anomaly
anomaly_rate = n_anomaly / n_total * 100

# ── Results ───────────────────────────────────────────────
print("\n" + "=" * 55)
print("  RESULTS")
print("=" * 55)
print(f"  Total samples  : {n_total}")
print(f"  Benign         : {n_benign}  ({100 - anomaly_rate:.1f}%)")
print(f"  Anomalous      : {n_anomaly} ({anomaly_rate:.1f}%)")
print(f"  MSE min        : {mse.min():.6f}")
print(f"  MSE max        : {mse.max():.6f}")
print(f"  MSE mean       : {mse.mean():.6f}")
print(f"  MSE std        : {mse.std():.6f}")
print(f"  Threshold      : {threshold:.6f}")
print("=" * 55)

print("\n  INTERPRETATION:")
if anomaly_rate < 5:
    print(f"  BENIGN -- Traffic looks normal to the {args.device_name} model")
elif anomaly_rate < 30:
    print("  MIXED -- Some patterns differ from training data")
else:
    print("  ANOMALOUS -- High reconstruction error detected")

# ── Save results (written straight to --output path) ──────
out_file = args.output
df_out = df[FEATURE_NAMES].copy()
df_out["mse"] = mse
df_out["anomaly"] = anomalies.astype(int)
df_out["threshold"] = threshold
df_out.to_csv(out_file, index=False)
print(f"\n  Results saved to : {out_file}")
print("=" * 55)
