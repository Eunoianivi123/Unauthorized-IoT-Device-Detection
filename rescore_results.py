#!/usr/bin/env python3
"""Re-score all devices with the all-in-one model using benign-only
median + 2.5*MAD thresholds and rewrite the results CSVs.

This updates the `anomaly` and `threshold` columns in
output/device{N}_{mode}_results.csv (mse stays identical) so the
identifier's anomaly watcher uses correct thresholds.

Run inside the nbaiot-pipeline image:
  docker run --rm --entrypoint python3 -v .../model:/model:ro -v .../output:/output:ro \
      -v .../rescore_results.py:/rescore_results.py:ro \
      -e MODEL_DIR=/model -e OUT_DIR=/output nbaiot-pipeline:latest /rescore_results.py
"""
import os

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

FEATURE_NAMES = [
    "MI_dir_L5_weight","MI_dir_L5_mean","MI_dir_L5_variance",
    "MI_dir_L3_weight","MI_dir_L3_mean","MI_dir_L3_variance",
    "MI_dir_L1_weight","MI_dir_L1_mean","MI_dir_L1_variance",
    "MI_dir_L0.1_weight","MI_dir_L0.1_mean","MI_dir_L0.1_variance",
    "MI_dir_L0.01_weight","MI_dir_L0.01_mean","MI_dir_L0.01_variance",
    "H_L5_weight","H_L5_mean","H_L5_variance",
    "H_L3_weight","H_L3_mean","H_L3_variance",
    "H_L1_weight","H_L1_mean","H_L1_variance",
    "H_L0.1_weight","H_L0.1_mean","H_L0.1_variance",
    "H_L0.01_weight","H_L0.01_mean","H_L0.01_variance",
    "HH_L5_weight","HH_L5_mean","HH_L5_std","HH_L5_magnitude","HH_L5_radius","HH_L5_covariance","HH_L5_pcc",
    "HH_L3_weight","HH_L3_mean","HH_L3_std","HH_L3_magnitude","HH_L3_radius","HH_L3_covariance","HH_L3_pcc",
    "HH_L1_weight","HH_L1_mean","HH_L1_std","HH_L1_magnitude","HH_L1_radius","HH_L1_covariance","HH_L1_pcc",
    "HH_L0.1_weight","HH_L0.1_mean","HH_L0.1_std","HH_L0.1_magnitude","HH_L0.1_radius","HH_L0.1_covariance","HH_L0.1_pcc",
    "HH_L0.01_weight","HH_L0.01_mean","HH_L0.01_std","HH_L0.01_magnitude","HH_L0.01_radius","HH_L0.01_covariance","HH_L0.01_pcc",
    "HH_jit_L5_weight","HH_jit_L5_mean","HH_jit_L5_variance",
    "HH_jit_L3_weight","HH_jit_L3_mean","HH_jit_L3_variance",
    "HH_jit_L1_weight","HH_jit_L1_mean","HH_jit_L1_variance",
    "HH_jit_L0.1_weight","HH_jit_L0.1_mean","HH_jit_L0.1_variance",
    "HH_jit_L0.01_weight","HH_jit_L0.01_mean","HH_jit_L0.01_variance",
    "HpHp_L5_weight","HpHp_L5_mean","HpHp_L5_std","HpHp_L5_magnitude","HpHp_L5_radius","HpHp_L5_covariance","HpHp_L5_pcc",
    "HpHp_L3_weight","HpHp_L3_mean","HpHp_L3_std","HpHp_L3_magnitude","HpHp_L3_radius","HpHp_L3_covariance","HpHp_L3_pcc",
    "HpHp_L1_weight","HpHp_L1_mean","HpHp_L1_std","HpHp_L1_magnitude","HpHp_L1_radius","HpHp_L1_covariance","HpHp_L1_pcc",
    "HpHp_L0.1_weight","HpHp_L0.1_mean","HpHp_L0.1_std","HpHp_L0.1_magnitude","HpHp_L0.1_radius","HpHp_L0.1_covariance","HpHp_L0.1_pcc",
    "HpHp_L0.01_weight","HpHp_L0.01_mean","HpHp_L0.01_std","HpHp_L0.01_magnitude","HpHp_L0.01_radius","HpHp_L0.01_covariance","HpHp_L0.01_pcc",
]

MODEL_DIR = os.environ.get("MODEL_DIR", "model")
OUT_DIR = os.environ.get("OUT_DIR", "output")

model = keras.models.load_model(f"{MODEL_DIR}/autoencoder_allinone.keras")
scaler = joblib.load(f"{MODEL_DIR}/scaler_allinone.pkl")


def score(X):
    Xs = scaler.transform(X)
    Xr = model.predict(Xs, batch_size=256, verbose=0)
    return np.mean(np.power(Xs - Xr, 2), axis=1)


def write_results(device, mode, mse, threshold):
    df = pd.read_csv(os.path.join(OUT_DIR, f"device{device}_{mode}_capture_features.csv"))
    out = df[FEATURE_NAMES].copy()
    out["mse"] = mse
    out["anomaly"] = (mse > threshold).astype(int)
    out["threshold"] = threshold
    path = os.path.join(OUT_DIR, f"device{device}_{mode}_results.csv")
    out.to_csv(path, index=False)
    print(f"  device{device}_{mode}: threshold={threshold:.4f} anomaly_rate={out['anomaly'].mean()*100:.1f}% -> {path}")


for d in range(1, 10):
    ben = pd.read_csv(os.path.join(OUT_DIR, f"device{d}_normal_capture_features.csv"))
    Xb = ben[FEATURE_NAMES].values.astype(np.float32)
    Xb = np.nan_to_num(Xb, nan=0.0, posinf=0.0, neginf=0.0)
    ben_mse = score(Xb)
    med = float(np.median(ben_mse))
    mad = float(np.median(np.abs(ben_mse - med)))
    thr = med + 2.5 * mad

    atk = pd.read_csv(os.path.join(OUT_DIR, f"device{d}_attack_capture_features.csv"))
    Xa = atk[FEATURE_NAMES].values.astype(np.float32)
    Xa = np.nan_to_num(Xa, nan=0.0, posinf=0.0, neginf=0.0)
    atk_mse = score(Xa)

    write_results(d, "normal", ben_mse, thr)
    write_results(d, "attack", atk_mse, thr)
