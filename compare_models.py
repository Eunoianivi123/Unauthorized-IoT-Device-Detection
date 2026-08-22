#!/usr/bin/env python3
"""Compare all-in-one autoencoder vs per-device autoencoders.

Scores each device's normal + attack feature CSVs with both model families,
applies a benign-only `median + 2.5*MAD` threshold per device, and reports
AUC / accuracy / TPR / FPR per device and overall.

Run inside the nbaiot-pipeline image:
  docker run --rm -v "$PWD:/work:ro" -w /work nbaiot-pipeline:latest python3 /work/compare_models.py
"""
import glob
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


def load_features(device, mode):
    path = os.path.join(OUT_DIR, f"device{device}_{mode}_capture_features.csv")
    df = pd.read_csv(path)
    X = df[FEATURE_NAMES].values.astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def score(model, scaler, X):
    Xs = scaler.transform(X)
    Xr = model.predict(Xs, batch_size=256, verbose=0)
    return np.mean(np.power(Xs - Xr, 2), axis=1)


def median_mad_threshold(mse_benign):
    med = float(np.median(mse_benign))
    mad = float(np.median(np.abs(mse_benign - med)))
    return med + 2.5 * mad


def metrics(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else float("nan")
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    return acc, tpr, fpr, tp, fp, tn, fn


def roc_auc(y, score):
    n = len(y)
    npos = sum(1 for v in y if v == 1)
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    order = sorted(range(n), key=lambda i: score[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and score[order[j]] == score[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    rank_sum = sum(ranks[i] for i in range(n) if y[i] == 1)
    return (rank_sum - npos * (npos + 1) / 2.0) / (npos * nneg)


def fmt(v):
    return f"{v:.4f}" if v == v else "  -  "


allinone_model = keras.models.load_model(f"{MODEL_DIR}/autoencoder_allinone.keras")
allinone_scaler = joblib.load(f"{MODEL_DIR}/scaler_allinone.pkl")

# cache loaded per-device models/scalers
dev_models = {}
dev_scalers = {}

results = {"allinone": [], "single": []}
tot = {"allinone": {"y": [], "s": []}, "single": {"y": [], "s": []}}

print("=" * 100)
print("  per-device threshold = median + 2.5*MAD of that device's benign MSE (benign-only)")
print("=" * 100)
hdr = (f"  {'device':<6}{'model':<10}{'n_ben':>7}{'n_atk':>7}{'AUC':>8}{'thr':>10}"
       f"{'acc':>8}{'TPR':>8}{'FPR':>8}   TP   FP   TN   FN")
print(hdr)
print("  " + "-" * 96)

for d in range(1, 10):
    ben = load_features(d, "normal")
    atk = load_features(d, "attack")

    if d not in dev_models:
        dev_models[d] = keras.models.load_model(f"{MODEL_DIR}/autoencoder_{d}.keras")
        dev_scalers[d] = joblib.load(f"{MODEL_DIR}/scaler_{d}.pkl")

    for name, model, scaler in (
        ("allinone", allinone_model, allinone_scaler),
        ("single", dev_models[d], dev_scalers[d]),
    ):
        label = "all-in-one" if name == "allinone" else "single"
        ben_mse = score(model, scaler, ben)
        atk_mse = score(model, scaler, atk)
        thr = median_mad_threshold(ben_mse)
        y = [0] * len(ben_mse) + [1] * len(atk_mse)
        s = list(ben_mse) + list(atk_mse)
        auc = roc_auc(y, s)
        acc, tpr, fpr, tp, fp, tn, fn = metrics(y, [1 if v > thr else 0 for v in s])
        results[name].append((d, len(ben_mse), len(atk_mse), auc, thr, acc, tpr, fpr, tp, fp, tn, fn))
        tot[name]["y"] += y
        tot[name]["s"] += s
        print(f"  device{d:<5}{label:<10}{len(ben_mse):>7}{len(atk_mse):>7}{fmt(auc):>8}{thr:>10.4f}"
              f"{fmt(acc):>8}{fmt(tpr):>8}{fmt(fpr):>8}   {tp:>4} {fp:>4} {tn:>4} {fn:>4}")

print("  " + "-" * 96)
for name in ("allinone", "single"):
    y = tot[name]["y"]
    s = tot[name]["s"]
    auc = roc_auc(y, s)
    print(f"  OVERALL ({name})  AUC={fmt(auc)}   n={len(y)}  (pooled, no global threshold)")

# Recompute pooled per-device thresholds (same as printed) for an overall accuracy
print()
print("  Overall accuracy using per-device thresholds:")
for name, label in (("allinone", "all-in-one"), ("single", "single-device")):
    tp = sum(r[8] for r in results[name])
    fp = sum(r[9] for r in results[name])
    tn = sum(r[10] for r in results[name])
    fn = sum(r[11] for r in results[name])
    n = tp + fp + tn + fn
    acc = (tp + tn) / n
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * tpr / (prec + tpr) if prec and tpr else float("nan")
    print(f"    {label:<12} acc={acc:.4f}  TPR={tpr:.4f}  FPR={fpr:.4f}  prec={prec:.4f}  F1={f1:.4f}"
          f"  TP={tp} FP={fp} TN={tn} FN={fn}")
