#!/usr/bin/env python3
"""Review comment 6 -- baseline comparison.

Scores every device's captures with BOTH model families on identical data:

  allinone   : model/autoencoder_allinone.keras  (this work -- unified model)
  per-device : model/autoencoder_{d}.keras       (N-BaIoT baseline, Meidan et al.)

Protocol (identical to final_evaluation.py / AGENTS.md):
  - threshold  : median + 2.5*MAD of benign RUN-1 MSE only, per device+model
  - FPR        : held-out benign RUN-2 (never seen by the threshold)
  - detection  : attack and attack_hard (stealth) captures, reported separately
  - AUC        : pooled r1+r2 benign vs that attack mode
  - robustness : stress captures (benign bursts) reported as extra flag rate

Run inside the pipeline image:
  docker run --rm -v "$PWD:/work" -w /work nbaiot-pipeline:latest \
      python3 baseline_comparison.py
"""
import csv
import os
import sys

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

# Exact 115 N-BaIoT feature names in correct order (keep in sync with
# pcap_csv.py output columns / testmodel.py).
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
assert len(FEATURE_NAMES) == 115

MODEL_DIR = os.environ.get("MODEL_DIR", "model")
OUT_DIR = os.environ.get("OUT_DIR", "output")
CSV_OUT = os.path.join(OUT_DIR, "baseline_comparison_results.csv")


def load_features(device, kind):
    path = os.path.join(OUT_DIR, f"device{device}_{kind}_capture_features.csv")
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Missing {path}")
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


def confusion(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return tp, fp, tn, fn


def rates(tp, fp, tn, fn):
    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else float("nan")
    tpr = tp / (tp + fn) if tp + fn else float("nan")          # recall
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = (2 * prec * tpr / (prec + tpr)) if (prec + tpr) else float("nan")
    return acc, tpr, fpr, prec, f1


def roc_auc(y, s):
    """Rank-based AUC with tie handling (Mann-Whitney)."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    npos = int((y == 1).sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and s[order[j]] == s[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def fmt(v):
    return f"{v:.4f}" if v == v else "  -  "


FAMILIES = ("allinone", "perdevice")

print("=" * 108)
print("  Review comment 6: unified (all-in-one) autoencoder vs per-device N-BaIoT autoencoder baseline")
print("  threshold = median + 2.5*MAD of benign r1 MSE | FPR from held-out benign r2 | same captures for both models")
print("=" * 108)

models = {
    "allinone": (
        keras.models.load_model(f"{MODEL_DIR}/autoencoder_allinone.keras"),
        joblib.load(f"{MODEL_DIR}/scaler_allinone.pkl"),
    )
}
dev_cache = {}

rows_csv = []
pooled = {f: {"y_atk": [], "s_atk": [], "y_hard": [], "s_hard": [],
              "tp_a": 0, "fp_a": 0, "tn_a": 0, "fn_a": 0,
              "tp_h": 0, "fp_h": 0, "tn_h": 0, "fn_h": 0} for f in FAMILIES}

for d in range(1, 10):
    ben1 = load_features(d, "normal_r1")
    ben2 = load_features(d, "normal_r2")
    atk = load_features(d, "attack")
    hard = load_features(d, "attack_hard")
    stress = load_features(d, "stress")

    if d not in dev_cache:
        dev_cache[d] = (
            keras.models.load_model(f"{MODEL_DIR}/autoencoder_{d}.keras"),
            joblib.load(f"{MODEL_DIR}/scaler_{d}.pkl"),
        )

    print(f"\n  device{d}")
    print(f"    {'model':<11}{'thr':>9}{'AUC@atk':>9}{'acc':>8}{'TPR/rec':>9}{'prec':>8}{'F1':>8}{'FPR(r2)':>9}"
          f"{' | AUC@hard':>10}{'TPR':>7}{'F1':>8}{'stressFR':>10}")

    for fam in FAMILIES:
        model, scaler = models["allinone"] if fam == "allinone" else dev_cache[d]
        m_r1 = score(model, scaler, ben1)
        thr = median_mad_threshold(m_r1)

        pred_r2 = (score(model, scaler, ben2) > thr).astype(int)
        mse_atk = score(model, scaler, atk)
        mse_hard = score(model, scaler, hard)
        fr_stress = float(np.mean(score(model, scaler, stress) > thr))

        # attack mode: held-out benign r2 (label 0) vs attack (label 1)
        y_atk = [0] * len(pred_r2) + [1] * len(mse_atk)
        s_atk = [0.0] * len(pred_r2)  # placeholder replaced below
        s_r2 = score(model, scaler, ben2)
        s_atk = list(s_r2) + list(mse_atk)
        auc_atk = roc_auc(y_atk, s_atk)
        tp, fp, tn, fn = confusion(y_atk, list(pred_r2) + list((mse_atk > thr).astype(int)))
        acc, rec, fpr, prec, f1 = rates(tp, fp, tn, fn)

        # stealth (attack_hard) mode
        y_h = [0] * len(pred_r2) + [1] * len(mse_hard)
        s_h = list(s_r2) + list(mse_hard)
        auc_h = roc_auc(y_h, s_h)
        tp_h, fp_h, tn_h, fn_h = confusion(y_h, list(pred_r2) + list((mse_hard > thr).astype(int)))
        _, rec_h, _, _, f1_h = rates(tp_h, fp_h, tn_h, fn_h)

        label = "all-in-one" if fam == "allinone" else "per-device"
        print(f"    {label:<11}{thr:>9.4f}{fmt(auc_atk):>9}{fmt(acc):>8}{fmt(rec):>9}{fmt(prec):>8}{fmt(f1):>8}"
              f"{fmt(fpr):>9}  |{fmt(auc_h):>9}{rec_h:>7.3f}{fmt(f1_h):>8}{fr_stress:>10.4f}")

        rows_csv.append({
            "device": d, "family": fam, "threshold": round(thr, 6),
            "n_benign_r1": int(len(m_r1)), "n_benign_r2": int(len(pred_r2)),
            "n_attack": int(len(mse_atk)), "n_attack_hard": int(len(mse_hard)), "n_stress": int(len(stress)),
            "fpr_r2": round(fpr, 6),
            "tpr_attack": round(rec, 6), "accuracy_attack": round(acc, 6),
            "precision_attack": round(prec, 6), "recall_attack": round(rec, 6),
            "f1_attack": round(f1, 6), "auc_attack": round(auc_atk, 6),
            "tpr_attack_hard": round(rec_h, 6),
            "precision_attack_hard": round(tp_h / (tp_h + fp_h), 6) if tp_h + fp_h else "",
            "f1_attack_hard": round(f1_h, 6), "auc_attack_hard": round(auc_h, 6),
            "stress_flag_rate": round(fr_stress, 6),
            "tp_attack": tp, "fp_attack": fp, "tn_attack": tn, "fn_attack": fn,
            "tp_attack_hard": tp_h, "fp_attack_hard": fp_h,
            "tn_attack_hard": tn_h, "fn_attack_hard": fn_h,
        })

        p = pooled[fam]
        p["y_atk"] += y_atk; p["s_atk"] += s_atk
        p["y_hard"] += y_h; p["s_hard"] += s_h
        p["tp_a"] += tp; p["fp_a"] += fp; p["tn_a"] += tn; p["fn_a"] += fn
        p["tp_h"] += tp_h; p["fp_h"] += fp_h; p["tn_h"] += tn_h; p["fn_h"] += fn_h

print("\n" + "=" * 108)
print("  OVERALL (all 9 devices pooled)")
print("=" * 108)
hdr = (f"    {'model':<12}{'AUC@atk':>9}{'acc':>8}{'TPR/rec':>9}{'prec':>8}{'F1':>8}{'FPR(r2)':>9}"
       f"{' | AUC@hard':>10}{'TPR':>7}{'F1':>8}")
print(hdr)
summary = []
for fam in FAMILIES:
    p = pooled[fam]
    auc_atk = roc_auc(p["y_atk"], p["s_atk"])
    auc_h = roc_auc(p["y_hard"], p["s_hard"])
    acc, rec, fpr, prec, f1 = rates(p["tp_a"], p["fp_a"], p["tn_a"], p["fn_a"])
    _, rec_h, _, _, f1_h = rates(p["tp_h"], p["fp_h"], p["tn_h"], p["fn_h"])
    label = "all-in-one" if fam == "allinone" else "per-device"
    print(f"    {label:<12}{fmt(auc_atk):>9}{fmt(acc):>8}{fmt(rec):>9}{fmt(prec):>8}{fmt(f1):>8}"
          f"{fmt(fpr):>9}  |{fmt(auc_h):>9}{rec_h:>7.3f}{fmt(f1_h):>8}")
    summary.append((label, fam))

with open(CSV_OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_csv[0].keys()))
    w.writeheader()
    w.writerows(rows_csv)
print(f"\n  Per-device results written to: {CSV_OUT}")
