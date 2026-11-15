#!/usr/bin/env python3
"""Recalibrate identity-engine CUSUM params for the deployed (winner) model.

After swapping model/autoencoder_allinone.keras to the stage-1 winner, the
per-device MSE thresholds and the benign anomaly probabilities (p_ben) that
drive the one-sided CUSUM must be re-derived from winner-model scores:

  - threshold : median + 2.5*MAD of benign RUN-1 MSE     (paper protocol)
  - p_ben     : fraction of benign RUN-1 packets above that threshold
  - CUSUM     : exact identity-engine logic (P_ATK, A kept fixed) replayed
                on winner-model flags for r2 / attack / attack_hard / stress

Output feeds entrypoint.sh (THRESHOLD per device) and identification_engine.py
(DEVICE_P_BEN dict). Run inside the pipeline image:
  docker run --rm --entrypoint python3 -v "$PWD:/work" -w /work \
      nbaiot-pipeline:latest recalibrate_cusum.py
"""
import csv
import os
import sys

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from sprt_accumulator import CUSUM, attack_onset

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

MODELS_DIR = os.environ.get("MODELS_DIR", "new_models/winner")
OUT_DIR = os.environ.get("OUT_DIR", "output")
CSV_OUT = os.environ.get("CSV_OUT", "output/cusum_recalibration_results.csv")

P_ATK = 0.3
CUSUM_A = 250.0


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


print("=" * 104)
print("  CUSUM recalibration for the deployed winner model  (P_ATK={}, A={})".format(P_ATK, CUSUM_A))
print("  threshold = median + 2.5*MAD of benign r1  |  p_ben = flag rate on r1")
print("=" * 104)

model = keras.models.load_model(f"{MODELS_DIR}/autoencoder_allinone.keras")
scaler = joblib.load(f"{MODELS_DIR}/scaler_allinone.pkl")

rows = []
for d in range(1, 10):
    m_r1 = score(model, scaler, load_features(d, "normal_r1"))
    thr = median_mad_threshold(m_r1)
    p_ben = float(np.mean(m_r1 > thr))

    mse_atk = score(model, scaler, load_features(d, "attack"))
    mse_hard = score(model, scaler, load_features(d, "attack_hard"))
    flags_r2 = (score(model, scaler, load_features(d, "normal_r2")) > thr).astype(int)
    flags_atk = (mse_atk > thr).astype(int)
    flags_hard = (mse_hard > thr).astype(int)
    flags_stress = (score(model, scaler, load_features(d, "stress")) > thr).astype(int)

    cusum = CUSUM(p_ben=p_ben, p_atk=P_ATK, A=CUSUM_A)

    def verdict(flags):
        path = cusum.path(flags)
        alarm = bool(path.max() >= CUSUM_A)
        idx = cusum.alarm_index(path, CUSUM_A)
        return alarm, idx

    a_r2, i_r2 = verdict(flags_r2)
    a_atk, i_atk = verdict(flags_atk)
    a_hard, i_hard = verdict(flags_hard)
    a_stress, i_stress = verdict(flags_stress)

    onset = attack_onset(mse_atk, thr, window=100, rate=0.2)  # proxy: first rising burst
    packets_after_onset = (i_atk - onset) if (a_atk and i_atk is not None and onset < i_atk) else None

    print(f"\n  device{d}")
    print(f"    thr={thr:.6f}  p_ben={p_ben:.4f}  | r2: {'ALARM@'+str(i_r2) if a_r2 else 'clean'}"
          f"  atk: {'ALARM(as in captured pos '+str(i_atk)+', +'+str(packets_after_onset)+' after onset)' if a_atk else 'MISSED'}"
          f"  hard: {'ALARM@'+str(i_hard) if a_hard else 'MISSED'}"
          f"  stress: {'ALARM@'+str(i_stress) if a_stress else 'clean'}")

    rows.append({"device": d, "threshold": round(thr, 6), "p_ben": round(p_ben, 4),
                 "n_r1": int(len(m_r1)), "fpr_r2": round(float(flags_r2.mean()), 4),
                 "tpr_attack": 1 if a_atk else 0, "alarm_packet_atk": i_atk,
                 "alarm_after_onset": packets_after_onset,
                 "tpr_attack_hard": 1 if a_hard else 0, "alarm_packet_hard": i_hard,
                 "stress_false_alarm": 1 if a_stress else 0})

print("\n" + "=" * 104)
print("  New DEVICE_P_BEN for identification_engine.py / sprt_accumulator.py")
print("=" * 104)
vals = {r["device"]: r["p_ben"] for r in rows}
print("  {" + ", ".join(f"{d}: {vals[d]:.3f}" for d in vals) + "}")

with open(CSV_OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\n  Written: {CSV_OUT}")
ok = all(r["tpr_attack"] and r["tpr_attack_hard"] and not r["fpr_r2"] and not r["stress_false_alarm"] for r in rows)
print(f"  All-device check: 100% TPR, 0% FPR-adjacent on r2/stress -> {'PASS — params sound' if ok else 'ATTENTION — inspect rows above'}")