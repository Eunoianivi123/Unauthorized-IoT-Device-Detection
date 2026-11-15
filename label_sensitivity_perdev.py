#!/usr/bin/env python3
"""Per-device N-BaIoT baseline under the same strict/session label views.

Companion to ``label_sensitivity.py`` (which scores the deployed unified C2
model).  This script scores ONE device at a time so that loading a different
``.keras`` file in each subprocess avoids the Keras layer-name weight-aliasing
bug documented in AGENTS.md.

Outputs (one row per definition x mode):
    output/label_sensitivity_perdev_device{D}.csv

Run one device per process, e.g.:
  for d in 1 2 3 4 5 6 7 8 9; do
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
      -v "$PWD:/work" -w /work --entrypoint python3 \
      nbaiot-pipeline:latest label_sensitivity_perdev.py --device $d
  done

Then:
  python3 label_sensitivity_perdev.py --aggregate
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

import cusum_eval as ce
from label_sensitivity import (
    CAPTURES,
    DEFINITIONS,
    fmt,
    load_labels,
    masks_for,
    rates,
)


CACHE_DIRNAME = "label_sensitivity_perdev_scores"


def run_device(device: int, out_dir: str, label_dir: str, model_dir: str):
    from tensorflow import keras
    import joblib

    model = keras.models.load_model(
        os.path.join(model_dir, f"autoencoder_{device}.keras")
    )
    scaler = joblib.load(os.path.join(model_dir, f"scaler_{device}.pkl"))

    scores = {}
    labels = {}
    for mode in CAPTURES:
        path = os.path.join(out_dir, f"device{device}_{mode}_capture_features.csv")
        scored = ce.score_capture(path, model, scaler)
        lab = load_labels(device, mode, label_dir)
        if len(lab) != len(scored):
            raise SystemExit(
                f"[ERROR] device{device} {mode}: {len(lab)} labels vs "
                f"{len(scored)} feature rows"
            )
        if not np.array_equal(lab["row"].to_numpy(), np.arange(len(lab))):
            raise SystemExit(f"[ERROR] device{device} {mode}: label row order broken")
        scores[mode] = scored["mse"].to_numpy()
        labels[mode] = lab

    thr = ce.median_mad_threshold(scores["normal_r1"])
    flags = {m: (scores[m] > thr).astype(int) for m in CAPTURES}

    cache_dir = os.path.join(out_dir, CACHE_DIRNAME)
    os.makedirs(cache_dir, exist_ok=True)
    payload = {"threshold": np.asarray([thr], dtype=np.float64)}
    for mode in CAPTURES:
        payload[f"{mode}_mse"] = scores[mode]
        for defn in ("strict", "session", "refined"):
            payload[f"{mode}_{defn}"] = (
                labels[mode][f"label_{defn}"].to_numpy()
            )
    np.savez_compressed(os.path.join(cache_dir, f"device{device}.npz"), **payload)

    rows = []

    for defn in DEFINITIONS:
        m = masks_for(defn, labels)
        r2_neg = flags["normal_r2"][m["r2_ben"]]
        hb = flags["attack_hard"][m["hard_ben"]]
        st = flags["stress"][m["stress_ben"]]

        neg_a = r2_neg
        pos_a = flags["attack"][m["atk_pos"]]
        tp, fp, tn, fn, acc, tpr, fpr, prec, f1 = rates(neg_a, pos_a)
        auc_a = ce.roc_auc(
            scores["normal_r2"][m["r2_ben"]],
            scores["attack"][m["atk_pos"]],
        )
        rows.append({
            "device": device, "family": "per_device", "definition": defn,
            "mode": "attack", "threshold": round(thr, 6),
            "n_pos": int(len(pos_a)), "n_neg": int(len(neg_a)),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "tpr": round(tpr, 6), "fpr": round(fpr, 6),
            "precision": round(prec, 6), "f1": round(f1, 6),
            "accuracy": round(acc, 6), "auc": round(auc_a, 6),
            "auc_r2": round(auc_a, 6), "auc_in_capture": "",
            "fpr_r2": round(float(r2_neg.mean()), 6) if len(r2_neg) else "",
            "stress_flag_rate": round(float(st.mean()), 6) if len(st) else "",
            "fpr_hard_benign": "",
        })

        neg_h = np.concatenate([r2_neg, hb])
        pos_h = flags["attack_hard"][m["hard_pos"]]
        tp, fp, tn, fn, acc, tpr, fpr, prec, f1 = rates(neg_h, pos_h)
        neg_s_r2 = scores["normal_r2"][m["r2_ben"]]
        neg_s_hb = scores["attack_hard"][m["hard_ben"]]
        pos_s = scores["attack_hard"][m["hard_pos"]]
        auc_r2 = ce.roc_auc(neg_s_r2, pos_s)
        auc_hb = ce.roc_auc(neg_s_hb, pos_s) if len(neg_s_hb) else float("nan")
        auc_h = ce.roc_auc(np.concatenate([neg_s_r2, neg_s_hb]), pos_s)
        rows.append({
            "device": device, "family": "per_device", "definition": defn,
            "mode": "attack_hard", "threshold": round(thr, 6),
            "n_pos": int(len(pos_h)), "n_neg": int(len(neg_h)),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "tpr": round(tpr, 6), "fpr": round(fpr, 6),
            "precision": round(prec, 6), "f1": round(f1, 6),
            "accuracy": round(acc, 6), "auc": round(auc_h, 6),
            "auc_r2": round(auc_r2, 6),
            "auc_in_capture": round(auc_hb, 6) if auc_hb == auc_hb else "",
            "fpr_r2": round(float(r2_neg.mean()), 6) if len(r2_neg) else "",
            "stress_flag_rate": round(float(st.mean()), 6) if len(st) else "",
            "fpr_hard_benign": round(float(hb.mean()), 6) if len(hb) else "",
        })

    out_path = os.path.join(out_dir, f"label_sensitivity_perdev_device{device}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[*] device {device} -> {out_path}  (threshold {thr:.6f})")


def _masks(defn: str, device_arrays: dict):
    """Per-device boolean masks for one label definition."""
    if defn == "current":
        return {
            "r2_ben": np.ones(len(device_arrays["normal_r2_mse"]), dtype=bool),
            "atk_pos": np.ones(len(device_arrays["attack_mse"]), dtype=bool),
            "hard_ben": np.zeros(len(device_arrays["attack_hard_mse"]), dtype=bool),
            "hard_pos": np.ones(len(device_arrays["attack_hard_mse"]), dtype=bool),
            "stress_ben": np.ones(len(device_arrays["stress_mse"]), dtype=bool),
        }
    return {
        "r2_ben": device_arrays[f"normal_r2_{defn}"] == 0,
        "atk_pos": device_arrays[f"attack_{defn}"] == 1,
        "hard_ben": device_arrays[f"attack_hard_{defn}"] == 0,
        "hard_pos": device_arrays[f"attack_hard_{defn}"] == 1,
        "stress_ben": device_arrays[f"stress_{defn}"] == 0,
    }


def aggregate(out_dir: str):
    paths = [
        os.path.join(out_dir, f"label_sensitivity_perdev_device{d}.csv")
        for d in range(1, 10)
    ]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"[ERROR] missing per-device files: {missing}")

    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    per_path = os.path.join(out_dir, "label_sensitivity_perdev_per_device.csv")
    df.to_csv(per_path, index=False)

    cache_dir = os.path.join(out_dir, CACHE_DIRNAME)
    caches = {}
    for d in range(1, 10):
        p = os.path.join(cache_dir, f"device{d}.npz")
        if not os.path.exists(p):
            raise SystemExit(f"[ERROR] missing score cache: {p}")
        with np.load(p) as z:
            caches[d] = {k: z[k] for k in z.files}

    pooled = []
    for defn in DEFINITIONS:
        for mode in ("attack", "attack_hard"):
            neg_f, pos_f = [], []
            neg_s_r2, neg_s_hb, pos_s = [], [], []
            r2_flags, st_flags, hb_flags = [], [], []
            for d in range(1, 10):
                c = caches[d]
                thr = float(c["threshold"][0])
                m = _masks(defn, c)
                flags = {
                    "normal_r2": (c["normal_r2_mse"] > thr).astype(int),
                    "attack": (c["attack_mse"] > thr).astype(int),
                    "attack_hard": (c["attack_hard_mse"] > thr).astype(int),
                    "stress": (c["stress_mse"] > thr).astype(int),
                }
                r2_neg = flags["normal_r2"][m["r2_ben"]]
                st = flags["stress"][m["stress_ben"]]
                hb = flags["attack_hard"][m["hard_ben"]]
                r2_flags.append(r2_neg)
                st_flags.append(st)
                hb_flags.append(hb)

                if mode == "attack":
                    neg_f.append(r2_neg)
                    pos_f.append(flags["attack"][m["atk_pos"]])
                    neg_s_r2.append(c["normal_r2_mse"][m["r2_ben"]])
                    pos_s.append(c["attack_mse"][m["atk_pos"]])
                else:
                    neg_h = np.concatenate([r2_neg, hb])
                    neg_f.append(neg_h)
                    pos_f.append(flags["attack_hard"][m["hard_pos"]])
                    neg_s_r2.append(c["normal_r2_mse"][m["r2_ben"]])
                    neg_s_hb.append(c["attack_hard_mse"][m["hard_ben"]])
                    pos_s.append(c["attack_hard_mse"][m["hard_pos"]])

            neg_f = np.concatenate(neg_f); pos_f = np.concatenate(pos_f)
            neg_s_r2 = np.concatenate(neg_s_r2); pos_s = np.concatenate(pos_s)
            neg_s_hb = (np.concatenate(neg_s_hb)
                        if neg_s_hb else np.array([], dtype=float))
            neg_s = np.concatenate([neg_s_r2, neg_s_hb])
            tp, fp, tn, fn, acc, tpr, fpr, prec, f1 = rates(neg_f, pos_f)
            auc = ce.roc_auc(neg_s, pos_s)
            auc_r2 = ce.roc_auc(neg_s_r2, pos_s)
            auc_hb = ce.roc_auc(neg_s_hb, pos_s) if len(neg_s_hb) else float("nan")
            r2_all = np.concatenate(r2_flags)
            st_all = np.concatenate(st_flags)
            hb_all = np.concatenate(hb_flags)
            pooled.append({
                "family": "per_device", "definition": defn, "mode": mode,
                "n_pos": int(len(pos_f)), "n_neg": int(len(neg_f)),
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
                "tpr": round(tpr, 6), "fpr": round(fpr, 6),
                "precision": round(prec, 6), "f1": round(f1, 6),
                "accuracy": round(acc, 6), "auc": round(auc, 6),
                "auc_r2": round(auc_r2, 6),
                "auc_in_capture": round(auc_hb, 6) if auc_hb == auc_hb else "",
                "fpr_r2": round(float(r2_all.mean()), 6) if len(r2_all) else "",
                "stress_flag_rate": round(float(st_all.mean()), 6) if len(st_all) else "",
                "fpr_hard_benign": round(float(hb_all.mean()), 6) if len(hb_all) else "",
                "note": "pooled over packets; AUC computed from cached per-device scores",
            })
    pool_df = pd.DataFrame(pooled)
    pool_path = os.path.join(out_dir, "label_sensitivity_perdev_pooled.csv")
    pool_df.to_csv(pool_path, index=False)

    print("\n" + "=" * 100)
    print("  PER-DEVICE N-BaIoT BASELINE -- strict vs session vs legacy-current")
    print("  (pooled confusion counts; AUC is the macro mean over 9 devices)")
    print("=" * 100)
    hdr = (f"  {'definition':<10}{'mode':<13}{'TPR':>8}{'FPR':>8}{'prec':>8}"
           f"{'F1':>8}{'acc':>8}{'AUC':>8}{'AUCr2':>8}{'AUCcap':>8}")
    print(hdr)
    for _, r in pool_df.iterrows():
        print(f"  {r['definition']:<10}{r['mode']:<13}"
              f"{fmt(r['tpr']):>8}{fmt(r['fpr']):>8}{fmt(r['precision']):>8}"
              f"{fmt(r['f1']):>8}{fmt(r['accuracy']):>8}{fmt(r['auc']):>8}"
              f"{fmt(r['auc_r2']):>8}{fmt(r['auc_in_capture']):>8}")
    print(f"\n[*] per-device -> {per_path}")
    print(f"[*] pooled     -> {pool_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--out-dir", default=os.environ.get("OUT_DIR", "output"))
    ap.add_argument("--label-dir", default=None)
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "new_models/winner"))
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args.out_dir)
        return
    if args.device is None:
        raise SystemExit("pass --device N or --aggregate")
    label_dir = args.label_dir or os.path.join(args.out_dir, "labels")
    run_device(args.device, args.out_dir, label_dir, args.model_dir)


if __name__ == "__main__":
    main()
