#!/usr/bin/env python3
"""Strict vs session per-packet metrics after the role-label fix.

This is the paper-facing follow-up to reviewer comment 1.  It re-scores every
canonical feature CSV with the deployed C2 model (same code path as
``baseline_comparison.py`` / ``cusum_eval.py``) and evaluates four label
definitions:

  current   legacy filename rule: every row of an attack/attack_hard capture is
            positive, every row of benign r2 is negative.
  strict    positives = device-origin attack packets; negatives = device-origin
            benign packets; target/gateway responses excluded from both.
  session   positives = attack packets + target responses; negatives =
            device-origin benign packets (the requested sensitivity check).
  refined   like session, but an attack_hard target response counts as attack
            only when its source port is not a known benign device port.

The threshold is unchanged: median + 2.5*MAD of ALL benign r1 rows, per device
(the deployed rule).  FPR is measured on held-out benign r2; for the strict
definitions target responses are excluded from the benign denominator.

Run in the pipeline image:
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
      -v "$PWD:/work" -w /work --entrypoint python3 \
      nbaiot-pipeline:latest label_sensitivity.py
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import pandas as pd

import cusum_eval as ce

CAPTURES = ("normal_r1", "normal_r2", "attack", "attack_hard", "stress")
DEFINITIONS = ("current", "strict", "session", "refined")


def load_labels(device: int, mode: str, label_dir: str) -> pd.DataFrame:
    path = os.path.join(label_dir, f"device{device}_{mode}_packet_labels.csv.gz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing -- run build_packet_labels.py first"
        )
    return pd.read_csv(path)


def masks_for(definition: str, labels: dict):
    """Return boolean masks for the r2/stress/hard-benign/attack/hard classes."""
    r2 = labels["normal_r2"]
    atk = labels["attack"]
    hard = labels["attack_hard"]
    stress = labels["stress"]

    if definition == "current":
        # Legacy rule: every attack_hard row is a positive, so there are NO
        # in-capture benign negatives; the only negatives are the benign r2 rows.
        return {
            "r2_ben": np.ones(len(r2), dtype=bool),
            "atk_pos": np.ones(len(atk), dtype=bool),
            "hard_ben": np.zeros(len(hard), dtype=bool),
            "hard_pos": np.ones(len(hard), dtype=bool),
            "stress_ben": np.ones(len(stress), dtype=bool),
        }

    col = f"label_{definition}"
    return {
        "r2_ben": r2[col].to_numpy() == 0,
        "atk_pos": atk[col].to_numpy() == 1,
        "hard_ben": hard[col].to_numpy() == 0,
        "hard_pos": hard[col].to_numpy() == 1,
        "stress_ben": stress[col].to_numpy() == 0,
    }


def rates(pred_neg: np.ndarray, pred_pos: np.ndarray):
    pred_neg = np.asarray(pred_neg, dtype=int)
    pred_pos = np.asarray(pred_pos, dtype=int)
    tp = int(pred_pos.sum())
    fn = int(len(pred_pos) - tp)
    fp = int(pred_neg.sum())
    tn = int(len(pred_neg) - fp)
    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else float("nan")
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * prec * tpr / (prec + tpr)) if (prec + tpr) else float("nan")
    return tp, fp, tn, fn, acc, tpr, fpr, prec, f1


def fmt(v):
    if isinstance(v, str):
        return v if v else "  -  "
    return f"{v:.4f}" if v == v else "  -  "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.environ.get("OUT_DIR", "output"))
    ap.add_argument("--label-dir", default=None)
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "new_models/winner"))
    ap.add_argument("--devices", default="1-9")
    args = ap.parse_args()

    label_dir = args.label_dir or os.path.join(args.out_dir, "labels")
    if "-" in args.devices:
        a, b = args.devices.split("-", 1)
        devices = list(range(int(a), int(b) + 1))
    else:
        devices = [int(x) for x in args.devices.split(",") if x]

    print(f"[*] model dir : {args.model_dir}")
    print(f"[*] label dir : {label_dir}")
    model, scaler = ce.load_model(args.model_dir)

    pooled = {
        (defn, mode): {"neg_s_r2": [], "neg_s_hb": [], "pos_s": [],
                       "neg_f": [], "pos_f": []}
        for defn in DEFINITIONS for mode in ("attack", "attack_hard")
    }
    pooled_extra = {
        (defn, key): {"flags": [], "n": 0}
        for defn in DEFINITIONS for key in ("r2", "stress", "hard_benign")
    }
    rows = []

    for d in devices:
        scores = {}
        labels = {}
        for mode in CAPTURES:
            path = os.path.join(args.out_dir, f"device{d}_{mode}_capture_features.csv")
            scored = ce.score_capture(path, model, scaler)
            lab = load_labels(d, mode, label_dir)
            if len(lab) != len(scored):
                raise SystemExit(
                    f"[ERROR] device{d} {mode}: {len(lab)} labels vs "
                    f"{len(scored)} feature rows"
                )
            if not np.array_equal(lab["row"].to_numpy(), np.arange(len(lab))):
                raise SystemExit(f"[ERROR] device{d} {mode}: label row order broken")
            scores[mode] = scored["mse"].to_numpy()
            labels[mode] = lab

        thr = ce.median_mad_threshold(scores["normal_r1"])
        flags = {m: (scores[m] > thr).astype(int) for m in CAPTURES}

        for defn in DEFINITIONS:
            m = masks_for(defn, labels)

            # ---- held-out benign r2 FPR ----
            r2_neg = flags["normal_r2"][m["r2_ben"]]
            pooled_extra[(defn, "r2")]["flags"].append(r2_neg)
            pooled_extra[(defn, "r2")]["n"] += int(m["r2_ben"].sum())

            # ---- stress flag rate ----
            st = flags["stress"][m["stress_ben"]]
            pooled_extra[(defn, "stress")]["flags"].append(st)
            pooled_extra[(defn, "stress")]["n"] += int(m["stress_ben"].sum())

            # ---- in-capture benign rows of attack_hard ----
            hb = flags["attack_hard"][m["hard_ben"]]
            pooled_extra[(defn, "hard_benign")]["flags"].append(hb)
            pooled_extra[(defn, "hard_benign")]["n"] += int(m["hard_ben"].sum())

            # ---- standard attack ----
            neg_a = r2_neg
            pos_a = flags["attack"][m["atk_pos"]]
            tp, fp, tn, fn, acc, tpr, fpr, prec, f1 = rates(neg_a, pos_a)
            neg_s = scores["normal_r2"][m["r2_ben"]]
            pos_s = scores["attack"][m["atk_pos"]]
            auc_a = ce.roc_auc(neg_s, pos_s)
            rows.append({
                "device": d, "definition": defn, "mode": "attack",
                "threshold": round(thr, 6),
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
            p = pooled[(defn, "attack")]
            p["neg_s_r2"].append(neg_s); p["pos_s"].append(pos_s)
            p["neg_f"].append(neg_a); p["pos_f"].append(pos_a)

            # ---- stealth attack_hard (r2 + in-capture benign negatives) ----
            neg_h = np.concatenate([r2_neg, hb])
            pos_h = flags["attack_hard"][m["hard_pos"]]
            tp, fp, tn, fn, acc, tpr, fpr, prec, f1 = rates(neg_h, pos_h)
            neg_s_r2 = scores["normal_r2"][m["r2_ben"]]
            neg_s_hb = scores["attack_hard"][m["hard_ben"]]
            pos_s = scores["attack_hard"][m["hard_pos"]]
            auc_r2 = ce.roc_auc(neg_s_r2, pos_s)
            auc_hb = ce.roc_auc(neg_s_hb, pos_s) if len(neg_s_hb) else float("nan")
            neg_s = np.concatenate([neg_s_r2, neg_s_hb])
            auc_h = ce.roc_auc(neg_s, pos_s)
            rows.append({
                "device": d, "definition": defn, "mode": "attack_hard",
                "threshold": round(thr, 6),
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
            p = pooled[(defn, "attack_hard")]
            p["neg_s_r2"].append(neg_s_r2)
            p["neg_s_hb"].append(neg_s_hb)
            p["pos_s"].append(pos_s)
            p["neg_f"].append(neg_h); p["pos_f"].append(pos_h)

    per_device = pd.DataFrame(rows)
    per_path = os.path.join(args.out_dir, "label_sensitivity_per_device.csv")
    per_device.to_csv(per_path, index=False)

    pooled_rows = []
    for defn in DEFINITIONS:
        for mode in ("attack", "attack_hard"):
            p = pooled[(defn, mode)]
            neg_f = np.concatenate(p["neg_f"])
            pos_f = np.concatenate(p["pos_f"])
            neg_s_r2 = np.concatenate(p["neg_s_r2"])
            neg_s_hb = (np.concatenate(p["neg_s_hb"])
                        if p["neg_s_hb"] else np.array([], dtype=float))
            pos_s = np.concatenate(p["pos_s"])
            neg_s = np.concatenate([neg_s_r2, neg_s_hb])
            tp, fp, tn, fn, acc, tpr, fpr, prec, f1 = rates(neg_f, pos_f)
            auc = ce.roc_auc(neg_s, pos_s)
            auc_r2 = ce.roc_auc(neg_s_r2, pos_s)
            auc_hb = (ce.roc_auc(neg_s_hb, pos_s)
                      if len(neg_s_hb) else float("nan"))
            r2 = pooled_extra[(defn, "r2")]
            st = pooled_extra[(defn, "stress")]
            hb = pooled_extra[(defn, "hard_benign")]
            r2_flags = np.concatenate(r2["flags"])
            st_flags = np.concatenate(st["flags"])
            hb_flags = np.concatenate(hb["flags"])
            pooled_rows.append({
                "definition": defn, "mode": mode,
                "n_pos": int(len(pos_f)), "n_neg": int(len(neg_f)),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "tpr": round(tpr, 6), "fpr": round(fpr, 6),
                "precision": round(prec, 6), "f1": round(f1, 6),
                "accuracy": round(acc, 6), "auc": round(auc, 6),
                "auc_r2": round(auc_r2, 6),
                "auc_in_capture": round(auc_hb, 6) if auc_hb == auc_hb else "",
                "fpr_r2": round(float(r2_flags.mean()), 6) if len(r2_flags) else "",
                "stress_flag_rate": round(float(st_flags.mean()), 6) if len(st_flags) else "",
                "fpr_hard_benign": round(float(hb_flags.mean()), 6) if len(hb_flags) else "",
            })
    pooled_df = pd.DataFrame(pooled_rows)
    pool_path = os.path.join(args.out_dir, "label_sensitivity_pooled.csv")
    pooled_df.to_csv(pool_path, index=False)

    # ---------------- report ----------------
    print("\n" + "=" * 100)
    print("  POOLED OVER 9 DEVICES -- strict vs session vs legacy-current")
    print("  threshold: median + 2.5*MAD of benign r1 (deployed rule)")
    print("=" * 100)
    hdr = (f"  {'definition':<10}{'mode':<13}{'TPR':>8}{'FPR':>8}{'prec':>8}"
           f"{'F1':>8}{'acc':>8}{'AUC':>8}{'AUCr2':>8}{'AUCcap':>8}"
           f"{'FPR(r2)':>9}{'stress':>8}{'hardBen':>9}")
    print(hdr)
    for _, r in pooled_df.iterrows():
        print(f"  {r['definition']:<10}{r['mode']:<13}"
              f"{fmt(r['tpr']):>8}{fmt(r['fpr']):>8}{fmt(r['precision']):>8}"
              f"{fmt(r['f1']):>8}{fmt(r['accuracy']):>8}{fmt(r['auc']):>8}"
              f"{fmt(r['auc_r2']):>8}{fmt(r['auc_in_capture']):>8}"
              f"{fmt(r['fpr_r2']):>9}{fmt(r['stress_flag_rate']):>8}"
              f"{fmt(r['fpr_hard_benign']):>9}")

    print("\n" + "=" * 100)
    print("  PER-DEVICE attack_hard -- the comment-1 headline")
    print("=" * 100)
    print(f"  {'dev':<5}{'definition':<10}{'n_pos':>7}{'n_neg':>7}{'TPR':>8}{'FPR':>8}"
          f"{'F1':>8}{'AUCr2':>8}{'AUCcap':>8}{'FPR(r2)':>9}{'hardBen':>9}")
    sub = per_device[per_device["mode"] == "attack_hard"]
    for _, r in sub.iterrows():
        print(f"  d{int(r['device']):<4}{r['definition']:<10}{int(r['n_pos']):>7}"
              f"{int(r['n_neg']):>7}{fmt(r['tpr']):>8}{fmt(r['fpr']):>8}"
              f"{fmt(r['f1']):>8}{fmt(r['auc_r2']):>8}"
              f"{fmt(r['auc_in_capture']):>8}{fmt(r['fpr_r2']):>9}"
              f"{fmt(r['fpr_hard_benign']):>9}")

    print(f"\n[*] Per-device -> {per_path}")
    print(f"[*] Pooled     -> {pool_path}")


if __name__ == "__main__":
    main()
