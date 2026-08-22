#!/usr/bin/env python3
"""Evaluate accuracy of the all-in-one autoencoder model.

Reads output/device{N}_{mode}_results.csv for mode in {normal, attack} and
uses the `mse` column as the anomaly score. Ground truth is taken from the
filename (normal=0, attack=1). The `anomaly` column in those files was
written with possibly-stale thresholds, so this script re-thresholds the raw
mse scores instead.

Outputs, per device and overall:
  - ROC-AUC (threshold-independent model quality)
  - Confusion-matrix metrics at a per-device Youden-optimal threshold
  - Confusion-matrix metrics at a single global Youden-optimal threshold
"""
import csv
import glob
import os

N_DEVICES = 9
MODES = ("normal", "attack")
OUT = "output"


def load_mse(device, mode):
    path = os.path.join(OUT, f"device{device}_{mode}_results.csv")
    mses = []
    with open(path) as fh:
        r = csv.DictReader(fh)
        if "mse" not in (r.fieldnames or []):
            raise SystemExit(f"[ERROR] No mse column in {path}")
        for row in r:
            try:
                mses.append(float(row["mse"]))
            except (TypeError, ValueError):
                continue
    return mses


def roc_auc(y, score):
    """Rank-based AUC (Mann-Whitney U) with average-rank tie handling."""
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


def youden_threshold(y, score):
    """Threshold maximizing TPR - FPR (Youden's J)."""
    order = sorted(range(len(score)), key=lambda i: score[i])
    tpr = 0.0
    fpr = 0.0
    npos = max(1, sum(1 for v in y if v == 1))
    nneg = max(1, len(y) - npos)
    best = (score[0] - 1.0, float("-inf"))
    for i in order:
        if y[i] == 1:
            tpr += 1
        else:
            fpr += 1
        if tpr / npos - fpr / nneg >= best[1] - 1e-12:
            best = (score[i], tpr / npos - fpr / nneg)
    return best[0]


def metrics(y, pred):
    tp = fp = tn = fn = 0
    for yt, p in zip(y, pred):
        if p == 1 and yt == 1:
            tp += 1
        elif p == 1 and yt == 0:
            fp += 1
        elif p == 0 and yt == 0:
            tn += 1
        else:
            fn += 1
    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else float("nan")
    return tp, fp, tn, fn, acc, prec, rec, f1


def fmt(v):
    return f"{v:.4f}" if v == v else "  -  "


# ── Load all data ──────────────────────────────────────────
data = {}
missing = []
for d in range(1, N_DEVICES + 1):
    entry = {}
    for m in MODES:
        path = os.path.join(OUT, f"device{d}_{m}_results.csv")
        if not os.path.exists(path):
            missing.append(path)
            continue
        entry[m] = load_mse(d, m)
    data[d] = entry

if missing:
    print("[ERROR] Missing results files (run the pipeline for these devices first):")
    for p in missing:
        print(f"        {p}")
    sys.exit(1)

# ── Global pool ────────────────────────────────────────────
g_y = []
g_s = []
for d in range(1, N_DEVICES + 1):
    for s in data[d]["normal"]:
        g_y.append(0)
        g_s.append(s)
    for s in data[d]["attack"]:
        g_y.append(1)
        g_s.append(s)

global_thr = youden_threshold(g_y, g_s)
global_auc = roc_auc(g_y, g_s)

# ── Per-device ─────────────────────────────────────────────
print("=" * 92)
print("  PER-DEVICE (threshold = Youden-optimal on that device's mse)")
print("=" * 92)
print(f"  {'device':<8}{'n_ben':>7}{'n_atk':>7}{'AUC':>9}{'thr':>10}"
      f"{'acc':>8}{'prec':>8}{'rec':>8}{'F1':>8}   TP FP TN FN")
hdr = True
tot = dict(tp=0, fp=0, tn=0, fn=0)
n_ben_tot = n_atk_tot = 0
for d in range(1, N_DEVICES + 1):
    ben = data[d]["normal"]
    atk = data[d]["attack"]
    y = [0] * len(ben) + [1] * len(atk)
    s = ben + atk
    thr = youden_threshold(y, s)
    auc = roc_auc(y, s)
    tp, fp, tn, fn, acc, prec, rec, f1 = metrics(y, [1 if v > thr else 0 for v in s])
    tot["tp"] += tp
    tot["fp"] += fp
    tot["tn"] += tn
    tot["fn"] += fn
    n_ben_tot += len(ben)
    n_atk_tot += len(atk)
    print(f"  device{d}      {len(ben):>7}{len(atk):>7}{fmt(auc):>9}{thr:>10.4f}"
          f"{fmt(acc):>8}{fmt(prec):>8}{fmt(rec):>8}{fmt(f1):>8}   {tp:>4} {fp:>4} {tn:>4} {fn:>4}")

acc = (tot["tp"] + tot["tn"]) / (n_ben_tot + n_atk_tot)
prec = tot["tp"] / (tot["tp"] + tot["fp"]) if tot["tp"] + tot["fp"] else float("nan")
rec = tot["tp"] / (tot["tp"] + tot["fn"]) if tot["tp"] + tot["fn"] else float("nan")
f1 = 2 * prec * rec / (prec + rec) if prec and rec else float("nan")
print("  " + "-" * 88)
print(f"  OVERALL   {n_ben_tot:>7}{n_atk_tot:>7}{'':>9}{'':>10}"
      f"{fmt(acc):>8}{fmt(prec):>8}{fmt(rec):>8}{fmt(f1):>8}   "
      f"{tot['tp']:>4} {tot['fp']:>4} {tot['tn']:>4} {tot['fn']:>4}")

# ── Global threshold ───────────────────────────────────────
print()
print("=" * 92)
print("  GLOBAL SINGLE THRESHOLD (Youden-optimal on pooled mse)")
print("=" * 92)
print(f"  Global threshold : {global_thr:.4f}")
print(f"  Global ROC-AUC   : {global_auc:.4f}")
print(f"  {'device':<8}{'n_ben':>7}{'n_atk':>7}{'acc':>8}{'prec':>8}{'rec':>8}{'F1':>8}   TP FP TN FN")
g_tot = dict(tp=0, fp=0, tn=0, fn=0)
for d in range(1, N_DEVICES + 1):
    ben = data[d]["normal"]
    atk = data[d]["attack"]
    pred = [1 if v > global_thr else 0 for v in ben + atk]
    y = [0] * len(ben) + [1] * len(atk)
    tp, fp, tn, fn, acc, prec, rec, f1 = metrics(y, pred)
    g_tot["tp"] += tp
    g_tot["fp"] += fp
    g_tot["tn"] += tn
    g_tot["fn"] += fn
    print(f"  device{d}      {len(ben):>7}{len(atk):>7}{fmt(acc):>8}{fmt(prec):>8}"
          f"{fmt(rec):>8}{fmt(f1):>8}   {tp:>4} {fp:>4} {tn:>4} {fn:>4}")

acc = (g_tot["tp"] + g_tot["tn"]) / (n_ben_tot + n_atk_tot)
prec = g_tot["tp"] / (g_tot["tp"] + g_tot["fp"]) if g_tot["tp"] + g_tot["fp"] else float("nan")
rec = g_tot["tp"] / (g_tot["tp"] + g_tot["fn"]) if g_tot["tp"] + g_tot["fn"] else float("nan")
f1 = 2 * prec * rec / (prec + rec) if prec and rec else float("nan")
print("  " + "-" * 84)
print(f"  OVERALL   {n_ben_tot:>7}{n_atk_tot:>7}{fmt(acc):>8}{fmt(prec):>8}"
      f"{fmt(rec):>8}{fmt(f1):>8}   {g_tot['tp']:>4} {g_tot['fp']:>4} {g_tot['tn']:>4} {g_tot['fn']:>4}")
