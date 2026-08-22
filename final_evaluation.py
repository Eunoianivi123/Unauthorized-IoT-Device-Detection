#!/usr/bin/env python3
"""Final all-in-one evaluation with more benign data (2 captures/device).

Methodology:
  - Threshold calibrated on RUN 1 benign MSE  ->  median + 2.5*MAD
  - FPR measured on RUN 2 benign MSE (held-out, never seen by the threshold)
  - TPR measured on attack MSE
  - AUC computed on RUN1+RUN2 benign vs attack (bigger benign pool)
"""
import csv
import statistics

N = 9


def read_mse(path):
    out = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            try:
                out.append(float(row["mse"]))
            except (TypeError, ValueError):
                continue
    return out


def median_mad(v):
    med = statistics.median(v)
    mad = statistics.median(abs(x - med) for x in v)
    return med + 2.5 * mad


def roc_auc(y, s):
    n = len(y)
    npos = sum(1 for v in y if v == 1)
    nneg = n - npos
    order = sorted(range(n), key=lambda i: s[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and s[order[j]] == s[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    rs = sum(ranks[i] for i in range(n) if y[i] == 1)
    return (rs - npos * (npos + 1) / 2.0) / (npos * nneg)


print(f"  {'device':<6}{'r1_rows':>8}{'r2_rows':>8}{'thr(r1)':>10}{'FPR(r2)':>9}"
      f"{'TPR':>7}{'acc':>8}{'AUC':>8}")
print("  " + "-" * 64)
tot_tp = tot_fp = tot_tn = tot_fn = 0
tot_n_ben = tot_n_atk = 0
auc_pool_y, auc_pool_s = [], []
for d in range(1, N + 1):
    r1 = read_mse(f"output/device{d}_normal_r1_results.csv")
    r2 = read_mse(f"output/device{d}_normal_r2_results.csv")
    atk = read_mse(f"output/device{d}_attack_results.csv")
    thr = median_mad(r1)
    fp = sum(1 for v in r2 if v > thr)
    tp = sum(1 for v in atk if v > thr)
    tn = len(r2) - fp
    fn = len(atk) - tp
    fpr = fp / len(r2) * 100
    tpr = tp / len(atk) * 100
    acc = (tp + tn) / (len(r2) + len(atk))
    y = [0] * (len(r1) + len(r2)) + [1] * len(atk)
    s = r1 + r2 + atk
    auc = roc_auc(y, s)
    tot_tp += tp; tot_fp += fp; tot_tn += tn; tot_fn += fn
    tot_n_ben += len(r2); tot_n_atk += len(atk)
    auc_pool_y += y; auc_pool_s += s
    print(f"  device{d:<5}{len(r1):>8}{len(r2):>8}{thr:>10.4f}{fpr:>8.1f}%"
          f"{tpr:>6.1f}%{acc:>8.4f}{auc:>8.4f}")

n = tot_tp + tot_fp + tot_tn + tot_fn
acc = (tot_tp + tot_tn) / n
prec = tot_tp / (tot_tp + tot_fp)
rec = tot_tp / (tot_tp + tot_fn)
f1 = 2 * prec * rec / (prec + rec)
auc = roc_auc(auc_pool_y, auc_pool_s)
print("  " + "-" * 64)
print(f"  OVERALL  benign(total)={tot_n_ben}  attack={tot_n_atk}")
print(f"  OVERALL  thr calibration: run1 benign only; FPR from held-out run2")
print(f"  OVERALL  acc={acc:.4f}  TPR={rec:.4f}  FPR={tot_fp/tot_n_ben:.4f}  "
      f"prec={prec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
print(f"  OVERALL  TP={tot_tp} FP={tot_fp} TN={tot_tn} FN={tot_fn}")
