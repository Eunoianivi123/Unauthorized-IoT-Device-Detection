#!/usr/bin/env python3
"""Sequential detector comparison: CUSUM (deployed) vs EWMA vs Shewhart.

Justifies the choice of the one-sided CUSUM decision rule in
identification_engine.py by running all three detectors over the SAME
per-packet anomaly flags (from *_results.csv, scored by the deployed
all-in-one autoencoder + median+2.5*MAD thresholds).

Detectors (all one-sided, alarm on a packet stream):
  CUSUM    : S = max(0, S + logLR); alarm at S >= A          [production rule]
  EWMA     : z = lam*x + (1-lam)*z;  alarm at z >= h_ewma
  Shewhart : non-overlapping windows of n packets; alarm when a window's
             anomaly proportion >= UCL

Calibration policy (identical footing):
  - CUSUM uses the production A=250 (already calibrated to zero alarms on
    benign r1; see identification_engine.py header).
  - EWMA / Shewhart thresholds are set per device to the smallest value with
    zero alarms on that device's benign r1 stream, times a small safety
    margin -- mirroring how A=250 was derived.
Evaluation streams: benign r2 (FPR), attack, attack_hard/stealth, stress.

Run inside the pipeline image:
  docker run --rm -v "$PWD:/work" -w /work nbaiot-pipeline:latest \
      python3 detector_comparison.py
"""
import argparse
import csv
import math
import os

import numpy as np
import pandas as pd

from sprt_accumulator import CUSUM

# ── Duplicated from identification_engine.py — keep in sync ──────────────
P_ATK = 0.3
CUSUM_A = 250.0
DEVICE_P_BEN = {
    1: 0.039, 2: 0.130, 3: 0.046, 4: 0.031, 5: 0.033,
    6: 0.043, 7: 0.058, 8: 0.043, 9: 0.046,
}


def read_flags(path):
    df = pd.read_csv(path)
    return df["anomaly"].astype(int).to_numpy()


class EWMA:
    """One-sided EWMA control chart over Bernoulli anomaly flags."""

    def __init__(self, p_ben, lam, h):
        self.p_ben = float(p_ben)
        self.lam = float(lam)
        self.h = float(h)

    def path(self, flags):
        z = self.p_ben
        out = np.empty(len(flags), dtype=np.float64)
        for i, f in enumerate(flags):
            z = self.lam * f + (1.0 - self.lam) * z
            out[i] = z
        return out

    def calibrate(self, cal_flags, margin=1.10, floor_mult=1.25):
        peak = float(self.path(cal_flags).max()) if len(cal_flags) else self.p_ben
        self.h = max(peak * margin, self.p_ben * floor_mult)
        return self.h

    def run(self, flags):
        hit = np.flatnonzero(self.path(flags) >= self.h)
        return int(hit[0]) + 1 if len(hit) else None


class Shewhart:
    """p-chart: non-overlapping windows, alarm when window proportion >= UCL."""

    def __init__(self, p_ben, window, ucl):
        self.p_ben = float(p_ben)
        self.n = int(window)
        self.ucl = float(ucl)

    def _props(self, flags):
        m = len(flags) // self.n
        if m == 0:
            return np.empty(0)
        return flags[: m * self.n].reshape(m, self.n).mean(axis=1)

    def calibrate(self, cal_flags, margin=1.05):
        props = self._props(np.asarray(cal_flags))
        peak = float(props.max()) if len(props) else 0.0
        floor = max(2.0 / self.n, 3.0 * self.p_ben)
        self.ucl = max(peak * margin, floor)
        return self.ucl

    def run(self, flags):
        props = self._props(np.asarray(flags))
        hit = np.flatnonzero(props >= self.ucl)
        return int(hit[0] + 1) * self.n if len(hit) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--p-atk", type=float, default=P_ATK)
    ap.add_argument("--A", type=float, default=CUSUM_A)
    ap.add_argument("--ewma-lambda", type=float, default=0.1)
    ap.add_argument("--window", type=int, default=100)
    args = ap.parse_args()

    streams = ("normal_r1", "normal_r2", "attack", "attack_hard", "stress")
    data = {}
    for d in range(1, 10):
        per = {}
        for s in streams:
            f = os.path.join(args.out_dir, f"device{d}_{s}_results.csv")
            per[s] = read_flags(f)
        data[d] = per

    detectors = ("cusum", "ewma", "shewhart")
    rows = []
    print("=" * 104)
    print("  Sequential detector comparison over identical per-packet anomaly flags")
    print(f"  CUSUM A={args.A:.0f} (production) | EWMA lam={args.ewma_lambda} "
          f"| Shewhart window={args.window}")
    print("  calibration: benign r1 only | FPR: held-out benign r2 (+stress robustness)")
    print("=" * 104)

    for d in range(1, 10):
        per = data[d]
        p_ben = DEVICE_P_BEN.get(d, 0.05)
        cusum = CUSUM(p_ben, args.p_atk, args.A)
        ewma = EWMA(p_ben, args.ewma_lambda, h=float("inf"))
        h_ewma = ewma.calibrate(per["normal_r1"])
        shew = Shewhart(p_ben, args.window, ucl=float("inf"))
        ucl = shew.calibrate(per["normal_r1"])

        print(f"\n  device{d}  (p_ben={p_ben}, h_ewma={h_ewma:.4f}, UCL={ucl:.4f})")
        print(f"    {'detector':<10}{'r2-FP':>7}{'atk-det':>9}{'lat@atk':>9}"
              f"{'hard-det':>10}{'lat@hard':>10}{'stress-FP':>11}")

        for name in detectors:
            det = {"cusum": lambda fl: cusum.run(fl),
                   "ewma": lambda fl: ewma.run(fl),
                   "shewhart": lambda fl: shew.run(fl)}[name]
            lat_r2 = det(per["normal_r2"])
            lat_atk = det(per["attack"])
            lat_hard = det(per["attack_hard"])
            lat_stress = det(per["stress"])

            fp_r2 = int(lat_r2 is not None)
            det_atk = int(lat_atk is not None)
            det_hard = int(lat_hard is not None)
            fp_stress = int(lat_stress is not None)

            fmt_lat = lambda v: str(v) if v is not None else "-"
            print(f"    {name:<10}{fp_r2:>7}{('YES' if det_atk else 'no'):>9}"
                  f"{fmt_lat(lat_atk):>9}{('YES' if det_hard else 'no'):>10}"
                  f"{fmt_lat(lat_hard):>10}{fp_stress:>11}")

            rows.append({
                "device": d, "detector": name,
                "alarm_normal_r2": fp_r2, "packets_to_alarm_r2": lat_r2 or "",
                "alarm_attack": det_atk, "packets_to_alarm_attack": lat_atk or "",
                "alarm_attack_hard": det_hard, "packets_to_alarm_attack_hard": lat_hard or "",
                "alarm_stress": fp_stress, "packets_to_alarm_stress": lat_stress or "",
            })

    # ── Overall summary ───────────────────────────────────────────────────
    print("\n" + "=" * 104)
    print("  OVERALL (9 devices)")
    print(f"    {'detector':<10}{'TPR@atk':>9}{'med-lat@atk':>13}{'worst-lat':>10}"
          f"{'TPR@hard':>10}{'med-lat@hard':>14}{'FPr2':>6}{'FPstress':>10}")
    summary = {}
    for name in detectors:
        rr = [r for r in rows if r["detector"] == name]
        tpr_a = sum(r["alarm_attack"] for r in rr)
        tpr_h = sum(r["alarm_attack_hard"] for r in rr)
        fpr_r2 = sum(r["alarm_normal_r2"] for r in rr)
        fps = sum(r["alarm_stress"] for r in rr)
        la = sorted(r["packets_to_alarm_attack"] for r in rr if r["alarm_attack"])
        lh = sorted(r["packets_to_alarm_attack_hard"] for r in rr if r["alarm_attack_hard"])
        med_a = int(np.median(la)) if la else None
        med_h = int(np.median(lh)) if lh else None
        worst_a = max(la) if la else None
        summary[name] = dict(tpr_a=tpr_a, med_a=med_a, worst_a=worst_a,
                             tpr_h=tpr_h, med_h=med_h, fpr=fpr_r2, fps=fps)
        print(f"    {name:<10}{tpr_a:>7}/9{str(med_a or '-'):>13}{str(worst_a or '-'):>10}"
              f"{tpr_h:>7}/9{str(med_h or '-'):>14}{fpr_r2:>6}{fps:>10}")

    csv_path = os.path.join(args.out_dir, "detector_comparison_results.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Per-stream results written to: {csv_path}")


if __name__ == "__main__":
    main()
