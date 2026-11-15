#!/usr/bin/env python3
"""Render the 3-repetition CUSUM calibration study as markdown tables.

Reads only the CSV artifacts produced by cusum_calibration_3run.py, so every
number in the paper-facing summary is generated, never transcribed by hand.

Run inside the pipeline image (or any python3 with the stdlib):
  python3 make_cusum_tables.py            # print all sections
  python3 make_cusum_tables.py --section optimism
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics as st
import sys

D = "output/repeats"

DEV_NAMES = {
    1: "Danmini Doorbell", 2: "Ecobee Thermostat", 3: "Ennio Doorbell",
    4: "Philips B120N10 Baby Monitor", 5: "Provision PT-737E Camera",
    6: "Provision PT-838 Camera", 7: "Samsung SNH-1011N Webcam",
    8: "SimpleHome XCS7-1002 Camera", 9: "SimpleHome XCS7-1003 Camera",
}
MODE_LABEL = {"normal": "benign", "attack": "attack", "attack_hard": "stealth (attack_hard)"}


def num(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def ms(vals, nd=3):
    """mean +/- std, ignoring missing entries."""
    v = [x for x in vals if x is not None and math.isfinite(x)]
    if not v:
        return "—"
    m = st.mean(v)
    s = st.stdev(v) if len(v) > 1 else 0.0
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def ratio(nums, dens, nd=3):
    """count/len rendered as a fraction with the raw counts."""
    nums = [x for x in nums if x is not None]
    if not dens:
        return "—"
    return f"{sum(nums)}/{len(dens)} ({sum(nums)/len(dens):.{nd}f})"


def fx(v, spec=".3f"):
    """Format a number, or an em-dash when the cell was not measured."""
    v = num(v)
    return "—" if v is None else format(v, spec)


# ----------------------------------------------------------------------
def sec_protocol():
    return """\
### Protocol (pre-registered before any held-out result was inspected)

| Item | Choice |
|---|---|
| Repetitions | 3 independent capture runs per device and mode (`r1`,`r2`,`r3`), 9 devices |
| Modes | benign (`normal`), `attack`, `attack_hard` (stealth) ×3; `stress` (FPR robustness) ×1 |
| Model | deployed all-in-one **C2** winner — flags recomputed from the fresh feature CSVs, never read from the stale `*_results.csv` `anomaly` columns |
| Threshold | `median + 2.5·MAD` of the **calibration** benign MSE only |
| `p_ben` | anomalous-packet rate over the **calibration** benign captures only |
| `(P_ATK, A)` search | grid `P_ATK ∈ {0.2,0.3,0.5}` × `A ∈ {50,75,100,150,200,250,350,500,750,1000}` |
| Selection rule | (1) fewest calibration-benign false alarms → (2) highest calibration TPR → (3) smallest `A` keeping a **2× safety margin** over the largest CUSUM statistic ever reached on the calibration benign captures |
| Cross-validation | **leave-one-run-out**: fit everything on 2 repetitions, evaluate on the 3rd; 3 folds, so every repetition is tested exactly once |
| Detection delay | one definition everywhere — packets from the **start of the capture** to the alarm (`alarm_index − 1`), plus seconds = packets ÷ measured capture rate |"""


def sec_calibration(nested, runs):
    out = ["### Per-fold calibration (fitted on the two held-in repetitions)", "",
           "| Device | Threshold (median+2.5·MAD), mean ± std | `p_ben`, mean ± std | Benign rows in calibration |",
           "|---|---|---|---|"]
    for d in range(1, 10):
        th = [num(r["threshold"]) for r in runs
              if r["device"] == str(d) and r["mode"] == "normal"]
        pb = [num(r["p_ben"]) for r in runs
              if r["device"] == str(d) and r["mode"] == "normal"]
        nb = [num(r["rows"]) for r in runs
              if r["device"] == str(d) and r["mode"] == "normal"]
        if not th:
            continue
        out.append(f"| {d} — {DEV_NAMES[d]} | {ms(th, 4)} | {ms(pb, 4)} | {ms(nb, 0)} |")
    out += ["",
            "> The benign-capture row count is the false-alarm **horizon**: it is the number of",
            "> consecutive benign packets the CUSUM is required not to alarm on. Device 2",
            "> (Ecobee thermostat) generates very little traffic, so its horizon is an order of",
            "> magnitude shorter than the cameras — see the caveats section.", ""]
    return "\n".join(out)


def sec_optimism(nested):
    """In-sample vs held-out, kept in two clearly separated tables so the
    optimistic numbers can never be read as the reported ones."""
    out = ["### Task 1a — in-sample vs held-out for the deployed parameters (`P_ATK=0.3`, `A=250`)", ""]
    for kind, tag in (("calibration_insample", "in-sample (the captures `A=250` was chosen on)"),
                      ("heldout", "held-out (the repetition never seen)")):
        out += [f"**{tag}**", "",
                "| Fold | Held-out rep | TPR attack | TPR stealth | FPR benign | Delay attack (mean±sd) |",
                "|---|---|---|---|---|---|"]
        for r in sorted((r for r in nested
                         if r["kind"] == kind and num(r["A"]) == 250.0
                         and num(r["p_atk"]) == 0.3), key=lambda x: x["fold"]):
            da = ("—" if num(r["delay_attack_mean"]) is None
                  else f"{num(r['delay_attack_mean']):.0f} ± {(num(r['delay_attack_std']) or 0):.0f}")
            out.append(f"| {r['fold']} | {r['held_out_rep']} | {fx(r['tpr_attack'])} | "
                       f"{fx(r['tpr_attack_hard'])} | {fx(r['fpr_benign'])} | {da} |")
        out.append("")
    return "\n".join(out)


def sec_sweep(sweep):
    out = ["### Task 1b — parameter sweep re-run on the deployed C2 model, held-out", "",
           "Every cell is evaluated **only** on the held-out repetition. `P_ATK = 0.3`",
           "(the production value) is shown; the deployed `A = 250` row is marked.", "",
           "| `P_ATK` | `A` | TPR attack | TPR stealth | FPR benign | median delay attack (pkts) | n attack | n stealth |",
           "|---|---|---|---|---|---|---|---|"]
    agg = {}
    for r in sweep:
        agg.setdefault((num(r["p_atk"]), num(r["A"])), []).append(r)
    for (pa, A), rs in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        ta = [num(x["tpr_attack"]) for x in rs]
        th = [num(x["tpr_attack_hard"]) for x in rs if num(x["tpr_attack_hard"]) is not None]
        fb = [num(x["fpr_benign"]) for x in rs]
        dl = [num(x["median_delay_attack"]) for x in rs if num(x["median_delay_attack"]) is not None]
        mark = " **(deployed)**" if (pa == 0.3 and A == 250) else ""
        out.append(f"| {pa:.1f}{mark} | {A:.0f} | {ms(ta)} | {ms(th) if th else '—'} | {ms(fb)} | "
                   f"{ms(dl, 0)} | {sum(int(x['n_attack']) for x in rs)//len(rs)} | "
                   f"{sum(int(x['n_attack_hard']) for x in rs)//len(rs)} |")
    out += ["", "*mean ± std over the 3 folds (each fold tests a different repetition).*", ""]
    return "\n".join(out)


def sec_detectors(det):
    out = ["### Task 1b — CUSUM vs EWMA vs Shewhart, re-run on the C2 flags", "",
           "All three detectors consume the **same** per-packet anomaly flags from the same",
           "fresh captures. CUSUM uses the nested-selected `A`; EWMA and Shewhart limits are",
           "calibrated on the held-in benign flags only — identical footing. The verdict",
           "statistic and the detection-delay definition are shared via `cusum_eval.latency()`.",
           "", "| Detector | TPR attack | TPR stealth | Benign false blocks | Stress false alarms | Median delay attack (pkts) |",
           "|---|---|---|---|---|---|"]
    for name in ("cusum", "ewma", "shewhart"):
        rs = [r for r in det if r["detector"] == name]
        if not rs:
            continue
        # A column is only counted where that mode was actually captured: the
        # delay field is written only when a capture for that mode existed, so
        # an empty delay means "not measured", not "missed".
        def tally(alarm_key, present_key):
            have = [r for r in rs if (r.get(present_key) or "").strip() != ""]
            return [int(r[alarm_key]) for r in have], have
        a, ah = tally("alarm_attack", "delay_attack")
        h, hh = tally("alarm_attack_hard", "delay_attack_hard")
        b, hb = tally("alarm_benign", "alarm_benign")
        s, hs = tally("alarm_stress", "delay_stress")
        dl = [num(r["delay_attack"]) for r in rs if num(r["delay_attack"]) is not None]
        out.append(
            f"| {name} | {ratio(a, ah)} | {ratio(h, hh)} | {ratio(b, hb)} | "
            f"{ratio(s, hs)} | {ms(dl, 0)} |")
    n_dev = len({r["device"] for r in det if r["device"]})
    n_fold = len({r["fold"] for r in det if r.get("fold")})
    total = n_dev * n_fold
    out += ["",
            f"Counts are out of {total} device×fold trials per column "
            f"({n_dev} devices × {n_fold} folds).",
            "Shewhart latency is quantised to its 100-packet window, so a `0` delay means the",
            "p-chart alarmed inside the first window.", ""]
    return "\n".join(out)


def sec_variance(runs):
    out = ["### Task 2 — variance across the 3 repetitions (per device)", "",
           "Each cell is mean ± std over the 3 independent capture runs. The CUSUM parameters",
           "for a given run are fitted on the *other* two, so no run is scored with parameters",
           "derived from itself.", ""]
    for mode in ("normal", "attack", "attack_hard"):
        out += [f"**{MODE_LABEL[mode]}**", "",
                "| Device | Threshold | `p_ben` | Flag rate % | CUSUM alarms | Peak S | Delay (packets) | Delay (s) | Capture rows |",
                "|---|---|---|---|---|---|---|---|---|"]
        for d in range(1, 10):
            rs = [r for r in runs if r["device"] == str(d) and r["mode"] == mode]
            if not rs:
                continue
            th = [num(r["threshold"]) for r in rs]
            pb = [num(r["p_ben"]) for r in rs]
            fr = [100 * num(r["flag_rate"]) for r in rs if num(r["flag_rate"]) is not None]
            al = [int(r["cusum_alarm"]) for r in rs]
            pk = [num(r["peak_S"]) for r in rs]
            dp = [num(r["delay_packets"]) for r in rs if num(r["delay_packets"]) is not None]
            ds = [num(r["delay_seconds"]) for r in rs if num(r["delay_seconds"]) is not None]
            rw = [num(r["rows"]) for r in rs]
            out.append(
                f"| {d} — {DEV_NAMES[d]} | {ms(th, 3)} | {ms(pb, 4)} | {ms(fr, 2)} | "
                f"{sum(al)}/{len(al)} | {ms(pk, 0)} | {ms(dp, 0)} | {ms(ds, 4)} | {ms(rw, 0)} |")
        out.append("")
    return "\n".join(out)


def sec_pooled(pooled):
    out = ["### Task 2 — pooled per-repetition metrics (comment-8 style, held-out repetitions)", "",
           "Pooled across all 9 devices within each repetition, scored with that fold's",
           "per-device thresholds. This is the mean ± std addendum next to — not a",
           "replacement for — the single-run Table VII numbers.", "",
           "| Metric | mean ± std over 3 repetitions |", "|---|---|"]
    for key, label in (("precision", "Precision"), ("recall", "Recall / TPR"),
                       ("f1", "F1"), ("accuracy", "Accuracy"), ("auc", "ROC-AUC"),
                       ("fpr", "FPR"), ("tpr", "TPR")):
        out.append(f"| {label} | {ms([num(r.get(key)) for r in pooled], 4)} |")
    out += ["", "| Fold | Held-out rep | Precision | Recall | F1 | Accuracy | ROC-AUC | FPR |",
            "|---|---|---|---|---|---|---|---|"]
    for r in sorted(pooled, key=lambda x: x["fold"]):
        out.append(f"| {r['fold']} | {r['held_out_rep']} | {num(r['precision']):.4f} | "
                   f"{num(r['recall']):.4f} | {num(r['f1']):.4f} | {num(r['accuracy']):.4f} | "
                   f"{num(r['auc']):.4f} | {num(r['fpr']):.4f} |")
    out.append("")
    return "\n".join(out)


SECTIONS = {
    "protocol": ("sec_protocol", lambda *_: sec_protocol()),
    "calibration": ("sec_calibration", lambda n, r: sec_calibration(n, r)),
    "optimism": ("sec_optimism", lambda n, r: sec_optimism(n)),
    "sweep": ("sec_sweep", lambda n, r: sec_sweep(r)),
    "detectors": ("sec_detectors", lambda n, r: sec_detectors(r)),
    "variance": ("sec_variance", lambda n, r: sec_variance(r)),
    "pooled": ("sec_pooled", lambda n, r: sec_pooled(r)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=D, help="directory holding the study CSVs")
    ap.add_argument("--section", choices=sorted(SECTIONS))
    args = ap.parse_args()
    d = args.dir

    def load(name):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            print(f"[WARN] missing {p}", file=sys.stderr)
            return []
        with open(p) as fh:
            return list(csv.DictReader(fh))

    nested = load("cusum_nested_calibration.csv")
    sweep = load("cusum_param_sweep_heldout.csv")
    det = load("detector_comparison_c2.csv")
    runs = load("run_variance_3run.csv")
    pooled = load("pooled_metrics_3run.csv")

    order = [args.section] if args.section else list(SECTIONS)
    for key in order:
        fn = SECTIONS[key][1]
        print(fn(nested, runs if key == "variance" or key == "calibration" else
                 (sweep if key == "sweep" else det if key == "detectors" else
                  pooled if key == "pooled" else [])))
        print()


if __name__ == "__main__":
    main()
