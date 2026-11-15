#!/usr/bin/env python3
"""Fix the three calibration-validity issues on the EXISTING canonical captures.

Why this exists
---------------
The 3-repetition campaign (output/repeats/) is still running, but all three
reviewer points can already be answered on the captures that already exist in
output/: each device has normal_r1, normal_r2, attack, attack_hard and stress.
That is enough for a genuine 2-fold leave-one-run-out split:

    fold A : calibrate on normal_r1  -> test on normal_r2
    fold B : calibrate on normal_r2  -> test on normal_r1

so every benign capture is scored with parameters derived from a *different*
benign capture. When the 3-repetition campaign finishes, cusum_calibration_3run.py
supersedes this with 3 folds and 3 repetitions; this script is the interim
version and uses exactly the same code path (cusum_eval) and the same unified
delay definition.

The three fixes
---------------
(a) in-sample parameter selection  -> 2-fold LORO; in-sample and held-out
    numbers are both reported so the optimism is quantified.
(b) pre-C2 flags                  -> every flag here is recomputed from the
    feature CSVs with the deployed C2 model. The `anomaly` column of the
    stale *_results.csv files is never read.
(c) inconsistent delay            -> cusum_eval.latency() only, for CUSUM, EWMA
    and Shewhart alike: packets from the start of the capture.

Run inside the pipeline image:
  docker run --rm --entrypoint python3 -v "$PWD:/work" -w /work \
      nbaiot-pipeline:latest cusum_fix_report.py
"""
from __future__ import annotations

import ast
import csv
import os

import numpy as np

import cusum_eval as ce

CAPTURE_DIR = "output"
MODEL_DIR = "new_models/winner"
OUT_DIR = "results"
CACHE = os.path.join(OUT_DIR, ".fix_score_cache.npz")

P_ATK_GRID = (0.2, 0.3, 0.5)
A_GRID = (50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 350.0, 500.0, 750.0, 1000.0)
PRODUCTION = (0.3, 250.0)
SAFETY_MARGIN = 2.0        # fixed, not fitted

BENIGN_REPS = ("r1", "r2")
ATTACK_MODES = ("attack", "attack_hard")


def path_for(device: int, mode: str, rep: str | None = None) -> str:
    tag = f"_{rep}" if rep else ""
    return os.path.join(CAPTURE_DIR, f"device{device}_{mode}{tag}_capture_features.csv")


# ----------------------------------------------------------------------
def load_scores(model, scaler, verbose=True):
    """Score every canonical capture with the deployed C2 model.

    Cache records WHICH files it was built from, so a partial cache is
    invalidated rather than silently reused.
    """
    wanted = []
    for d in ce.DEVICES:
        for rep in BENIGN_REPS:
            wanted.append((d, "normal", rep, path_for(d, "normal", rep)))
        for m in (*ATTACK_MODES, ce.STRESS_MODE):
            wanted.append((d, m, None, path_for(d, m)))
    wanted = [w for w in wanted if os.path.exists(w[3])]
    manifest = "|".join(w[3] for w in wanted)

    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if "manifest" in z.files and str(z["manifest"].item()) == manifest:
            data = {ast.literal_eval(k): z[k].item() for k in z.files if k != "manifest"}
            if verbose:
                print(f"[*] score cache hit: {len(data)} captures")
            return data
        if verbose:
            print("[*] score cache stale -- rescoring")

    data = {}
    for d, mode, rep, p in wanted:
        df = ce.score_capture(p, model, scaler)
        data[(d, mode, rep)] = {"mse": df["mse"].to_numpy(),
                                "ts": df["timestamp"].to_numpy()}
        if verbose:
            print(f"    scored d{d} {mode}{rep or ''}: {len(df)} rows", flush=True)
    np.savez_compressed(CACHE, manifest=np.array(manifest),
                        **{repr(k): v for k, v in data.items()})
    return data


# ----------------------------------------------------------------------
def fit_calibration(data, device, cal_rep):
    thr = ce.median_mad_threshold(data[(device, "normal", cal_rep)]["mse"])
    flags = ce.flags_from(data[(device, "normal", cal_rep)]["mse"], thr)
    p_ben = float(np.mean(flags))
    return {"threshold": thr, "p_ben": max(p_ben, 1e-4),
            "cal_flags": flags, "cal_rows": int(len(flags))}


def select_A(cal, p_atk, grid=A_GRID):
    """Pre-registered rule: smallest grid A keeping a 2x margin over the
    largest CUSUM statistic reached on the calibration benign capture."""
    cus = ce.CUSUM(p_ben=cal["p_ben"], p_atk=p_atk)
    peak = cus.peak(cus.path(cal["cal_flags"]))
    for A in grid:
        if A >= SAFETY_MARGIN * peak:
            return A, peak
    return grid[-1], peak


def evaluate(cap, cal, p_atk, A, detector="cusum", ewma_lam=0.1, window=100):
    """Run one detector on one capture with calibration-derived parameters."""
    mse, ts = cap["mse"], cap["ts"]
    flags = ce.flags_from(mse, cal["threshold"])
    if detector == "cusum":
        obj, limit = ce.CUSUM(p_ben=cal["p_ben"], p_atk=p_atk, A=A), A
    elif detector == "ewma":
        obj = ce.EWMA(p_ben=cal["p_ben"], lam=ewma_lam)
        limit = obj.calibrate(cal["cal_flags"])
    elif detector == "shewhart":
        obj = ce.Shewhart(p_ben=cal["p_ben"], window=window)
        limit = obj.calibrate(cal["cal_flags"])
    else:
        raise ValueError(detector)
    lat = ce.latency(obj.path(flags), flags, ts, limit)
    d = lat.as_dict()
    d["limit"] = float(limit)
    d["flag_rate"] = float(np.mean(flags))
    d["threshold"] = cal["threshold"]
    d["p_ben"] = cal["p_ben"]
    d["p_atk"] = p_atk if detector == "cusum" else None
    d["A"] = A if detector == "cusum" else None
    return d


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model, scaler = ce.load_model(MODEL_DIR)
    data = load_scores(model, scaler)
    if not data:
        raise SystemExit("[FATAL] no captures scored")

    folds = [("r1", "r2"), ("r2", "r1")]        # (calibration rep, held-out rep)
    per_dev, sweep_rows, det_rows, nested = [], [], [], []

    for cal_rep, held_rep in folds:
        cals = {d: fit_calibration(data, d, cal_rep) for d in ce.DEVICES
                if (d, "normal", cal_rep) in data}

        for d, cal in cals.items():
            A_star, peak = select_A(cal, PRODUCTION[0])
            cus = ce.CUSUM(p_ben=cal["p_ben"], p_atk=PRODUCTION[0])
            p_atk_star = PRODUCTION[0]

            # (a) in-sample vs held-out, at the deployed parameters
            for kind, rep in (("in_sample", cal_rep), ("held_out", held_rep)):
                ev = evaluate(data[(d, "normal", rep)], cal, *PRODUCTION)
                nested.append({
                    "cal_rep": cal_rep, "held_out_rep": held_rep, "device": d,
                    "kind": kind, "p_atk": PRODUCTION[0], "A": PRODUCTION[1],
                    "threshold": cal["threshold"], "p_ben": cal["p_ben"],
                    "max_benign_peak_S": peak, "alarm": ev["alarm_index"] is not None,
                    "peak_S": ev["peak_S"], "rows": ev["rows"],
                    "flag_rate": ev["flag_rate"],
                })

            # (b)+(c) held-out detection with the production and the selected A
            for A, tag in ((PRODUCTION[1], "production"), (A_star, "selected")):
                for mode, rep in [("attack", None), ("attack_hard", None),
                                  (ce.STRESS_MODE, None), ("normal", held_rep)]:
                    key = (d, mode, rep)
                    if key not in data:
                        continue
                    ev = evaluate(data[key], cal, PRODUCTION[0], A)
                    per_dev.append({
                        "fold_cal": cal_rep, "held_out_rep": held_rep, "device": d,
                        "mode": mode, "params": tag, "p_atk": PRODUCTION[0], "A": A,
                        "threshold": cal["threshold"], "p_ben": cal["p_ben"],
                        "alarm": ev["alarm_index"] is not None,
                        "alarm_index": ev["alarm_index"],
                        "delay_packets": ev["delay_packets"],
                        "delay_seconds": ev["delay_seconds"],
                        "packets_after_first_flag": ev["packets_after_first_flag"],
                        "peak_S": ev["peak_S"], "flag_rate": ev["flag_rate"],
                        "packet_rate_pps": ev["packet_rate_pps"], "rows": ev["rows"],
                    })

            # (b) the parameter sweep, re-run on C2 flags, held-out only
            for p_atk in P_ATK_GRID:
                cus = ce.CUSUM(p_ben=cal["p_ben"], p_atk=p_atk)
                for mode, rep in [("normal", held_rep), ("attack", None),
                                  ("attack_hard", None), (ce.STRESS_MODE, None)]:
                    key = (d, mode, rep)
                    if key not in data:
                        continue
                    cap = data[key]
                    flags = ce.flags_from(cap["mse"], cal["threshold"])
                    path = cus.path(flags)
                    for A in A_GRID:
                        lat = ce.latency(path, flags, cap["ts"], A)
                        sweep_rows.append({
                            "cal_rep": cal_rep, "held_out_rep": held_rep,
                            "device": d, "mode": mode, "p_atk": p_atk, "A": A,
                            "alarm": lat.alarm_index is not None,
                            "delay_packets": lat.delay_packets,
                        })

            # (b) detector comparison on identical C2 flags.
            # Every detector gets the SAME footing: its limit is 2x the largest
            # value its own statistic reaches on the calibration benign flags
            # (the same shape of rule as the CUSUM safety margin).  A separate
            # row pins CUSUM at the deployed A=250 for a like-for-like
            # reference, because "A chosen by rule" and "A=250" are not the
            # same thing and must not be presented as if they were.
            cal_flags = cal["cal_flags"]
            specs = [("cusum_2x", "cusum", None), ("cusum_deployed", "cusum", 250.0),
                     ("ewma_2x", "ewma", None), ("shewhart_2x", "shewhart", None)]
            for label, det, fixed_A in specs:
                if det == "cusum":
                    obj = ce.CUSUM(p_ben=cal["p_ben"], p_atk=PRODUCTION[0])
                    limit = fixed_A if fixed_A else SAFETY_MARGIN * obj.peak(obj.path(cal_flags))
                elif det == "ewma":
                    obj = ce.EWMA(p_ben=cal["p_ben"])
                    limit = max(SAFETY_MARGIN * float(obj.path(cal_flags).max()),
                                cal["p_ben"] * 1.25)
                else:
                    obj = ce.Shewhart(p_ben=cal["p_ben"])
                    limit = max(SAFETY_MARGIN * float(obj.path(cal_flags).max()),
                                2.0 / obj.n)
                for mode, rep in [("normal", held_rep), ("attack", None),
                                  ("attack_hard", None), (ce.STRESS_MODE, None)]:
                    key = (d, mode, rep)
                    if key not in data:
                        continue
                    cap = data[key]
                    flags = ce.flags_from(cap["mse"], cal["threshold"])
                    lat = ce.latency(obj.path(flags), flags, cap["ts"], limit)
                    det_rows.append({
                        "cal_rep": cal_rep, "held_out_rep": held_rep, "device": d,
                        "detector": label, "mode": mode, "limit": float(limit),
                        "alarm": lat.alarm_index is not None,
                        "delay_packets": lat.delay_packets,
                        "delay_seconds": lat.delay_seconds,
                    })

    for name, rows in (("cusum_fix_per_device.csv", per_dev),
                       ("cusum_fix_sweep.csv", sweep_rows),
                       ("cusum_fix_detectors.csv", det_rows),
                       ("cusum_fix_nested.csv", nested)):
        if not rows:
            print(f"[WARN] {name} empty -- not written")
            continue
        with open(os.path.join(OUT_DIR, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote results/{name}  ({len(rows)} rows)")
    print("\nNow run:  python3 render_cusum_fix.py")


if __name__ == "__main__":
    main()
