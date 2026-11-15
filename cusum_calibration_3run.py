#!/usr/bin/env python3
"""Honest CUSUM calibration + detector comparison + 3-run variance.

Answers the reviewer's two remaining points:

  Task 1  (a) CUSUM parameters (P_ATK, A) were selected on the same
              captures the device-level results were reported on.
          -> leave-one-run-out (LORO) nested calibration.  For each fold
             the *whole* calibration (MSE threshold, p_ben, P_ATK, A) is
             fitted on the two held-in repetitions and evaluated only on
             the held-out repetition.  The in-sample vs held-out gap is
             reported so the optimism is quantified, not hidden.
          (b) The parameter sweep and the CUSUM/EWMA/Shewhart comparison
              ran on pre-C2 flags.
          -> both are re-run here on flags from the DEPLOYED C2 model,
             read from the fresh feature CSVs (never the stale
             `*_results.csv` anomaly columns).
          (c) Detection delay was measured inconsistently (210 vs 2143).
          -> a single definition in cusum_eval.latency() is used by the
             sweep, the detector comparison and the spoof test alike.

  Task 2  Results were calibrated from a single run.
          -> 3 independent capture repetitions per device and mode; every
             headline number is reported as mean +/- std over repetitions.

Run inside the pipeline image:
  docker run --rm --entrypoint python3 -v "$PWD:/work" -w /work \
      nbaiot-pipeline:latest cusum_calibration_3run.py
"""
from __future__ import annotations

import argparse
import ast
import csv
import os

import numpy as np

import cusum_eval as ce

CAPTURE_DIR = "output/repeats"
MODEL_DIR = "new_models/winner"
CACHE = "output/repeats/.score_cache.npz"

# Pre-registered search grid.  Declared before looking at any held-out
# result, so the selection cannot be tuned on the evaluation data.
P_ATK_GRID = (0.2, 0.3, 0.5)
A_GRID = (50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 350.0, 500.0, 750.0, 1000.0)

PRODUCTION = dict(p_atk=0.3, A=250.0)


# ----------------------------------------------------------------------
def load_scores(model, scaler, capture_dir, cache_path, verbose=True):
    """Score every campaign capture with the C2 model.

    Cached as a single .npz.  The cache records WHICH files it was built
    from, so a cache produced from a partial campaign (e.g. a smoke test
    while captures are still running) is invalidated automatically rather
    than silently reused.
    """
    present = []
    for d in ce.DEVICES:
        for mode in (*ce.CALIB_MODES, ce.STRESS_MODE):
            for rep in (ce.REPS if mode != ce.STRESS_MODE else ("r1",)):
                p = ce.features_path(capture_dir, d, mode, rep)
                if os.path.exists(p):
                    present.append(f"device{d}_{mode}_{rep}")
    manifest = "|".join(present)

    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        if "manifest" in z.files and str(z["manifest"].item()) == manifest:
            # npz keys are the repr() of the (device, mode, rep) tuple
            data = {ast.literal_eval(k): z[k].item()
                    for k in z.files if k != "manifest"}
            if verbose:
                print(f"[*] score cache hit: {len(data)} captures")
            return data
        if verbose:
            print("[*] score cache stale (capture set changed) -- rescoring")

    data = {}
    missing = []
    for d in ce.DEVICES:
        for mode in (*ce.CALIB_MODES, ce.STRESS_MODE):
            for rep in (ce.REPS if mode != ce.STRESS_MODE else ("r1",)):
                p = ce.features_path(capture_dir, d, mode, rep)
                if not os.path.exists(p):
                    missing.append(os.path.basename(p))
                    continue
                df = ce.score_capture(p, model, scaler)
                data[(d, mode, rep)] = {
                    "mse": df["mse"].to_numpy(),
                    "ts": df["timestamp"].to_numpy(),
                }
                if verbose:
                    print(f"    scored d{d} {mode} {rep}: {len(df)} rows", flush=True)

    if verbose and missing:
        print(f"[*] {len(missing)} capture(s) absent, e.g. {', '.join(missing[:4])}"
              f"{' ...' if len(missing) > 4 else ''}")

    if cache_path:
        np.savez_compressed(cache_path, manifest=np.array(manifest),
                            **{repr(k): v for k, v in data.items()})
    return data


def missing_runs(data):
    """(device, mode) pairs that do not have all 3 repetitions."""
    bad = []
    for d in ce.DEVICES:
        for mode in ce.CALIB_MODES:
            have = [r for r in ce.REPS if (d, mode, r) in data]
            if len(have) < len(ce.REPS):
                bad.append((d, mode, have))
    return bad


# ----------------------------------------------------------------------
# Calibration fitted on the held-IN repetitions only
# ----------------------------------------------------------------------
def fit_calibration(data, held_out_rep, devices=ce.DEVICES):
    """Median+2.5*MAD threshold and p_ben from the benign held-in runs."""
    per_device = {}
    for d in devices:
        parts = [data[(d, "normal", r)]["mse"]
                 for r in ce.REPS if r != held_out_rep and (d, "normal", r) in data]
        if not parts:
            continue
        pool = np.concatenate(parts)
        thr = ce.median_mad_threshold(pool)
        p_ben = float(np.mean(pool > thr))
        per_device[d] = {"threshold": thr, "p_ben": max(p_ben, 1e-4), "n_benign_rows": int(len(pool))}
    return per_device


def _fmt(v, spec="-.3f"):
    return format(v, spec) if v is not None else "-"


# ----------------------------------------------------------------------
def select_params(data, cal, reps, devices=ce.DEVICES):
    """Pre-registered selection rule, fitted on calibration captures only.

    Priority:
      1. fewest false alarms on the calibration benign captures;
      2. highest calibration TPR;
      3. the SMALLEST A that still keeps a 2x safety margin over the
         largest CUSUM statistic ever reached on the calibration benign
         captures (A >= 2 * max_benign_peak_S).

    Rule 3 matters: without it the search would always collapse onto the
    smallest A in the grid, which is knife-edge tuning against the very
    captures it is being scored on.  The margin is a fixed constant, not a
    value fitted to the data, and it is applied identically to every grid
    cell so the comparison stays fair.
    """
    best = None
    rows = []
    for p_atk in P_ATK_GRID:
        for A in A_GRID:
            per = {}
            for mode in ce.CALIB_MODES:
                per.update(collect(data, cal, reps, mode, p_atk, A, devices))
            s = summarise(per, "calib")
            ben = [v for (d, m, r), v in per.items() if m == "normal"]
            peak_ben = max((v.peak_S for v in ben), default=0.0)
            margin_ok = 1 if (ben and A >= 2.0 * peak_ben) else 0

            n_ben = s["n_benign"]
            fp = int(round((1.0 - s["fpr_benign"]) * n_ben)) if s["fpr_benign"] is not None else 0
            tp = (int(round((s["tpr_attack"] or 0.0) * s["n_attack"]))
                  + int(round((s["tpr_attack_hard"] or 0.0) * s["n_attack_hard"])))
            rows.append((p_atk, A, fp, tp, margin_ok, peak_ben, s))
            key = (fp, -tp, -margin_ok, A)
            if best is None or key < best[0]:
                best = (key, p_atk, A)
    return best[1], best[2], rows


def collect(data, cal, reps, mode, p_atk, A, devices=ce.DEVICES):
    """Alarm/Latency for every (device, rep) of `mode` in `reps`."""
    out = {}
    for d in devices:
        if d not in cal:
            continue
        for r in reps:
            key = (d, mode, r)
            if key not in data:
                continue
            mse = data[key]["mse"]
            ts = data[key]["ts"]
            flags = ce.flags_from(mse, cal[d]["threshold"])
            cu = ce.CUSUM(cal[d]["p_ben"], p_atk, A)
            out[key] = ce.latency(cu.path(flags), flags, ts, A)
    return out


def summarise(per_capture, label):
    """TPR / FPR / delay aggregates over a set of captures."""
    atk = [v for (d, m, r), v in per_capture.items() if m == "attack"]
    hard = [v for (d, m, r), v in per_capture.items() if m == "attack_hard"]
    ben = [v for (d, m, r), v in per_capture.items() if m == "normal"]

    def tpr(lst):
        return (sum(1 for v in lst if v.alarm_index is not None) / len(lst)) if lst else None

    def delay(lst):
        vals = [v.delay_packets for v in lst if v.alarm_index is not None]
        return ce.mean_std(vals)

    return {
        "label": label,
        "n_benign": len(ben), "n_attack": len(atk), "n_attack_hard": len(hard),
        "tpr_attack": tpr(atk), "tpr_attack_hard": tpr(hard),
        "fpr_benign": tpr(ben),
        "delay_attack_mean": delay(atk)[0], "delay_attack_std": delay(atk)[1],
        "delay_hard_mean": delay(hard)[0], "delay_hard_std": delay(hard)[1],
    }



# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", default=CAPTURE_DIR)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--cache", default=CACHE)
    args = ap.parse_args()

    print("=" * 100)
    print("  3-repetition CUSUM calibration study -- DEPLOYED C2 model")
    print(f"  captures : {args.capture_dir}")
    print(f"  model    : {args.model_dir}")
    print("=" * 100)

    model, scaler = ce.load_model(args.model_dir)
    data = load_scores(model, scaler, args.capture_dir, args.cache)

    bad = missing_runs(data)
    if bad:
        print("\n[WARN] incomplete repetitions (device, mode, reps present):")
        for d, mode, have in bad:
            print(f"   device{d} {mode}: {have}")
        print("   -> results below cover only the complete repetitions.\n")

    os.makedirs(args.capture_dir, exist_ok=True)
    per_run_rows = []
    sweep_rows = []
    nested_rows = []
    det_rows = []

    for fold_idx, held in enumerate(ce.REPS):
        held_in = [r for r in ce.REPS if r != held]
        cal = fit_calibration(data, held)
        print("\n" + "#" * 100)
        print(f"#  FOLD {fold_idx+1}/3   held-out repetition = {held}   "
              f"(calibrated on {', '.join(held_in)})")
        print("#" * 100)
        print(f"  {'dev':<5}{'thr(med+2.5MAD)':>18}{'p_ben':>9}{'n_ben_rows':>12}")
        for d in ce.DEVICES:
            if d in cal:
                c = cal[d]
                print(f"  {d:<5}{c['threshold']:>18.6f}{c['p_ben']:>9.4f}{c['n_benign_rows']:>12}")

        # ---- calibration-set performance (OPTIMISTIC, reported for the gap)
        p_star, A_star, grid_rows = select_params(data, cal, held_in)
        peak_ben_grid = [g[5] for g in grid_rows]
        peak_ben = max(peak_ben_grid) if peak_ben_grid else 0.0
        print(f"\n  [nested selection]  P_ATK* = {p_star}   A* = {A_star:g}"
              f"   (max benign peak S on held-in captures = {peak_ben:.1f},"
              f" safety margin requires A >= {2*peak_ben:.1f})")

        cal_star = summarise({**collect(data, cal, held_in, "normal", p_star, A_star),
                              **collect(data, cal, held_in, "attack", p_star, A_star),
                              **collect(data, cal, held_in, "attack_hard", p_star, A_star)},
                             "calib_star")
        cal_prod = summarise({**collect(data, cal, held_in, "normal", PRODUCTION["p_atk"], PRODUCTION["A"]),
                              **collect(data, cal, held_in, "attack", PRODUCTION["p_atk"], PRODUCTION["A"]),
                              **collect(data, cal, held_in, "attack_hard", PRODUCTION["p_atk"], PRODUCTION["A"])},
                             "calib_prod")

        # ---- held-out (HONEST) performance, same params
        out_star = summarise({**collect(data, cal, [held], "normal", p_star, A_star),
                              **collect(data, cal, [held], "attack", p_star, A_star),
                              **collect(data, cal, [held], "attack_hard", p_star, A_star)},
                             "heldout_star")
        out_prod = summarise({**collect(data, cal, [held], "normal", PRODUCTION["p_atk"], PRODUCTION["A"]),
                              **collect(data, cal, [held], "attack", PRODUCTION["p_atk"], PRODUCTION["A"]),
                              **collect(data, cal, [held], "attack_hard", PRODUCTION["p_atk"], PRODUCTION["A"])},
                             "heldout_prod")

        print(f"\n  {'evaluation':<26}{'TPR_atk':>9}{'TPR_hard':>10}{'FPR_ben':>9}"
              f"{'del_atk(mean+/-sd)':>26}{'del_hard(mean+/-sd)':>26}")

        def show(name, s):
            da = ("-" if s["delay_attack_mean"] is None
                  else f"{s['delay_attack_mean']:.0f}+/-{(s['delay_attack_std'] or 0):.0f}")
            dh = ("-" if s["delay_hard_mean"] is None
                  else f"{s['delay_hard_mean']:.0f}+/-{(s['delay_hard_std'] or 0):.0f}")
            print(f"  {name:<26}{_fmt(s['tpr_attack']):>9}{_fmt(s['tpr_attack_hard']):>10}"
                  f"{_fmt(s['fpr_benign']):>9}{da:>26}{dh:>26}")

        show(f"calib (in-sample) A*={A_star:g}", cal_star)
        show(f"calib (in-sample) A=250", cal_prod)
        show(f"HELD-OUT  A*={A_star:g}", out_star)
        show(f"HELD-OUT  A=250", out_prod)

        for s, kind, pa, A in ((cal_star, "calibration_insample", p_star, A_star),
                               (cal_prod, "calibration_insample", PRODUCTION["p_atk"], PRODUCTION["A"]),
                               (out_star, "heldout", p_star, A_star),
                               (out_prod, "heldout", PRODUCTION["p_atk"], PRODUCTION["A"])):
            nested_rows.append({"fold": fold_idx + 1, "held_out_rep": held, "kind": kind,
                                "p_atk": pa, "A": A, "max_benign_peak_S": round(peak_ben, 2),
                                **{k: v for k, v in s.items() if k != "label"}})

        # ---- full grid on the HELD-OUT repetition (re-run sweep on C2)
        print(f"\n  HELD-OUT grid (re-run of the parameter sweep on the C2 model), rep {held}:")
        print(f"    {'p_atk':>6}{'A':>7}{'TPR_atk':>9}{'TPR_hard':>10}{'FPR_ben':>9}"
              f"{'del_atk_med':>13}{'n_atk':>7}")
        for p_atk in P_ATK_GRID:
            for A in A_GRID:
                s = summarise({**collect(data, cal, [held], "normal", p_atk, A),
                               **collect(data, cal, [held], "attack", p_atk, A),
                               **collect(data, cal, [held], "attack_hard", p_atk, A)}, "heldout")
                dl = [v.delay_packets for (d, m, r), v in
                      collect(data, cal, [held], "attack", p_atk, A).items()
                      if v.alarm_index is not None]
                dl_h = [v.delay_packets for (d, m, r), v in
                        collect(data, cal, [held], "attack_hard", p_atk, A).items()
                        if v.alarm_index is not None]
                med = float(np.median(dl)) if dl else None
                medh = float(np.median(dl_h)) if dl_h else None
                print(f"    {p_atk:>6.1f}{A:>7.0f}{_fmt(s['tpr_attack']):>9}"
                      f"{_fmt(s['tpr_attack_hard']):>10}{_fmt(s['fpr_benign']):>9}"
                      f"{_fmt(med, '.0f'):>13}{s['n_attack']:>7}")
                sweep_rows.append({
                    "fold": fold_idx + 1, "held_out_rep": held, "p_atk": p_atk, "A": A,
                    "tpr_attack": s["tpr_attack"], "tpr_attack_hard": s["tpr_attack_hard"],
                    "fpr_benign": s["fpr_benign"], "n_attack": s["n_attack"],
                    "n_attack_hard": s["n_attack_hard"], "n_benign": s["n_benign"],
                    "median_delay_attack": med, "median_delay_attack_hard": medh,
                })

        # ---- detector comparison on the SAME C2 flags
        print(f"\n  Detector comparison on the C2 flags (calibrated on {', '.join(held_in)}, "
              f"evaluated on {held}):")
        print(f"    {'dev':<5}{'detector':<10}{'r2/ben FP':>11}{'atk det':>9}{'delay':>8}"
              f"{'hard det':>10}{'hdelay':>8}{'stress FP':>11}")
        for d in ce.DEVICES:
            if d not in cal:
                continue
            p_ben = cal[d]["p_ben"]
            thr = cal[d]["threshold"]
            rows_d = {}
            for mode, rep in (("normal", held), ("attack", held), ("attack_hard", held)):
                key = (d, mode, rep)
                rows_d[mode] = (ce.flags_from(data[key]["mse"], thr), data[key]["ts"]) if key in data else None

            # CUSUM: A comes from the nested selection; EWMA/Shewhart limits
            # are calibrated on the held-IN benign flags only -- identical
            # footing for all three detectors.
            ccu = ce.CUSUM(p_ben, p_star, A_star)
            cal_flags = {
                mode: (np.concatenate(parts) if (parts := [
                    ce.flags_from(data[(d, mode, r)]["mse"], thr)
                    for r in held_in if (d, mode, r) in data]) else
                       np.empty(0, dtype=np.int8))
                for mode in ce.CALIB_MODES
            }
            ewm = ce.EWMA(p_ben, 0.1)
            ewm.calibrate(cal_flags["normal"])
            shew = ce.Shewhart(p_ben, 100)
            shew.calibrate(cal_flags["normal"])

            det = {
                "cusum": (ccu, A_star),
                "ewma": (ewm, ewm.h),
                "shewhart": (shew, shew.ucl),
            }
            results = {}
            for name, (obj, lim) in det.items():
                res = {}
                for mode in ("normal", "attack", "attack_hard"):
                    r = rows_d.get(mode)
                    if r is None:
                        res[mode] = None
                        continue
                    f, ts = r
                    res[mode] = ce.latency(obj.path(f), f, ts, lim)
                sk = (d, ce.STRESS_MODE, "r1")
                if sk in data:
                    f = ce.flags_from(data[sk]["mse"], thr)
                    res["stress"] = ce.latency(obj.path(f), f, data[sk]["ts"], lim)
                else:
                    res["stress"] = None
                results[name] = (res, lim)

            for name, (out, lim) in results.items():
                fp_ben = int(out["normal"] and out["normal"].alarm_index is not None)
                a = out["attack"]; h = out["attack_hard"]; st = out["stress"]
                print(f"    {d:<5}{name:<10}{fp_ben:>11}"
                      f"{('YES' if a and a.alarm_index else 'no'):>9}"
                      f"{(a.delay_packets if a and a.alarm_index else '-'):>8}"
                      f"{('YES' if h and h.alarm_index else 'no'):>10}"
                      f"{(h.delay_packets if h and h.alarm_index else '-'):>8}"
                      f"{int(bool(st and st.alarm_index)):>11}")
                det_rows.append({
                    "fold": fold_idx + 1, "held_out_rep": held, "device": d, "detector": name,
                    "limit": round(lim, 4),
                    "alarm_benign": fp_ben,
                    "alarm_attack": int(bool(a and a.alarm_index)),
                    "delay_attack": a.delay_packets if a and a.alarm_index else None,
                    "alarm_attack_hard": int(bool(h and h.alarm_index)),
                    "delay_attack_hard": h.delay_packets if h and h.alarm_index else None,
                    "alarm_stress": int(bool(st and st.alarm_index)),
                    "delay_stress": st.delay_packets if st and st.alarm_index else None,
                })

        # ---- per-repetition task-2 table (packet metrics + latency)
        for d in ce.DEVICES:
            if d not in cal:
                continue
            thr = cal[d]["threshold"]
            for mode in ce.CALIB_MODES:
                k = (d, mode, held)
                if k not in data:
                    continue
                mse = data[k]["mse"]; ts = data[k]["ts"]
                flags = ce.flags_from(mse, thr)
                cu = ce.CUSUM(cal[d]["p_ben"], p_star, A_star)
                L = ce.latency(cu.path(flags), flags, ts, A_star)
                per_run_rows.append({
                    "fold": fold_idx + 1, "held_out_rep": held, "device": d, "mode": mode,
                    "threshold": round(thr, 6), "p_ben": round(cal[d]["p_ben"], 4),
                    "p_atk": p_star, "A": A_star,
                    "rows": L.rows, "flags": L.flags,
                    "flag_rate": round(L.flags / L.rows, 4) if L.rows else None,
                    "cusum_alarm": int(L.alarm_index is not None),
                    "peak_S": L.peak_S,
                    "delay_packets": L.delay_packets,
                    "delay_seconds": (round(L.delay_seconds, 4)
                                      if L.delay_seconds is not None else None),
                    "packets_after_first_flag": L.packets_after_first_flag,
                    "packet_rate_pps": L.packet_rate_pps,
                })

    # ---- pooled packet-level metrics per fold (comment 8 metrics)
    print("\n" + "=" * 100)
    print("  POOLED per-repetition metrics (held-out repetition, C2 model, fold thresholds)")
    print("=" * 100)
    print(f"  {'fold':<6}{'held':<7}{'Prec':>9}{'Rec':>9}{'F1':>9}{'Acc':>9}{'AUC':>9}{'FPR':>9}{'TPR':>9}")
    pooled_rows = []
    for fold_idx, held in enumerate(ce.REPS):
        cal = fit_calibration(data, held)
        neg, pos, fneg, fpos = [], [], [], []
        for d in ce.DEVICES:
            if d not in cal:
                continue
            thr = cal[d]["threshold"]
            for mode, store_m, store_f in (("normal", neg, fneg), ("attack", pos, fpos)):
                k = (d, mode, held)
                if k not in data:
                    continue
                m = data[k]["mse"]
                store_m.append(m)
                store_f.append(ce.flags_from(m, thr))
        if not neg or not pos:
            continue
        neg = np.concatenate(neg); pos = np.concatenate(pos)
        fneg = np.concatenate(fneg).astype(bool); fpos = np.concatenate(fpos).astype(bool)
        labels = np.concatenate([np.zeros(len(fneg), bool), np.ones(len(fpos), bool)])
        m = ce.binary_metrics(np.concatenate([fneg, fpos]), labels)
        auc = ce.roc_auc(neg, pos)
        print(f"  {fold_idx+1:<6}{held:<7}{m['precision']:>9.4f}{m['recall']:>9.4f}"
              f"{m['f1']:>9.4f}{m['accuracy']:>9.4f}{auc:>9.4f}{m['fpr']:>9.4f}{m['tpr']:>9.4f}")
        pooled_rows.append({"fold": fold_idx + 1, "held_out_rep": held, **m, "auc": auc})

    for key in ("precision", "recall", "f1", "accuracy", "auc", "fpr", "tpr"):
        mu, sd = ce.mean_std([r[key] for r in pooled_rows])
        if mu is not None:
            print(f"    {key:<10} mean = {mu:.4f}   std = {sd:.4f}")

    # ---- write CSVs
    def dump(path, rows):
        if not rows:
            print(f"  [WARN] no rows for {os.path.basename(path)} -- NOT written. "
                  f"This table is empty, not a crash; check missing_runs() above.")
            return
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {path}  ({len(rows)} rows)")

    print("\n" + "=" * 100)
    dump(os.path.join(args.capture_dir, "cusum_nested_calibration.csv"), nested_rows)
    dump(os.path.join(args.capture_dir, "cusum_param_sweep_heldout.csv"), sweep_rows)
    dump(os.path.join(args.capture_dir, "detector_comparison_c2.csv"), det_rows)
    dump(os.path.join(args.capture_dir, "run_variance_3run.csv"), per_run_rows)
    dump(os.path.join(args.capture_dir, "pooled_metrics_3run.csv"), pooled_rows)


if __name__ == "__main__":
    main()
