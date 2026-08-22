#!/usr/bin/env python3
"""Robust sequential decisioning for IoT attack detection.

Replaces the legacy "anomaly rate > 30% over the whole capture" rule with a
one-sided CUSUM over per-packet anomaly flags -- the change-detection form of
Wald's Sequential Probability Ratio Test (SPRT).

Setup
-----
For each device the autoencoder scores every packet with an MSE. A packet is
flagged anomalous when mse > threshold (median + 2.5 * MAD of benign r1 MSE).

Under H0 (benign) each packet is anomalous with probability p_ben; under H1
(attack) with probability p_atk. Every packet contributes a log-likelihood
ratio weight

    w(t) = log( P(flag(t)|attack) / P(flag(t)|benign) )
         = log(p_atk/p_ben)        if the packet is anomalous
           log((1-p_atk)/(1-p_ben)) otherwise

The CUSUM statistic is the one-sided cumulative sum

    S(n) = max(0, S(n-1) + w(n))

An ATTACK is declared the first time S(n) >= A. S is reset to zero on every
negative excursion, so a short burst of anomalous packets in otherwise benign
traffic does not alarm -- only sustained attack evidence accumulates across
the boundary. This is the sequential formulation used for change detection;
Wald proved the two-sided version minimises the expected number of packets for
a decision at given error rates, and the one-sided CUSUM (Page, 1954) is its
asymptotic relative-efficiency-optimal monitoring counterpart.

CLI (evaluation vs the legacy whole-capture anomaly-rate rule):
    python sprt_accumulator.py
    python sprt_accumulator.py --p-atk 0.3 --A 250
    python sprt_accumulator.py --devices 1 2 3 --out-dir output
"""
import argparse
import csv
import json
import math
import os

import numpy as np

LOG_2PI = math.log(2.0 * math.pi)


# ----------------------------------------------------------------------
# Density models (optional; used by fit_densities.py)
# ----------------------------------------------------------------------

class Density:
    """A fitted probability density model for per-packet MSE scores.

    Families (fitted with numpy only so it runs anywhere numpy runs):
      - gamma      shape k, scale theta        (good for right-skewed MSE)
      - lognormal  mu, sigma over log(mse)
    """

    __slots__ = ("family", "params")

    def __init__(self, family: str, params: dict):
        self.family = family
        self.params = dict(params)

    def to_dict(self) -> dict:
        return {"family": self.family, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict) -> "Density":
        return cls(d["family"], d["params"])

    def logpdf(self, x):
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, 1e-12, None)  # MSE is non-negative; guard zeros
        if self.family == "gamma":
            k = float(self.params["k"])
            th = float(self.params["theta"])
            lgk = math.lgamma(k)
            return (k - 1.0) * np.log(x) - x / th - k * math.log(th) - lgk
        if self.family == "lognormal":
            mu = float(self.params["mu"])
            sig = float(self.params["sigma"])
            lx = np.log(x)
            return -lx - math.log(sig) - 0.5 * LOG_2PI - ((lx - mu) ** 2) / (2.0 * sig * sig)
        raise ValueError(f"unknown density family: {self.family}")

    def pdf(self, x):
        return np.exp(self.logpdf(x))


def _fit_gamma(samples):
    m = float(samples.mean())
    v = float(samples.var(ddof=1)) if len(samples) > 1 else 0.0
    if m <= 0.0 or v <= 0.0:
        return None
    theta = v / m
    k = m * m / v
    if k <= 0.0 or theta <= 0.0 or not math.isfinite(k) or not math.isfinite(theta):
        return None
    return Density("gamma", {"k": k, "theta": theta})


def _fit_lognormal(samples):
    lx = np.log(np.clip(samples, 1e-12, None))
    mu = float(lx.mean())
    sig = float(lx.std(ddof=1)) if len(lx) > 1 else 0.0
    if sig <= 0.0 or not math.isfinite(mu) or not math.isfinite(sig):
        return None
    return Density("lognormal", {"mu": mu, "sigma": sig})


def fit_density(samples) -> Density:
    """Fit the best density for a sample list, choosing family by log-likelihood."""
    samples = np.asarray(list(samples), dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return Density("lognormal", {"mu": 0.0, "sigma": 1.0})
    best, best_ll = None, float("-inf")
    for cand in (_fit_gamma(samples), _fit_lognormal(samples)):
        if cand is None:
            continue
        ll = float(np.sum(cand.logpdf(samples)))
        if ll > best_ll:
            best, best_ll = cand, ll
    if best is None:  # degenerate data: fall back to a tiny-spread lognormal
        return Density("lognormal", {"mu": 0.0, "sigma": 1.0})
    return best


# ----------------------------------------------------------------------
# Bernoulli-flag CUSUM engine (the robust sequential detector)
# ----------------------------------------------------------------------

class CUSUM:
    """One-sided CUSUM over per-packet anomaly flags.

    p_ben and p_atk are the anomaly rates (probability a packet exceeds the
    device threshold) under the benign and attacked hypotheses. A is the
    alarm threshold: an ATTACK is declared when the accumulated evidence
    S(n) = max(0, S(n-1) + w(n)) first reaches A.
    """

    __slots__ = ("p_ben", "p_atk", "A", "w_anom", "w_normal")

    def __init__(self, p_ben: float, p_atk: float, A: float):
        p_ben = float(np.clip(p_ben, 1e-12, 1.0 - 1e-12))
        p_atk = float(np.clip(p_atk, 1e-12, 1.0 - 1e-12))
        if A <= 0.0:
            raise ValueError("alarm threshold A must be positive")
        self.p_ben = p_ben
        self.p_atk = p_atk
        self.A = float(A)
        self.w_anom = math.log(p_atk / p_ben)
        self.w_normal = math.log((1.0 - p_atk) / (1.0 - p_ben))

    def run(self, flags) -> int | None:
        """Consume anomaly flags; return the packet index (1-based) where S>=A
        was first reached, or None if the sequence ended without an alarm."""
        s = 0.0
        for i, f in enumerate(flags):
            s = max(0.0, s + (self.w_anom if f else self.w_normal))
            if s >= self.A:
                return i + 1
        return None

    def path(self, flags) -> np.ndarray:
        """Full S(n) trajectory (independent of A)."""
        s = 0.0
        out = np.empty(len(flags), dtype=np.float64)
        for i, f in enumerate(flags):
            s = max(0.0, s + (self.w_anom if f else self.w_normal))
            out[i] = s
        return out

    def peak(self, flags) -> float:
        """max S(n) -- the alarm statistic.  An alarm fires at threshold A iff
        peak >= A, because the CUSUM path does not depend on A."""
        return float(self.path(flags).max()) if len(flags) else 0.0

    def alarm_index(self, path: np.ndarray, A: float) -> int | None:
        """First packet index (1-based) where S>=A along a precomputed path."""
        hit = np.flatnonzero(path >= A)
        return int(hit[0]) + 1 if len(hit) else None


# ----------------------------------------------------------------------
# Data loading / threshold estimation
# ----------------------------------------------------------------------

def read_mse(path: str) -> np.ndarray:
    with open(path) as fh:
        vals = []
        for row in csv.DictReader(fh):
            try:
                vals.append(float(row["mse"]))
            except (TypeError, ValueError, KeyError):
                continue
    return np.asarray(vals, dtype=np.float64)


def load_densities(path: str) -> dict:
    """Load {device: {"benign": Density, "attack": Density}} from json."""
    with open(path) as fh:
        data = json.load(fh)
    models = {}
    for dev, classes in data.get("devices", {}).items():
        models[dev] = {
            "benign": Density.from_dict(classes["benign"]),
            "attack": Density.from_dict(classes["attack"]),
        }
    return models


def median_mad_threshold(mse: np.ndarray) -> float:
    med = float(np.median(mse))
    mad = float(np.median(np.abs(mse - med)))
    return med + 2.5 * mad


def fit_bernoulli(mse: np.ndarray, thr: float) -> float:
    """Estimate the benign anomaly rate p_ben from a calibration capture."""
    return float(np.mean(mse > thr))


def attack_onset(atk_mse: np.ndarray, thr: float, window: int = 100,
                 rate: float = 0.2) -> int:
    """First packet index where a `window`-sized rolling anomaly rate exceeds
    `rate` -- a proxy for when attack traffic actually begins."""
    flags = (atk_mse > thr).astype(int)
    for i in range(len(flags) - window):
        if flags[i:i + window].mean() > rate:
            return i
    return len(flags)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _find_files(out_dir: str, device: int, ben_suffix: str,
                atk_suffix: str) -> tuple:
    return (
        os.path.join(out_dir, f"device{device}_normal_r1_results.csv"),
        os.path.join(out_dir, f"device{device}_{ben_suffix}_results.csv"),
        os.path.join(out_dir, f"device{device}_{atk_suffix}_results.csv"),
    )


def _load_device(args, device):
    """Load the three MSE arrays + threshold + p_ben for a device."""
    ben_r1, ben_r2, atk = _find_files(args.out_dir, device,
                                      args.ben_suffix, args.atk_suffix)
    if not all(os.path.exists(p) for p in (ben_r1, ben_r2, atk)):
        return None
    mse_ben1 = read_mse(ben_r1)
    mse_ben2 = read_mse(ben_r2)
    mse_atk = read_mse(atk)
    thr = median_mad_threshold(mse_ben1)
    p_ben = fit_bernoulli(mse_ben1, thr)
    return {"r1": mse_ben1, "r2": mse_ben2, "atk": mse_atk,
            "thr": thr, "p_ben": p_ben}


def _auc(pos_scores, neg_scores) -> float:
    """Rank-based area under the ROC curve from raw alarm statistics."""
    scores = np.concatenate([np.asarray(pos_scores), np.asarray(neg_scores)])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n1 = len(pos_scores)
    n0 = len(neg_scores)
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--devices", type=int, nargs="+", default=list(range(1, 10)))
    ap.add_argument("--p-atk", type=float, default=0.3,
                    help="design anomaly rate under the attack hypothesis (default 0.3)")
    ap.add_argument("--A", type=float, default=250.0,
                    help="CUSUM alarm threshold (default 250)")
    ap.add_argument("--old-rate-rule", type=float, default=30.0,
                    help="legacy whole-capture anomaly-rate % cutoff")
    ap.add_argument("--onset-window", type=int, default=100)
    ap.add_argument("--onset-rate", type=float, default=0.2)
    ap.add_argument("--atk-suffix", default="attack",
                    help="attack results file suffix (default 'attack')")
    ap.add_argument("--ben-suffix", default="normal_r2",
                    help="benign test results file suffix (default 'normal_r2')")
    ap.add_argument("--sweep-a", type=float, nargs="+", default=None,
                    help="A values for the ROC sweep (default: auto grid)")
    args = ap.parse_args()

    print("=" * 96)
    print("  CUSUM (one-sided SPRT) sequential decisioning -- vs legacy anomaly-rate rule")
    print(f"  p_atk={args.p_atk}  A={args.A:.0f}   "
          f"legacy rule: anomaly rate > {args.old_rate_rule:.0f}% of whole capture")
    print(f"  files: deviceN_{args.atk_suffix}_results.csv (attack) vs "
          f"deviceN_{args.ben_suffix}_results.csv (benign test)")
    print("=" * 96)

    loaded = {}
    for d in args.devices:
        data = _load_device(args, d)
        if data is None:
            print(f"  device{d}: missing results files -- skipped")
        else:
            loaded[d] = data
    if not loaded:
        print("  No device data available.")
        return
    n = len(loaded)
    devs = sorted(loaded)

    hdr = (f"  {'dev':<5}{'p_ben':>7}{'onset':>7} | "
           f"{'CUSUM-TPR':>10}{'lat-after':>10}{'lat-abs':>8} | "
           f"{'OLD-TPR':>8}{'CUSUM-FP':>9}{'OLD-FP':>8}  peakS atk/ben")
    print(hdr)
    print("  " + "-" * 92)

    tp = fp = old_tp = old_fp = 0
    lats = []
    peaks_atk, peaks_ben = [], []

    for d in devs:
        data = loaded[d]
        mse_ben1, mse_ben2, mse_atk = data["r1"], data["r2"], data["atk"]
        thr, p_ben = data["thr"], data["p_ben"]
        det = CUSUM(p_ben, args.p_atk, args.A)

        flags_ben2 = mse_ben2 > thr
        flags_atk = mse_atk > thr
        hit_atk = det.run(flags_atk)
        hit_ben2 = det.run(flags_ben2)

        onset = attack_onset(mse_atk, thr, args.onset_window, args.onset_rate)
        lat_abs = hit_atk if hit_atk else -1
        lat_after = max(0, lat_abs - onset) if hit_atk else -1

        old_tp_d = int(np.mean(flags_atk) * 100.0 > args.old_rate_rule)
        old_fp_d = int(np.mean(flags_ben2) * 100.0 > args.old_rate_rule)
        cusum_tp_d = int(hit_atk is not None)
        cusum_fp_d = int(hit_ben2 is not None)

        tp += cusum_tp_d
        fp += cusum_fp_d
        old_tp += old_tp_d
        old_fp += old_fp_d
        if hit_atk:
            lats.append(lat_after)

        pk_atk = det.peak(flags_atk)
        pk_ben = det.peak(flags_ben2)
        peaks_atk.append(pk_atk)
        peaks_ben.append(pk_ben)

        print(f"  d{d:<4}{p_ben:>7.3f}{onset:>7} | "
              f"{'YES' if cusum_tp_d else 'no':>10}{lat_after if cusum_tp_d else '--':>10}"
              f"{lat_abs if cusum_tp_d else '--':>8} | "
              f"{'YES' if old_tp_d else 'no':>8}{'FP' if cusum_fp_d else 'ok':>9}"
              f"{'FP' if old_fp_d else 'ok':>8}  "
              f"{pk_atk:>7.0f}/{pk_ben:>6.0f}")

    print("  " + "-" * 92)
    med = np.median(lats) if lats else float("nan")
    print(f"  OVERALL (n={n}) | "
          f"CUSUM TPR={tp}/{n} ({100.0*tp/n:.0f}%)  FPR={fp}/{n} ({100.0*fp/n:.0f}%)  "
          f"median packets-after-onset={med:.0f}")
    print(f"  LEGACY    | "
          f"TPR={old_tp}/{n} ({100.0*old_tp/n:.0f}%)  FPR={old_fp}/{n} ({100.0*old_fp/n:.0f}%)  "
          f"(decision needs the whole capture)")

    # -- sensitivity sweep --------------------------------------------
    print("\n  Sensitivity sweep (p_atk x A):")
    print(f"  {'p_atk':>6}{'A':>6} | {'TPR':>5}{'FPR':>5}{'med-lat':>9}")
    for pa, A in ((0.2, 250.0), (0.3, 150.0), (0.3, 250.0), (0.3, 400.0),
                  (0.5, 250.0), (0.5, 400.0), (0.7, 250.0), (0.7, 400.0)):
        tpc = fpc = 0
        lc = []
        for d in devs:
            data = loaded[d]
            thr = data["thr"]
            det = CUSUM(data["p_ben"], pa, A)
            ha = det.run(data["atk"] > thr)
            hb = det.run(data["r2"] > thr)
            tpc += int(ha is not None)
            fpc += int(hb is not None)
            if ha is not None:
                on = attack_onset(data["atk"], thr, args.onset_window, args.onset_rate)
                lc.append(max(0, ha - on))
        print(f"{pa:>6.1f}{A:>6.0f} | {tpc:>4}/{n}{fpc:>4}/{n}"
              f"{np.median(lc) if lc else -1:>9.0f}")

    # -- ROC / threshold sweep on the peak-S alarm statistic ----------
    As = args.sweep_a or [50, 75, 100, 150, 200, 250, 350, 500, 750, 1000, 1500]
    auc = _auc(peaks_atk, peaks_ben)
    print(f"\n  ROC / A-sweep  (peak-S alarm statistic, p_atk={args.p_atk})")
    print(f"  AUC = {auc:.3f}   (benign test: {args.ben_suffix}, attack test: {args.atk_suffix})")
    print(f"  {'A':>6} | {'TPR':>5}{'FPR':>5}{'med-lat-after':>13}{'med-lat-abs':>12}")
    paths = {}
    for d in devs:
        data = loaded[d]
        det = CUSUM(data["p_ben"], args.p_atk, 1.0)
        paths[d] = {
            "atk": det.path(data["atk"] > data["thr"]),
            "ben": det.path(data["r2"] > data["thr"]),
            "onset": attack_onset(data["atk"], data["thr"],
                                  args.onset_window, args.onset_rate),
        }
    for A in As:
        tpc = fpc = 0
        la = []
        la_abs = []
        for d in devs:
            pth = paths[d]
            ha = det.alarm_index(pth["atk"], A)
            hb = det.alarm_index(pth["ben"], A)
            tpc += int(ha is not None)
            fpc += int(hb is not None)
            if ha is not None:
                la.append(max(0, ha - pth["onset"]))
                la_abs.append(ha)
        med_la = np.median(la) if la else -1
        med_abs = np.median(la_abs) if la_abs else -1
        print(f"{A:>6.0f} | {tpc:>4}/{n}{fpc:>4}/{n}"
              f"{med_la:>13.0f}{med_abs:>12.0f}")


if __name__ == "__main__":
    main()
