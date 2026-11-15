#!/usr/bin/env python3
"""Shared evaluation core for the CUSUM calibration / detector study.

Why this module exists
----------------------
The reviewer correctly identified three inconsistencies in the earlier
evidence:

  1. The CUSUM parameters (P_ATK, A) were chosen by looking at the very
     captures the device-level results were then reported on, so those
     results are optimistic.
  2. The parameter sweep and the CUSUM/EWMA/Shewhart comparison ran on
     per-packet flags produced by the PRE-C2 model, not the deployed one.
  3. Detection delay was measured two different ways on two different
     capture sets (about 210 packets in the sweep vs 2143 in the C2
     spoof test), so the two numbers were not comparable.

This module is the single implementation used by every analysis in the
study, so that (2) and (3) cannot drift apart again:

  * scoring always uses the DEPLOYED C2 model, read from the feature CSVs
    (never from the stale ``*_results.csv`` ``anomaly`` columns);
  * thresholds and p_ben are *fitted on a caller-supplied calibration
    split only*, so a caller can hold out captures honestly;
  * detection delay has ONE definition, `latency()`, used identically by
    the parameter sweep, the detector comparison and the spoof test.

Run inside the pipeline image (tensorflow + joblib):
  docker run --rm --entrypoint python3 -v "$PWD:/work" -w /work \
      nbaiot-pipeline:latest cusum_eval.py
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pcap_csv import feature_columns as FEATURE_NAMES  # single source of truth

assert len(FEATURE_NAMES) == 115

DEVICES = tuple(range(1, 10))
REPS = ("r1", "r2", "r3")
META = ("timestamp", "src_ip", "dst_ip")

# Modes captured by capture_repeats.sh.  `stress` is a single-repetition
# FPR-robustness probe and is therefore never part of a calibration split.
CALIB_MODES = ("normal", "attack", "attack_hard")
STRESS_MODE = "stress"


# ----------------------------------------------------------------------
# Model / scoring
# ----------------------------------------------------------------------
def load_model(model_dir: str = "new_models/winner"):
    """Load the deployed all-in-one (C2 winner) autoencoder + scaler."""
    from tensorflow import keras
    import joblib

    model = keras.models.load_model(os.path.join(model_dir, "autoencoder_allinone.keras"))
    scaler = joblib.load(os.path.join(model_dir, "scaler_allinone.pkl"))
    return model, scaler


def features_path(capture_dir: str, device: int, mode: str, rep: str) -> str:
    return os.path.join(capture_dir, f"device{device}_{mode}_{rep}_capture_features.csv")


def score_capture(path: str, model, scaler) -> pd.DataFrame:
    """Score one feature CSV -> DataFrame[mse, timestamp].

    Mirrors testmodel.py exactly (same feature order, same float32 cast,
    same nan_to_num, same per-row MSE over the scaled space) so the study
    and the deployed pipeline can never disagree about a score.
    """
    df = pd.read_csv(path)
    X = df[list(FEATURE_NAMES)].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = scaler.transform(X)
    rec = model.predict(Xs, batch_size=256, verbose=0)
    mse = np.mean(np.power(Xs - rec, 2), axis=1)
    if "timestamp" in df.columns:
        ts = df["timestamp"].to_numpy(dtype=np.float64)
    else:
        ts = np.arange(len(mse), dtype=np.float64)
    return pd.DataFrame({"mse": mse, "timestamp": ts})


# ----------------------------------------------------------------------
# Threshold / p_ben estimation  (paper rule: median + 2.5 * MAD)
# ----------------------------------------------------------------------
def median_mad_threshold(mse: np.ndarray) -> float:
    med = float(np.median(mse))
    mad = float(np.median(np.abs(mse - med)))
    return med + 2.5 * mad


def flags_from(mse: np.ndarray, thr: float) -> np.ndarray:
    return (np.asarray(mse) > float(thr)).astype(np.int8)


# ----------------------------------------------------------------------
# One-sided CUSUM over per-packet anomaly flags
# ----------------------------------------------------------------------
@dataclass
class CUSUM:
    """S(n) = max(0, S(n-1) + w(n));  w = log(p_atk/p_ben) if flag else
    log((1-p_atk)/(1-p_ben)).  An attack is declared at the first n with
    S(n) >= A.  Identical in form to identification_engine._cusum_alarm."""

    p_ben: float
    p_atk: float
    A: float = 250.0

    def __post_init__(self):
        self.p_ben = float(np.clip(self.p_ben, 1e-12, 1.0 - 1e-12))
        self.p_atk = float(np.clip(self.p_atk, 1e-12, 1.0 - 1e-12))
        self.w_anom = math.log(self.p_atk / self.p_ben)
        self.w_normal = math.log((1.0 - self.p_atk) / (1.0 - self.p_ben))

    def path(self, flags) -> np.ndarray:
        """Full S(n) trajectory.  Independent of A, so it can be computed
        once per capture and then thresholded for the whole A-grid.

        Implements the recursion S(n) = max(0, S(n-1) + w(n)) exactly, using
        the max-suffix-sum identity

            S(n) = P(n) - min_{0<=j<=n} P(j),      P(n) = sum_{i<=n} w(i)

        so it vectorises.  Do NOT replace this with np.maximum.accumulate:
        that is a running MAXIMUM, which never resets after a negative
        excursion and is NOT the CUSUM.
        """
        w = np.where(np.asarray(flags) > 0, self.w_anom, self.w_normal)
        P = np.concatenate(([0.0], np.cumsum(w, dtype=np.float64)))
        run_min = np.minimum.accumulate(P)
        return P[1:] - run_min[1:]

    def alarm_index(self, path: np.ndarray, A: float | None = None):
        """1-based packet index where S first reaches A, else None."""
        A = self.A if A is None else float(A)
        hit = np.flatnonzero(path >= A)
        return int(hit[0]) + 1 if len(hit) else None

    def peak(self, path: np.ndarray) -> float:
        return float(path.max()) if len(path) else 0.0


# ----------------------------------------------------------------------
# Detection delay  --  ONE definition, used by every analysis
# ----------------------------------------------------------------------
# Definition (stated once, in the paper and in every table):
#   The attack captures are produced by starting the attack generator
#   inside the capture window, so the attack stream begins at the first
#   captured packet.  Detection delay is therefore measured from the
#   START OF THE CAPTURE:
#       delay_packets = (alarm packet index) - 1
#       delay_seconds = delay_packets / packet_rate
#   where packet_rate = rows / (last_timestamp - first_timestamp).
#
# `packets_after_first_flag` is reported alongside purely for traceability
# to the older analyses, which used a first-flag / rolling-window "onset".
# It is NOT the headline metric.
@dataclass
class Latency:
    alarm_index: int | None = None      # 1-based packet index of the alarm
    delay_packets: int | None = None    # alarm_index - 1, from capture start
    delay_seconds: float | None = None
    packets_after_first_flag: int | None = None
    packet_rate_pps: float = 0.0
    peak_S: float = 0.0
    flags: int = 0
    rows: int = 0
    extra: dict = field(default_factory=dict)

    def as_dict(self):
        return {
            "alarm_index": self.alarm_index,
            "delay_packets": self.delay_packets,
            "delay_seconds": (round(self.delay_seconds, 4)
                              if self.delay_seconds is not None else None),
            "packets_after_first_flag": self.packets_after_first_flag,
            "packet_rate_pps": round(self.packet_rate_pps, 1),
            "peak_S": round(self.peak_S, 2),
            "flags": self.flags,
            "rows": self.rows,
        }


def latency(path: np.ndarray, flags: np.ndarray, ts: np.ndarray,
            A: float, packet_rate: float | None = None) -> Latency:
    """Single source of truth for detection delay (see comment above)."""
    flags = np.asarray(flags)
    ts = np.asarray(ts, dtype=np.float64)
    n = len(flags)
    if n == 0:
        return Latency()

    span = float(ts[-1] - ts[0])
    if packet_rate is None:
        pps = (n / span) if span > 0 else 0.0
    else:
        pps = float(packet_rate)

    hit = np.flatnonzero(path >= float(A))
    k = int(hit[0]) + 1 if len(hit) else None

    first_flag = np.flatnonzero(flags > 0)
    f0 = int(first_flag[0]) if len(first_flag) else None

    delay_pk = (k - 1) if k is not None else None
    return Latency(
        alarm_index=k,
        delay_packets=delay_pk,
        delay_seconds=(delay_pk / pps) if (delay_pk is not None and pps > 0) else None,
        packets_after_first_flag=(k - f0 - 1) if (k is not None and f0 is not None and k > f0) else None,
        packet_rate_pps=pps,
        peak_S=float(path.max()) if n else 0.0,
        flags=int(flags.sum()),
        rows=int(n),
    )


# ----------------------------------------------------------------------
# Alternative sequential detectors (same flags, same latency contract)
# ----------------------------------------------------------------------
class EWMA:
    """One-sided EWMA control chart over Bernoulli anomaly flags.

    z(n) = lam*x(n) + (1-lam)*z(n-1);  alarm at z >= h.
    """

    def __init__(self, p_ben: float, lam: float = 0.1, h: float = float("inf")):
        self.p_ben = float(np.clip(p_ben, 1e-12, 1.0 - 1e-12))
        self.lam = float(lam)
        self.h = float(h)

    def path(self, flags) -> np.ndarray:
        x = np.asarray(flags, dtype=np.float64)
        # recursive EWMA, vectorised via the geometric-kernel identity
        n = len(x)
        if n == 0:
            return np.empty(0)
        z = np.empty(n)
        acc = self.p_ben
        for i in range(n):
            acc = self.lam * x[i] + (1.0 - self.lam) * acc
            z[i] = acc
        return z

    def calibrate(self, cal_flags, margin: float = 1.10, floor_mult: float = 1.25) -> float:
        """Smallest limit with a safety margin over the calibration peak --
        the same footing on which the production A was originally derived."""
        peak = float(self.path(cal_flags).max()) if len(cal_flags) else self.p_ben
        self.h = max(peak * margin, self.p_ben * floor_mult)
        return self.h


class Shewhart:
    """p-chart: non-overlapping windows of n packets; alarm when a window's
    anomalous proportion reaches the UCL."""

    def __init__(self, p_ben: float, window: int = 100, ucl: float = float("inf")):
        self.p_ben = float(np.clip(p_ben, 1e-12, 1.0 - 1e-12))
        self.n = int(window)
        self.ucl = float(ucl)

    def proportions(self, flags) -> np.ndarray:
        f = np.asarray(flags)
        m = len(f) // self.n
        if m == 0:
            return np.empty(0)
        return f[: m * self.n].reshape(m, self.n).mean(axis=1)

    def path(self, flags) -> np.ndarray:
        """Expand the p-chart to one value per packet so `latency()` can be
        reused unchanged: a window's value is held from its first packet."""
        props = self.proportions(flags)
        f = np.asarray(flags)
        if len(props) == 0:
            return np.zeros(len(f))
        out = np.zeros(len(f))
        for i in range(len(props)):
            out[i * self.n: (i + 1) * self.n] = props[i]
        return out

    def calibrate(self, cal_flags, margin: float = 1.05) -> float:
        props = self.proportions(cal_flags)
        peak = float(props.max()) if len(props) else 0.0
        floor = max(2.0 / self.n, 3.0 * self.p_ben)
        self.ucl = max(peak * margin, floor)
        return self.ucl


# ----------------------------------------------------------------------
# Packet-level metrics (for the mean +/- std addendum)
# ----------------------------------------------------------------------
def binary_metrics(flags, labels) -> dict:
    """flags: predicted anomaly (1 = anomalous). labels: 1 = attack."""
    f = np.asarray(flags).astype(bool)
    y = np.asarray(labels).astype(bool)
    tp = int(np.sum(f & y)); fp = int(np.sum(f & ~y))
    fn = int(np.sum(~f & y)); tn = int(np.sum(~f & ~y))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / len(f) if len(f) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
            "tpr": rec}


def roc_auc(neg_scores, pos_scores) -> float:
    """Rank-based AUC (ties get average ranks)."""
    neg = np.asarray(neg_scores, dtype=np.float64)
    pos = np.asarray(pos_scores, dtype=np.float64)
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    srt = scores[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def mean_std(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return (None, None)
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
