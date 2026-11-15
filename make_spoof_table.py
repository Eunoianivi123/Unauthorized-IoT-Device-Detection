#!/usr/bin/env python3
"""Regenerate the MAC-spoofing table from the 3-repetition study.

The spoof scenario is "a rogue container binds a *whitelisted* MAC and sends
Mirai traffic".  The two tiers are evidenced differently and this script is
explicit about it:

  identity tier  -- LIVE evidence, carried over unchanged from the Sep-20 spoof
                    test (results/spoof_test_results.csv, `identity_tier`
                    column).  Binding a whitelisted MAC is by construction
                    indistinguishable to the identity tier, so this column is
                    9/9 AUTHORIZED and re-running it would add nothing.

  behavioural tier -- RECOMPUTED here, offline, on the fresh 3-repetition
                    attack captures with the deployed C2 model and the
                    leave-one-run-out CUSUM parameters.  The autoencoder
                    scores only the 115 flow statistics; src_ip/dst_ip are
                    dropped before scoring (see pcap_csv.feature_columns and
                    the comment-4 ablation, which established this), so the
                    behavioural verdict is invariant to the bound MAC/IP.

Reads   output/repeats/run_variance_3run.csv   (written by cusum_calibration_3run.py)
        results/spoof_test_results.csv        (identity-tier evidence)
Writes  results/spoof_test_results_3run.csv
        results/spoof_test_3run_summary.md
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics as st

STUDY = "output/repeats/run_variance_3run.csv"
IDENT = "results/spoof_test_results.csv"
OUT_CSV = "results/spoof_test_results_3run.csv"
OUT_MD = "results/spoof_test_3run_summary.md"

DEV_NAMES = {
    1: "Danmini_Doorbell", 2: "Ecobee_Thermostat", 3: "Ennio_Doorbell",
    4: "Philips_B120N10_Baby_Monitor", 5: "Provision_PT_737E_Security_Camera",
    6: "Provision_PT_838_Security_Camera", 7: "Samsung_SNH_1011_N_Webcam",
    8: "SimpleHome_XCS7_1002_WHT_Security_Camera",
    9: "SimpleHome_XCS7_1003_WHT_Security_Camera",
}
SPOOF_IPS = {i: f"192.168.100.{50 + i}" for i in range(1, 10)}


def load(path):
    if not os.path.exists(path):
        raise SystemExit(f"[FATAL] missing {path}")
    with open(path) as fh:
        return list(csv.DictReader(fh))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def ms(vals, nd=1):
    v = [x for x in vals if x is not None and math.isfinite(x)]
    if not v:
        return "—", None, None
    m, s = st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
    return f"{m:.{nd}f} ± {s:.{nd}f}", m, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default=STUDY)
    ap.add_argument("--ident", default=IDENT)
    ap.add_argument("--out-csv", default=OUT_CSV)
    ap.add_argument("--out-md", default=OUT_MD)
    args = ap.parse_args()

    study = [r for r in load(args.study) if r["mode"] == "attack"]
    ident = {r["device_id"]: r for r in load(args.ident)}

    rows = []
    for d in range(1, 10):
        rs = [r for r in study if int(r["device"]) == d]
        if not rs:
            print(f"[WARN] device {d}: no attack rows in the study -- skipped")
            continue
        alarms = [int(r["cusum_alarm"]) for r in rs]
        dp, dpm, dps = ms([num(r["delay_packets"]) for r in rs], 0)
        ds, dsm, dss = ms([num(r["delay_seconds"]) for r in rs], 4)
        thr, _, _ = ms([num(r["threshold"]) for r in rs], 3)
        pben, _, _ = ms([num(r["p_ben"]) for r in rs], 4)
        patk, _, _ = ms([num(r["p_atk"]) for r in rs], 2)
        A, _, _ = ms([num(r["A"]) for r in rs], 0)
        ident_v = ident.get(str(d), {}).get("identity_tier", "NOT MEASURED")
        rows.append({
            "device_id": d,
            "device_name": DEV_NAMES[d],
            "spoofed_mac": ident.get(str(d), {}).get("spoofed_mac", ""),
            "spoof_ip": SPOOF_IPS[d],
            "identity_tier": ident_v,
            "behavioural_reps": len(rs),
            "behavioural_detected": sum(alarms),
            "behavioural_TPR": f"{sum(alarms)/len(alarms):.3f}",
            "delay_packets_mean_sd": dp,
            "delay_packets_mean": "" if dpm is None else f"{dpm:.1f}",
            "delay_packets_sd": "" if dps is None else f"{dps:.1f}",
            "delay_seconds_mean_sd": ds,
            "delay_seconds_mean": "" if dsm is None else f"{dsm:.4f}",
            "threshold_mean_sd": thr,
            "p_ben_mean_sd": pben,
            "p_atk_mean_sd": patk,
            "A_mean_sd": A,
        })

    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out_csv} ({len(rows)} rows)")

    n = len(rows)
    det = sum(r["behavioural_detected"] for r in rows)
    reps = sum(r["behavioural_reps"] for r in rows)
    dev_det = sum(1 for r in rows
                  if r["behavioural_detected"] == r["behavioural_reps"])
    id_blocked = sum(1 for r in rows if r["identity_tier"] != "AUTHORIZED")
    dpm = [num(r["delay_packets_mean"]) for r in rows]
    dsm = [num(r["delay_seconds_mean"]) for r in rows]
    dpack, _, _ = ms(dpm, 0)
    dsec, _, _ = ms(dsm, 4)

    md = f"""# MAC-spoofing re-test on the 3-repetition campaign (comment 3)

A rogue container binds a **whitelisted** MAC — the identity tier cannot
distinguish it from the real device — and sends Mirai traffic. Only the
behavioural tier (autoencoder + CUSUM) can catch it.

## How each column is evidenced

| Column | Source | Kind of evidence |
|---|---|---|
| `identity_tier` | `results/spoof_test_results.csv` (20 Sep 2026) | **live**, carried over unchanged |
| everything else | `output/repeats/run_variance_3run.csv` | **recomputed offline** on the fresh 3-repetition attack captures |

The behavioural verdict is invariant to the bound MAC/IP: the autoencoder scores
only the 115 N-BaIoT flow statistics and `src_ip`/`dst_ip` are dropped before
scoring (established in `results/ablation_comment4_summary.md`, where the
foreign-MAC and spoofed-MAC behavioural cells are identical). So the same
offline computation answers the spoof question, and the identity column is the
only part that genuinely needs live containers.

## Result

| Device | Spoofed MAC | Identity tier | Behavioural | Delay (packets) | Delay (s) | Thr | `p_ben` | `P_ATK` | `A` |
|---|---|---|---|---|---|---|---|---|---|
"""
    for r in rows:
        md += (f"| {r['device_id']} — {r['device_name']} | `{r['spoofed_mac']}` | "
               f"{r['identity_tier']} | {r['behavioural_detected']}/{r['behavioural_reps']} | "
               f"{r['delay_packets_mean_sd']} | {r['delay_seconds_mean_sd']} | "
               f"{r['threshold_mean_sd']} | {r['p_ben_mean_sd']} | {r['p_atk_mean_sd']} | "
               f"{r['A_mean_sd']} |\n")

    md += f"""
## Aggregates

| Question | Answer |
|---|---|
| Spoofed devices that tried to connect | **{n}** |
| Blocked by the identity tier | **{id_blocked}/{n}** — spoofing succeeds by design |
| Blocked by the behavioural tier | **{dev_det}/{n}** devices, **{det}/{reps}** capture-repetitions |
| Detection delay, packets | {dpack} (mean ± sd over devices) |
| Detection delay, seconds | {dsec} |

Delay is measured from the **start of the capture** with the single definition in
`cusum_eval.latency()` (`alarm_index − 1`), the same one used by the calibration
study and the detector comparison. The attack captures are a short dense burst
(~5.3 kpps), so the seconds figure is "time into the attack burst", not
wall-clock time from an attacker's first connection.

## Caveats

- The identity column is from a single live spoof run per device (20 Sep 2026);
  it was not repeated 3× because binding a whitelisted MAC is a deterministic
  pass of the identity tier, not a stochastic event.
- The behavioural tier sees only flow statistics, so it would behave identically
  for a spoofed container and for the genuine device sending the same traffic.
  That is the point of the hybrid, but it means the behavioural numbers here are
  the same experiment as the plain-attack numbers in the main study, not an
  independent replication.
"""
    with open(args.out_md, "w") as fh:
        fh.write(md)
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
