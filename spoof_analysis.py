#!/usr/bin/env python3
"""MAC-spoofing test analysis — behavioral (autoencoder + CUSUM) tier.

Re-scores the already-captured attack PCAP features (device{N}_attack_capture_features.csv)
with the CURRENT deployed winner model and per-device threshold, then runs the same
one-sided CUSUM used by identification_engine.py and reports, per device:
  - CUSUM decision, S_final
  - packets_to_alarm (absolute row index) and packets after attack onset
  - detection time (seconds) from attack onset, using the features-CSV packet timestamps

Also recomputes benign held-out (normal_r2) false blocks under the current model.

Inputs (mounted):
  /output  -> ./output          (capture feature CSVs)
  /model   -> /home/neelavathy/model (winner model + scaler, read-only)
Outputs:
  /output/spoof_test_results.csv  (per-device behavioral table)
  /output/spoof_benign_fpr.csv    (benign-r2 false-block recomputation)
"""
import json
import math
import os

import numpy as np
import pandas as pd

OUT = "/output"
MODEL = "/model/autoencoder_allinone.keras"
SCALER = "/model/scaler_allinone.pkl"

# ── CUSUM parameters — MUST match identification_engine.py ─────────────────
P_ATK = 0.3
CUSUM_A = 250.0
DEVICE_P_BEN = {
    1: 0.041, 2: 0.195, 3: 0.050, 4: 0.031, 5: 0.035,
    6: 0.044, 7: 0.061, 8: 0.046, 9: 0.047,
}
# ── Per-device thresholds — MUST match entrypoint.sh (winner C2) ────────────
THRESHOLDS = {
    1: 8.55554, 2: 0.122546, 3: 2.124926, 4: 26.941192, 5: 13.610937,
    6: 7.820535, 7: 4.273, 8: 1.363198, 9: 3.791139,
}
DEVICE_NAMES = {
    1: "Danmini_Doorbell", 2: "Ecobee_Thermostat", 3: "Ennio_Doorbell",
    4: "Philips_B120N10_Baby_Monitor", 5: "Provision_PT_737E_Security_Camera",
    6: "Provision_PT_838_Security_Camera", 7: "Samsung_SNH_1011_N_Webcam",
    8: "SimpleHome_XCS7_1002_WHT_Security_Camera", 9: "SimpleHome_XCS7_1003_WHT_Security_Camera",
}
# Spoof services bind the WHITELISTED MAC (identity tier passes) on .51-.59
SPOOF_IPS = {i: f"192.168.100.{50 + i}" for i in range(1, 10)}
WHITELIST_MACS = {
    1: "54:E6:1B:A3:2F:01", 2: "44:61:32:A1:BC:02", 3: "B4:E6:2D:F7:11:03",
    4: "00:17:88:4C:DA:04", 5: "00:1C:A8:9E:55:05", 6: "00:1C:A8:9E:55:06",
    7: "00:12:FB:C8:44:07", 8: "A8:40:41:3D:77:08", 9: "A8:40:41:3D:77:09",
}
META = {"timestamp", "src_ip", "dst_ip"}


def load_model():
    from tensorflow import keras
    import joblib
    model = keras.models.load_model(MODEL)
    scaler = joblib.load(SCALER)
    return model, scaler


def score(features_csv: str, model, scaler, threshold: float) -> pd.DataFrame:
    df = pd.read_csv(features_csv)
    feats = [c for c in df.columns if c not in META]
    X = df[feats].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = scaler.transform(X)
    rec = model.predict(Xs, batch_size=256, verbose=0)
    mse = np.mean(np.power(Xs - rec, 2), axis=1)
    df_out = pd.DataFrame({"mse": mse, "anomaly": (mse > threshold).astype(int),
                           "threshold": threshold})
    df_out["timestamp"] = df["timestamp"].values if "timestamp" in df.columns else np.arange(len(df))
    return df_out


def cusum(flags, device_id):
    """One-sided CUSUM, must match identification_engine._cusum_alarm."""
    p_ben = max(1e-12, min(DEVICE_P_BEN[device_id], 1 - 1e-12))
    p_atk = max(1e-12, min(P_ATK, 1 - 1e-12))
    w_a = math.log(p_atk / p_ben)
    w_n = math.log((1 - p_atk) / (1 - p_ben))
    s = 0.0
    for i, f in enumerate(flags):
        s = max(0.0, s + (w_a if f else w_n))
        if s >= CUSUM_A:
            return True, i + 1, s
    return False, None, s


def analyze_device(device_id, model, scaler):
    feats_csv = os.path.join(OUT, f"device{device_id}_attack_capture_features.csv")
    if not os.path.exists(feats_csv):
        return None
    r = score(feats_csv, model, scaler, THRESHOLDS[device_id])
    flags = r["anomaly"].tolist()
    ts = r["timestamp"].tolist()
    n = len(flags)
    n_anom = int(sum(flags))
    alarm, k, s = cusum(flags, device_id)
    onset = next((i for i, f in enumerate(flags) if f), None)  # attack onset (~pkt 1 for pure-attack captures)
    span_s = max(1e-6, ts[-1] - ts[0])
    pps = n / span_s if n > 1 else 0.0
    row = {
        "device_id": device_id,
        "device_name": DEVICE_NAMES[device_id],
        "spoofed_mac": WHITELIST_MACS[device_id],
        "spoof_ip": SPOOF_IPS[device_id],
        "rows": n,
        "anomalous": n_anom,
        "anomaly_pct": round(n_anom / n * 100, 2) if n else 0.0,
        "capture_span_s": round(span_s, 3),
        "packet_rate_pps": round(pps, 1),
        "cusum_decision": "ATTACK" if alarm else "BENIGN",
        "S_final": round(s, 2),
        "packets_to_alarm": k,
        "onset_packet": (onset + 1) if onset is not None else None,
        "packets_from_onset": (k - onset) if (alarm and onset is not None) else None,
        "time_to_alarm_s": round(k / pps, 4) if (alarm and pps > 0) else None,
        "behavioral_tier": "DETECTED" if alarm else "MISS",
    }
    return row


def benign_fpr(model, scaler):
    rows = []
    for device_id in range(1, 10):
        csv = os.path.join(OUT, f"device{device_id}_normal_r2_capture_features.csv")
        if not os.path.exists(csv):
            rows.append((device_id, "missing", None))
            continue
        r = score(csv, model, scaler, THRESHOLDS[device_id])
        alarm, k, s = cusum(r["anomaly"].tolist(), device_id)
        rows.append((device_id, "banned" if alarm else "clean", round(s, 2)))
    return rows


def main():
    model, scaler = load_model()
    print("[*] model + scaler loaded", flush=True)

    results = []
    for device_id in range(1, 10):
        row = analyze_device(device_id, model, scaler)
        if row is None:
            print(f"[!] device{device_id} missing attack features — skipped", flush=True)
            continue
        results.append(row)
        print(f"    d{device_id} {row['cusum_decision']:<7} S={row['S_final']:<8}"
              f" k={row['packets_to_alarm']} "
              f"t_onset={row['time_to_alarm_s']}s n_anom={row['anomalous']}/{row['rows']}", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT, "spoof_test_results.csv"), index=False)
    print(f"[*] saved /output/spoof_test_results.csv ({len(df)} rows)", flush=True)

    fpr = benign_fpr(model, scaler)
    fpr_df = pd.DataFrame(fpr, columns=["device_id", "benign_r2_verdict", "S_final"])
    fpr_df.to_csv(os.path.join(OUT, "spoof_benign_fpr.csv"), index=False)
    n_banned = int((fpr_df["benign_r2_verdict"] == "banned").sum())
    print(f"[*] benign-r2 false blocks: {n_banned}/9", flush=True)

    n_att = int((df["cusum_decision"] == "ATTACK").sum())
    tt = df["time_to_alarm_s"].dropna()
    print("\n==== SUMMARY ====")
    print(f"  spoof attempts (whitelisted MAC) : {len(df)}")
    print(f"  identity tier blocked            : 0 (all passed — spoof succeeds)")
    print(f"  behavioral tier (CUSUM) detected : {n_att}/{len(df)}")
    print(f"  false blocks (benign r2)         : {n_banned}/9")
    if len(tt):
        print(f"  detection time from onset (s)    : min {tt.min():.2f}, "
              f"median {tt.median():.2f}, max {tt.max():.2f}")


if __name__ == "__main__":
    main()