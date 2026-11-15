#!/usr/bin/env python3
"""Build per-packet role labels for the Docker-testbed captures.

Reviewer comment 1 fix: the old metrics labelled EVERY row in an
``attack_hard`` capture as attack, even though that capture deliberately mixes
three kinds of packets:

  attack          device-origin packet carrying attack traffic
  benign          device-origin packet from the normal device workload
  target_response packet sent by the target/gateway (192.168.100.3) to the device

This script assigns a role to every feature row and writes three label views:

  label_strict   1 = attack, 0 = benign, -1 = excluded (target response / other)
  label_session  1 = attack OR target response, 0 = benign, -1 = excluded
  label_refined  1 = attack OR attack-related target response,
                 0 = benign OR benign-related target response, -1 = excluded

``label_strict`` is the paper's main per-packet definition.  ``label_session``
is the sensitivity check requested in the review plan.  ``label_refined`` is
the honest middle ground for ``attack_hard``: a target response is treated as
attack-related only when its source port is not a known benign device port
(so benign HTTP replies are not silently promoted to attack).

Row alignment
-------------
``pcap_csv.py`` emits one row per IPv4 packet, in pcap order, skipping ARP /
IPv6.  For ``attack_hard`` the canonical pcap and feature CSV match, so we
recover the destination port from the pcap.  For every other mode the feature
CSV already carries ``src_ip``/``dst_ip`` and no port-based decision is needed;
we label directly from the feature CSV (this also avoids the Sep-2026 spoof
test overwriting the standard-attack pcaps).

Run in the pipeline image:
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
      -v "$PWD:/work" -w /work --entrypoint python3 \
      nbaiot-pipeline:latest build_packet_labels.py
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
from scapy.all import IP, TCP, UDP, PcapReader

TARGET_IP = "192.168.100.3"

# Ports used by device_traffic.py's benign workload.  Anything else in an
# attack_hard capture is attack traffic (attack_hard.py picks random ports).
BENIGN_PORTS = {53, 80, 123, 554, 5004, 5222, 8883, 9999, 37777}

MODES = ("normal_r1", "normal_r2", "attack", "attack_hard", "stress")
ATTACK_MODES = {"attack", "attack_hard"}


def device_ips(device: int) -> set[str]:
    """Authorized device IP plus its spoof-test IP (50+N).

    The Sep-2026 spoof test overwrote the device1/device2 standard-attack
    feature CSVs with captures from spoof-device-1/2, which bind the SAME
    whitelisted MAC but use 192.168.100.51/52.  Those captures are still valid
    attack traffic (features carry no IP addresses), so accept both IPs as
    device-origin.
    """
    return {f"192.168.100.{10 + device}", f"192.168.100.{50 + device}"}


def feature_path(device: int, mode: str, out_dir: str) -> str:
    return os.path.join(out_dir, f"device{device}_{mode}_capture_features.csv")


def pcap_path(device: int, mode: str, out_dir: str) -> str:
    return os.path.join(out_dir, f"device{device}_{mode}_capture.pcap")


def read_pcap_rows(path: str):
    """Return one tuple per IPv4 packet: (ts, src, dst, sport, dport, proto, size)."""
    rows = []
    with PcapReader(path) as reader:
        for pkt in reader:
            if not pkt.haslayer(IP):
                continue
            ip = pkt[IP]
            sport = dport = 0
            if pkt.haslayer(TCP):
                sport, dport, proto = int(pkt[TCP].sport), int(pkt[TCP].dport), 6
            elif pkt.haslayer(UDP):
                sport, dport, proto = int(pkt[UDP].sport), int(pkt[UDP].dport), 17
            else:
                proto = int(ip.proto)
            rows.append((float(pkt.time), str(ip.src), str(ip.dst),
                         sport, dport, proto, len(pkt)))
    return rows


def pcap_matches_features(pcap_rows, df: pd.DataFrame) -> bool:
    """Check row count and a spread of rows before trusting pcap order."""
    if len(pcap_rows) != len(df):
        return False
    n = len(df)
    idxs = sorted({0, 1, n // 4, n // 2, (3 * n) // 4, n - 2, n - 1})
    for i in idxs:
        ts, src, dst = pcap_rows[i][0], pcap_rows[i][1], pcap_rows[i][2]
        if abs(ts - float(df.iloc[i]["timestamp"])) > 1e-3:
            return False
        if src != str(df.iloc[i]["src_ip"]) or dst != str(df.iloc[i]["dst_ip"]):
            return False
    return True


def assign_role(mode: str, src: str, dport: int, dev_ips: set[str]) -> str:
    if src == TARGET_IP:
        return "target_response"
    if src not in dev_ips:
        return "other"
    if mode in ("normal_r1", "normal_r2", "stress"):
        return "benign"
    if mode == "attack":
        return "attack"
    if mode == "attack_hard":
        return "benign" if dport in BENIGN_PORTS else "attack"
    return "other"


def labels_for_mode(device: int, mode: str, out_dir: str):
    feat = feature_path(device, mode, out_dir)
    if not os.path.exists(feat):
        raise FileNotFoundError(feat)
    df = pd.read_csv(feat, usecols=["timestamp", "src_ip", "dst_ip"])
    dev_ips = device_ips(device)

    source = "features"
    rows = None
    if mode == "attack_hard":
        pc = pcap_path(device, mode, out_dir)
        if not os.path.exists(pc):
            raise FileNotFoundError(f"attack_hard needs the pcap for port labels: {pc}")
        pcap_rows = read_pcap_rows(pc)
        if not pcap_matches_features(pcap_rows, df):
            raise RuntimeError(
                f"[ERROR] {pc} does not align with {feat} "
                f"({len(pcap_rows)} IPv4 pcap rows vs {len(df)} feature rows). "
                "Refusing to guess attack_hard labels."
            )
        rows = pcap_rows
        source = "pcap"
    else:
        rows = [(float(r.timestamp), str(r.src_ip), str(r.dst_ip), 0, 0, 0, 0)
                for r in df.itertuples(index=False)]

    out = pd.DataFrame({
        "row": range(len(rows)),
        "timestamp": [r[0] for r in rows],
        "src_ip": [r[1] for r in rows],
        "dst_ip": [r[2] for r in rows],
        "sport": [r[3] for r in rows],
        "dport": [r[4] for r in rows],
        "proto": [r[5] for r in rows],
        "size": [r[6] for r in rows],
    })
    out["role"] = [assign_role(mode, r[1], r[4], dev_ips) for r in rows]

    out["label_strict"] = out["role"].map(
        {"attack": 1, "benign": 0, "target_response": -1, "other": -1}
    ).astype(int)

    if mode in ATTACK_MODES:
        out["label_session"] = out["role"].map(
            {"attack": 1, "target_response": 1, "benign": 0, "other": -1}
        ).astype(int)
        # Refined: a target response is attack-related only when it answers an
        # attack port.  In attack mode all requests are attack; in attack_hard
        # only non-benign destination ports are attack.
        out["label_refined"] = out["role"].map(
            {"attack": 1, "benign": 0, "other": -1}
        )
        tr_mask = out["role"] == "target_response"
        if mode == "attack":
            out.loc[tr_mask, "label_refined"] = 1
        else:
            refined_tr = out["sport"].map(
                lambda p: 0 if int(p) in BENIGN_PORTS else 1
            )
            out.loc[tr_mask, "label_refined"] = refined_tr[tr_mask]
        out["label_refined"] = out["label_refined"].astype(int)
    else:
        out["label_session"] = out["label_strict"]
        out["label_refined"] = out["label_strict"]

    return out, source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="1-9",
                    help="e.g. 1-9 or comma list 1,3,5")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--label-dir", default=None,
                    help="default: <out-dir>/labels")
    args = ap.parse_args()

    label_dir = args.label_dir or os.path.join(args.out_dir, "labels")
    os.makedirs(label_dir, exist_ok=True)

    if "-" in args.devices:
        a, b = args.devices.split("-", 1)
        devices = list(range(int(a), int(b) + 1))
    else:
        devices = [int(x) for x in args.devices.split(",") if x]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    summary = []
    for d in devices:
        for mode in modes:
            try:
                labels, source = labels_for_mode(d, mode, args.out_dir)
            except Exception as exc:
                print(f"[SKIP] device{d} {mode}: {exc}")
                summary.append({"device": d, "mode": mode, "source": "ERROR",
                                "rows": 0, "attack": 0, "benign": 0,
                                "target_response": 0, "other": 0,
                                "note": str(exc)})
                continue

            out_path = os.path.join(label_dir, f"device{d}_{mode}_packet_labels.csv.gz")
            labels.to_csv(out_path, index=False, compression="gzip")
            counts = labels["role"].value_counts().to_dict()
            row = {
                "device": d, "mode": mode, "source": source,
                "rows": int(len(labels)),
                "attack": int(counts.get("attack", 0)),
                "benign": int(counts.get("benign", 0)),
                "target_response": int(counts.get("target_response", 0)),
                "other": int(counts.get("other", 0)),
                "note": "",
            }
            summary.append(row)
            print(f"[OK] device{d} {mode:<11} rows={row['rows']:>6} "
                  f"attack={row['attack']:>6} benign={row['benign']:>6} "
                  f"target={row['target_response']:>5} other={row['other']:>4} "
                  f"({source})")

    summ = pd.DataFrame(summary)
    summ_path = os.path.join(label_dir, "label_summary.csv")
    summ.to_csv(summ_path, index=False)
    print(f"\n[*] Labels   -> {label_dir}")
    print(f"[*] Summary  -> {summ_path}")


if __name__ == "__main__":
    main()
