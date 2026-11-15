#!/usr/bin/env python3
"""Validate the attack_hard destination-port label heuristic against ground truth.

The instrumented run (``run_instrumented_attack_hard.sh``) writes:

  * a pcap of the device's eth0
  * a CSV of every attack packet the generator actually sent
    (timestamp, emitter, proto, src/dst IP, src/dst port, payload size)

This script checks that the port-based rule used by
``build_packet_labels.py`` recovers the logged attack packets, and that it
does not misclassify the benign device workload.

Run in the pipeline image:
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
      -v "$PWD:/work" -w /work --entrypoint python3 \
      nbaiot-pipeline:latest verify_instrumented_labels.py
"""
from __future__ import annotations

import argparse
import os
from collections import Counter

import pandas as pd
from scapy.all import IP, TCP, UDP, PcapReader

from build_packet_labels import (BENIGN_PORTS, TARGET_IP, device_ips)

DEVICE_ID = 1


def read_pcap(path: str):
    rows = []
    with PcapReader(path) as reader:
        for pkt in reader:
            if not pkt.haslayer(IP):
                continue
            ip = pkt[IP]
            sport = dport = 0
            proto = int(ip.proto)
            flags = 0
            payload_len = 0
            if pkt.haslayer(TCP):
                tcp = pkt[TCP]
                sport, dport = int(tcp.sport), int(tcp.dport)
                proto, flags = 6, int(tcp.flags)
                payload_len = len(tcp.payload)
            elif pkt.haslayer(UDP):
                udp = pkt[UDP]
                sport, dport = int(udp.sport), int(udp.dport)
                proto = 17
                payload_len = len(udp.payload)
            rows.append({
                "ts": float(pkt.time), "src": str(ip.src), "dst": str(ip.dst),
                "sport": sport, "dport": dport, "proto": proto,
                "flags": flags, "payload_len": payload_len,
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=DEVICE_ID)
    ap.add_argument("--dir", default="output/instrumented")
    args = ap.parse_args()

    d = args.device
    pcap = os.path.join(args.dir, f"device{d}_attack_hard_instrumented_capture.pcap")
    log = os.path.join(args.dir, f"device{d}_attack_hard_instrumented_attack_packets.csv")
    for p in (pcap, log):
        if not os.path.exists(p):
            raise SystemExit(f"[ERROR] missing {p} -- run run_instrumented_attack_hard.sh")

    pkts = read_pcap(pcap)
    logged = pd.read_csv(log)
    dev_ips = device_ips(d)

    # ── pcap packet index for the logged attack packets ──
    syn = Counter((r["sport"], r["dport"]) for r in pkts
                  if r["proto"] == 6 and (r["flags"] & 0x02))
    udp = Counter((r["sport"], r["dport"]) for r in pkts
                  if r["proto"] == 17)

    matched = 0
    unmatched = []
    for r in logged.itertuples(index=False):
        proto = int(r.proto)
        sport = int(r.src_port) if pd.notna(r.src_port) else 0
        dport = int(r.dst_port)
        key = (sport, dport)
        if proto == 6 and syn.get(key, 0) > 0:
            syn[key] -= 1
            matched += 1
        elif proto == 17 and udp.get(key, 0) > 0:
            udp[key] -= 1
            matched += 1
        else:
            unmatched.append({
                "timestamp": r.timestamp, "emitter": r.emitter,
                "proto": proto, "src_port": sport, "dst_port": dport,
                "payload_size": r.payload_size,
            })

    # ── port-based labels, exactly as build_packet_labels.py assigns them ──
    port_attack = [r for r in pkts
                   if r["src"] in dev_ips and r["dst"] == TARGET_IP
                   and r["dport"] not in BENIGN_PORTS]
    port_benign = [r for r in pkts
                   if r["src"] in dev_ips and r["dst"] == TARGET_IP
                   and r["dport"] in BENIGN_PORTS]
    target_resp = [r for r in pkts if r["src"] == TARGET_IP]

    port_syn = sum(1 for r in port_attack if r["proto"] == 6 and (r["flags"] & 0x02))
    port_udp = sum(1 for r in port_attack if r["proto"] == 17)
    port_tcp_nonsyn = sum(1 for r in port_attack
                          if r["proto"] == 6 and not (r["flags"] & 0x02))

    collision = int(logged["dst_port"].isin(BENIGN_PORTS).sum()) if len(logged) else 0

    summary = {
        "device": d,
        "pcap_ipv4": len(pkts),
        "logged_attack": len(logged),
        "logged_matched_to_pcap": matched,
        "logged_unmatched": len(unmatched),
        "port_attack_total": len(port_attack),
        "port_attack_tcp_syn": port_syn,
        "port_attack_udp": port_udp,
        "port_attack_tcp_non_syn": port_tcp_nonsyn,
        "port_benign_device_packets": len(port_benign),
        "logged_dst_port_in_benign_set": collision,
        "target_responses": len(target_resp),
    }

    out_csv = os.path.join(args.dir, f"device{d}_verify_instrumented_labels.csv")
    pd.DataFrame([summary]).to_csv(out_csv, index=False)
    if unmatched:
        pd.DataFrame(unmatched).to_csv(
            os.path.join(args.dir, f"device{d}_verify_instrumented_unmatched.csv"),
            index=False,
        )

    print("=" * 78)
    print("  Instrumented attack_hard label validation")
    print("=" * 78)
    for k, v in summary.items():
        print(f"  {k:<34} {v}")
    print("\n  Reading:")
    print("  * logged_matched_to_pcap should equal logged_attack (or be very close);")
    print("    the log is the generator's ground truth.")
    print("  * port_attack_total >= port_attack_tcp_syn + port_attack_udp; the")
    print("    remainder is TCP ACK/FIN/RST generated by the device kernel for the")
    print("    same attack connection, which the strict label intentionally keeps.")
    print("  * logged_dst_port_in_benign_set is the rare random-port collision that")
    print("    the heuristic would label benign.")
    if unmatched:
        print(f"\n  [!] {len(unmatched)} logged packets had no pcap match -- see")
        print(f"      {os.path.join(args.dir, f'device{d}_verify_instrumented_unmatched.csv')}")
    print(f"\n[*] summary -> {out_csv}")


if __name__ == "__main__":
    main()
