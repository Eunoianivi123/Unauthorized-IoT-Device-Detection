#!/usr/bin/env python3
"""Complexity check: reference (pcap_csv.py) vs fast (pcap_csv_fast.py) extractor.

Runs both extractors on the same pcap truncated to several packet counts and
records wall time, so the per-packet growth of each can be compared:
reference is O(n^2) in capture length, fast is O(n) in the length of the
largest sliding window (bounded by the capture for captures < 60 s).

Usage (pipeline image):
  python3 bench_extract_scaling.py --pcap output_overhead/device1_attack_capture.pcap
"""
import argparse
import os
import sys
import time

import pandas as pd
from scapy.all import PcapReader, Raw, wrpcap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--pcap", default="output_overhead/device1_attack_capture.pcap")
parser.add_argument("--sizes", default="2500,5000,10000,20000")
parser.add_argument("--out", default="output/extract_scaling_results.csv")
parser.add_argument("--tmp", default="/tmp/scaling")
args = parser.parse_args()

os.makedirs(args.tmp, exist_ok=True)
sizes = [int(s) for s in args.sizes.split(",")]


def truncate(src, dst, n):
    """Byte-level pcap truncation: copy the global header + first n packet records."""
    import struct
    with open(src, "rb") as f:
        gh = f.read(24)
        if len(gh) < 24:
            raise ValueError("not a pcap file")
        magic = gh[:4]
        endian = "<" if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1") else ">"
        rec_struct = struct.Struct(endian + "IIII")
        written = 0
        with open(dst, "wb") as out:
            out.write(gh)
            while written < n:
                ph = f.read(16)
                if len(ph) < 16:
                    break
                ts_sec, ts_usec, incl_len, orig_len = rec_struct.unpack(ph)
                data = f.read(incl_len)
                out.write(ph)
                out.write(data)
                written += 1
    return written


def run(mod, pcap):
    t0 = time.monotonic()
    df = mod.extract_features_from_pcap(pcap)
    dt = time.monotonic() - t0
    return dt, len(df)


import pcap_csv
import pcap_csv_fast

rows = []
for n in sizes:
    sub = os.path.join(args.tmp, f"cap_{n}.pcap")
    got = truncate(args.pcap, sub, n)
    print(f"\n=== capture truncated to {got} packets ===", flush=True)

    t_ref, n_ref = run(pcap_csv, sub)
    print(f"    reference : {t_ref:8.1f}s  ({n_ref/max(t_ref,1e-9):7.1f} rows/s, "
          f"{t_ref/max(n_ref,1)*1000:6.2f} ms/row)", flush=True)

    t_fast, n_fast = run(pcap_csv_fast, sub)
    print(f"    fast      : {t_fast:8.1f}s  ({n_fast/max(t_fast,1e-9):7.1f} rows/s, "
          f"{t_fast/max(n_fast,1)*1000:6.2f} ms/row)", flush=True)

    rows.append({
        "packets": got, "rows": n_ref,
        "reference_seconds": round(t_ref, 2),
        "reference_rows_per_s": round(n_ref / t_ref, 1),
        "reference_ms_per_row": round(t_ref / n_ref * 1000, 3),
        "fast_seconds": round(t_fast, 2),
        "fast_rows_per_s": round(n_fast / t_fast, 1),
        "fast_ms_per_row": round(t_fast / n_fast * 1000, 3),
        "speedup": round(t_ref / t_fast, 1),
    })
    pd.DataFrame(rows).to_csv(args.out, index=False)

print("\n" + "=" * 78)
df = pd.DataFrame(rows)
print(df.to_string(index=False))
print(f"\n-> {args.out}")
print("=" * 78)
