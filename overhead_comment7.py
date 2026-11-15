#!/usr/bin/env python3
"""Comment 7 - real system overhead: per-stage timing + CPU/RAM.

Part 1 (this script, offline): times the feature-extraction, inference, and
SIEM decision/logging stages on the canonical captures using the SAME code
paths the pipeline runs (pcap_csv.extract_features_from_pcap for STEP 2;
spoof_analysis.score for STEP 3; the CUSUM + JSON alert path from
identification_engine for decisioning). Each stage runs in an isolated
subprocess so its peak RSS (ru_maxrss) and CPU time are not polluted by
other stages, and per-packet throughput is reported against the real-time
capture rate.

Stage model load is timed separately (one-time cost, not per-capture).

Usage (inside nbaiot-pipeline image, /work = repo root, /output = ./output):
    python3 overhead_comment7.py [device_id, ...]   (default: 1)

Writes:
    /output/overhead_comment7_stages.json   raw per-stage rows
    prints the summary table to stdout
"""
import json
import os
import subprocess
import sys

OUT = "/output"
PY = sys.executable
DEVICES = [int(x) for x in sys.argv[1:]] or [1]

# Every stage snippet runs this tail to emit one STAGEJSON line.
CHILD_TAIL = """
import json as _j, resource as _r, time as _t
dt = _t.time() - _T0
_u = _r.getrusage(_r.RUSAGE_SELF)
_info = {"rows": rows, "wall_s": dt,
         "cpu_s": _u.ru_utime + _u.ru_stime,
         "peak_rss_mb": round(_u.ru_maxrss / 1024.0, 1)}
for _k in ("load_s", "alert_bytes"):
    if _k in globals():
        _info[_k] = globals()[_k]
print("STAGEJSON " + _j.dumps(_info))
"""


def stage_code(kind, device):
    feats_csv = f"{OUT}/device{device}_attack_capture_features.csv"
    pcap = f"{OUT}/device{device}_attack_capture.pcap"
    results = f"{OUT}/device{device}_attack_capture_results.csv"
    head = (
        "import sys; sys.path.insert(0, '/work'); "
        "import time as _t; _T0=_t.time(); "
        "import spoof_analysis as sa\n"
    )
    if kind == "feature_extraction":
        body = (
            f"from pcap_csv import extract_features_from_pcap\n"
            f"df = extract_features_from_pcap('{pcap}', output_csv='/tmp/feats_{device}.csv')\n"
            f"rows = len(df)\n"
        )
        return head + body + CHILD_TAIL
    if kind == "inference":
        body = (
            f"t0 = _t.time(); model, scaler = sa.load_model(); "
            f"globals()['load_s'] = _t.time() - t0\n"
            f"r = sa.score('{feats_csv}', model, scaler, sa.THRESHOLDS[{device}])\n"
            f"rows = len(r)\n"
        )
        return head + body + CHILD_TAIL
    if kind == "decision_logging":
        body = (
            "import os, pandas as pd, json\n"
            f"results = ('{results}' if os.path.exists('{results}') "
            f"else '{OUT}/device{device}_attack_results.csv')\n"
            f"df = pd.read_csv(results)\n"
            f"flags = df['anomaly'].astype(int).tolist(); rows = len(flags)\n"
            f"alarm, k, s = sa.cusum(flags, {device})\n"
            "alert_bytes = len(json.dumps({'event': 'CUSUM_TEST', 'device_id': "
            f"{device}, 'S': round(s,2), 'alarm': alarm}}))\n"
        )
        return head + body + CHILD_TAIL
    if kind == "model_load":
        body = "model, scaler = sa.load_model(); rows = 1\n"
        return head + body + CHILD_TAIL
    raise ValueError(kind)


def run_stage(kind, device):
    code = stage_code(kind, device)
    p = subprocess.run([PY, "-c", code], capture_output=True, text=True)
    info = {}
    for line in p.stdout.splitlines():
        if line.startswith("STAGEJSON "):
            info = json.loads(line[len("STAGEJSON "):].strip())
    info["stage"] = kind
    info["device_id"] = device
    if p.returncode != 0:
        info["error"] = (p.stderr or p.stdout).strip()[-500:]
    return info


def fmt_rows(info):
    rows = info.get("rows", 0)
    dt = info.get("wall_s")
    if not dt:
        return "n/a"
    return f"{rows / dt:,.0f} rows/s ({dt * 1000:,.0f} ms total; {1e6 * dt / max(rows, 1):.1f} us/row)"


def main():
    results = []
    for device in DEVICES:
        for kind in ("feature_extraction", "inference", "decision_logging"):
            info = run_stage(kind, device)
            results.append(info)
            print(f"[d{device} {kind}] {info}", flush=True)
    for kind in ("model_load",):
        info = run_stage(kind, 0)
        results.append(info)
        print(f"[{kind}] {info}", flush=True)

    with open(f"{OUT}/overhead_comment7_stages.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n==== OVERHEAD (reviewer comment 7) - offline stages ====")
    print("device stage                rows   wall(s)  cpu(s)  peakRSS(MB)  throughput")
    for info in results:
        rss = info.get("peak_rss_mb", "n/a")
        print(f"  d{info.get('device_id'):<4} {info.get('stage','?'):<20} "
              f"{info.get('rows',0):>6} {info.get('wall_s',0):>7.2f} "
              f"{info.get('cpu_s',0):>6.2f} {str(rss):>10}  {fmt_rows(info)}")
    model = next((i for i in results if i.get("stage") == "model_load"), {})
    print(f"\nmodel+scaler load : {model.get('wall_s','?'):.2f} s, "
          f"peak RSS {model.get('peak_rss_mb','?')} MB (one-time, per engine process)")
    print(f"\n[*] raw rows saved -> {OUT}/overhead_comment7_stages.json")


if __name__ == "__main__":
    main()