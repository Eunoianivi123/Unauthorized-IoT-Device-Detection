#!/usr/bin/env python3
"""Hard-mode attack traffic generator: stealthy, intermittent attacks.

Runs the device's NORMAL benign traffic (same profiles as device_traffic.py)
and, on top of it, launches short low-rate attack bursts separated by long
benign gaps.  Unlike attack_traffic.py's continuous floods, the anomaly
fraction stays small and a detector must accumulate evidence across many
bursts -- a realistic stress test for the CUSUM decision rule.

Env knobs:
  BURST_SECS  duration of each attack burst in seconds   (default 2)
  GAP_SECS    benign quiet period between bursts         (default 15)
  RATE        attack packets/sec during a burst          (default 50)
  ATTACK_LOG_PATH  optional CSV of every attack packet sent (ground truth
                   for validating the destination-port label heuristic)
"""
import os
import random
import socket
import sys
import threading
import time

TARGET_IP = os.environ.get("TARGET_IP", "192.168.100.3")
DEVICE_ID = int(os.environ.get("DEVICE_ID", "1"))
BURST_SECS = float(os.environ.get("BURST_SECS", "6"))
GAP_SECS = float(os.environ.get("GAP_SECS", "45"))
RATE = int(os.environ.get("RATE", "120"))
START_DELAY = float(os.environ.get("START_DELAY", "0"))
PROBE = os.environ.get("PROBE", "0") == "1"

sys.path.insert(0, "/app")
import device_traffic as dt  # noqa: E402

if DEVICE_ID not in dt.DEVICES:
    print(f"[ERROR] DEVICE_ID must be 1-9, got {DEVICE_ID}")
    sys.exit(1)

DEVICE_NAME = dt.DEVICES[DEVICE_ID]
print(f"\n[HardAttack] {DEVICE_NAME} (ID={DEVICE_ID}) -> {TARGET_IP}")
print(f"[HardAttack] bursts {BURST_SECS}s @ {RATE} pps, gap {GAP_SECS}s\n")

# Optional ground-truth log.  Writing is opt-in so normal runs are unchanged.
ATTACK_LOG_PATH = os.environ.get("ATTACK_LOG_PATH", "").strip()
_log_lock = threading.Lock()
_log_fh = None


def _log_attack(emitter, proto, src_ip, src_port, dst_port, payload_size):
    global _log_fh
    if not ATTACK_LOG_PATH:
        return
    if _log_fh is None:
        new = not os.path.exists(ATTACK_LOG_PATH)
        _log_fh = open(ATTACK_LOG_PATH, "a", buffering=1)
        if new:
            _log_fh.write("timestamp,emitter,proto,src_ip,src_port,"
                          "dst_ip,dst_port,payload_size\n")
    with _log_lock:
        _log_fh.write(f"{time.time():.6f},{emitter},{proto},{src_ip},"
                      f"{src_port},{TARGET_IP},{dst_port},{payload_size}\n")
        _log_fh.flush()


# ── Single-packet attack emitters (one per burst tick) ────
def _emit_syn():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setblocking(False)
    port = random.randint(1, 65535)
    try:
        s.connect_ex((TARGET_IP, port))
        src_ip, src_port = s.getsockname()[:2]
    except Exception:
        src_ip, src_port = "", 0
    finally:
        _log_attack("syn", 6, src_ip, src_port, port, 0)
        s.close()


def _emit_udp_random():
    size = random.randint(512, 1024)
    port = random.randint(1, 65535)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(bytes(size), (TARGET_IP, port))
    src_ip, src_port = s.getsockname()[:2]
    _log_attack("udp_random", 17, src_ip, src_port, port, size)
    s.close()


def _emit_udp_junk():
    port = random.randint(1, 65535)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(b"X" * 64, (TARGET_IP, port))
    src_ip, src_port = s.getsockname()[:2]
    _log_attack("udp_junk", 17, src_ip, src_port, port, 64)
    s.close()


def _emit_bashlite():
    size = random.randint(256, 768)
    port = random.randint(1024, 9999)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(bytes([0xAB] * size), (TARGET_IP, port))
    src_ip, src_port = s.getsockname()[:2]
    _log_attack("bashlite", 17, src_ip, src_port, port, size)
    s.close()


EMITTERS = [_emit_syn, _emit_udp_random, _emit_udp_junk, _emit_bashlite]


def attacker():
    """Cycle: short paced attack burst, then a long benign gap."""
    if START_DELAY > 0:
        print(f"[HardAttack] waiting {START_DELAY:.0f}s before first burst ...")
        time.sleep(START_DELAY)
    probe_durs = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0] if PROBE else None
    while True:
        if probe_durs is not None:
            durs = list(probe_durs)
            random.shuffle(durs)
            for d in durs:
                emit = random.choice(EMITTERS)
                burst_pkts = int(RATE * d)
                print(f"[HardAttack] probe burst: {emit.__name__} "
                      f"{d}s @ {RATE} pps (~{burst_pkts} pkts) ...")
                _run_burst(emit, d)
                time.sleep(GAP_SECS + random.uniform(0.0, 3.0))
        else:
            emit = random.choice(EMITTERS)
            print(f"[HardAttack] burst: {emit.__name__} for {BURST_SECS}s @ {RATE} pps ...")
            _run_burst(emit, BURST_SECS)
            time.sleep(GAP_SECS + random.uniform(0.0, 3.0))


def _run_burst(emit, duration):
    end = time.time() + duration
    interval = 1.0 / RATE
    next_t = time.time()
    while time.time() < end:
        emit()
        next_t += interval
        time.sleep(max(0.0, next_t - time.time()))


def main():
    threads = dt.DEVICE_THREADS[DEVICE_ID]()
    for t in threads:
        t.daemon = True
        t.start()
        time.sleep(0.1)

    atk = threading.Thread(target=attacker, daemon=True)
    atk.start()

    print(f"[{DEVICE_NAME}] {len(threads)} benign threads + intermittent attacker running.\n")
    while True:
        time.sleep(10)


if __name__ == "__main__":
    main()
