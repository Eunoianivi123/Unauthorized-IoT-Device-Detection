#!/usr/bin/env python3
"""Stressed benign traffic generator.

Runs the device's NORMAL benign traffic (same profiles as device_traffic.py)
plus occasional NATURAL bursts (firmware/OTA sync, video re-buffering, bulk
cloud upload).  These bursts are plausible legitimate behaviour but push the
anomaly rate up, so a detector faces a realistic false-alarm challenge
instead of a trivially clean baseline.

Env knobs:
  BURST_EVERY_LO  min seconds between natural bursts  (default 20)
  BURST_EVERY_HI  max seconds between natural bursts  (default 45)
  BURST_RATE_LO   min packets/sec during a burst      (default 50)
  BURST_RATE_HI   max packets/sec during a burst      (default 120)
  BURST_SECS_LO   min burst duration in seconds       (default 0.6)
  BURST_SECS_HI   max burst duration in seconds       (default 1.8)
"""
import os
import random
import socket
import sys
import threading
import time

TARGET_IP = os.environ.get("TARGET_IP", "192.168.100.3")
DEVICE_ID = int(os.environ.get("DEVICE_ID", "1"))

BURST_EVERY = (float(os.environ.get("BURST_EVERY_LO", "30")),
               float(os.environ.get("BURST_EVERY_HI", "55")))
BURST_RATE = (float(os.environ.get("BURST_RATE_LO", "40")),
              float(os.environ.get("BURST_RATE_HI", "80")))
BURST_SECS = (float(os.environ.get("BURST_SECS_LO", "2")),
              float(os.environ.get("BURST_SECS_HI", "4")))

sys.path.insert(0, "/app")
import device_traffic as dt  # noqa: E402

if DEVICE_ID not in dt.DEVICES:
    print(f"[ERROR] DEVICE_ID must be 1-9, got {DEVICE_ID}")
    sys.exit(1)

DEVICE_NAME = dt.DEVICES[DEVICE_ID]
print(f"\n[Stress] {DEVICE_NAME} (ID={DEVICE_ID}) -> {TARGET_IP}")
print(f"[Stress] natural bursts every {BURST_EVERY}s, "
      f"{BURST_RATE[0]:.0f}-{BURST_RATE[1]:.0f} pps for {BURST_SECS[0]:.1f}-{BURST_SECS[1]:.1f}s\n")


def natural_bursts():
    """Occasional bulk UDP bursts to plausible cloud/media ports."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        time.sleep(random.uniform(*BURST_EVERY))
        dur = random.uniform(*BURST_SECS)
        rate = random.uniform(*BURST_RATE)
        end = time.time() + dur
        n = 0
        while time.time() < end:
            size = random.randint(400, 1400)
            s.sendto(bytes(random.randint(64, 255) for _ in range(size)),
                     (TARGET_IP, random.choice([554, 8080, 53, 443, 8081])))
            n += 1
            time.sleep(1.0 / rate)
        print(f"[Stress] natural burst: {n} packets")


def main():
    threads = dt.DEVICE_THREADS[DEVICE_ID]()
    for t in threads:
        t.daemon = True
        t.start()
        time.sleep(0.1)

    sb = threading.Thread(target=natural_bursts, daemon=True)
    sb.start()

    print(f"[{DEVICE_NAME}] {len(threads)} benign threads + natural-burst stress running.\n")
    while True:
        time.sleep(10)


if __name__ == "__main__":
    main()
