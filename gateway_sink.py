#!/usr/bin/env python3
"""Listens on common IoT ports so benign_traffic.py has valid targets."""
import socket
import threading
import time

def udp_sink(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    print(f"[Gateway] UDP sink listening on :{port}")
    while True:
        s.recvfrom(4096)

for p in [9999, 123, 53, 554]:
    threading.Thread(target=udp_sink, args=(p,), daemon=True).start()

print("[Gateway] All sinks running.")
while True:
    time.sleep(60)
