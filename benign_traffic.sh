#!/bin/bash
# benign_traffic.sh - standalone benign IoT traffic simulator (curl/ping/nc based)
echo "[IoT Device] Starting benign traffic simulation..."

while true; do
    # Sensor data reporting (HTTP)
    curl -s http://192.168.100.3 > /dev/null 2>&1

    # NTP-like ping
    ping -c 2 8.8.8.8 > /dev/null 2>&1

    # Small UDP heartbeat
    echo "heartbeat" | nc -u -w1 192.168.100.3 9999 > /dev/null 2>&1

    # Periodic HTTP GET (firmware check)
    wget -q --spider http://example.com > /dev/null 2>&1

    sleep 5
done
