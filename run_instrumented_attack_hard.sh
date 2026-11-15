#!/usr/bin/env bash
# Short instrumented attack_hard run: capture packets AND log every attack
# packet emitted by attack_hard.py, so the destination-port label heuristic can
# be checked against ground truth.
#
# Produces (under ./output/instrumented/):
#   device1_attack_hard_instrumented_capture.pcap
#   device1_attack_hard_instrumented_attack_packets.csv
#   device1_attack_hard_instrumented_attack.log
#
# The run is deliberately short (~60 s) and low-rate.  It does not touch the
# canonical output/device1_attack_hard_* artifacts.
set -euo pipefail

cd "$(dirname "$0")"
DEVICE_ID="${DEVICE_ID:-1}"
DURATION="${DURATION:-60}"
MAX_PACKETS="${MAX_PACKETS:-8000}"
BURST_SECS="${BURST_SECS:-6}"
GAP_SECS="${GAP_SECS:-10}"
RATE="${RATE:-120}"

OUT_DIR="output/instrumented"
mkdir -p "$OUT_DIR"
PCAP="/shared/instrumented/device${DEVICE_ID}_attack_hard_instrumented_capture.pcap"
LOG="/shared/instrumented/device${DEVICE_ID}_attack_hard_instrumented_attack_packets.csv"
RUNLOG="/shared/instrumented/device${DEVICE_ID}_attack_hard_instrumented_attack.log"

rm -f "output/instrumented/device${DEVICE_ID}_attack_hard_instrumented_capture.pcap" \
      "output/instrumented/device${DEVICE_ID}_attack_hard_instrumented_attack_packets.csv" \
      "output/instrumented/device${DEVICE_ID}_attack_hard_instrumented_attack.log"

# The gateway must be up (target 192.168.100.3).
if ! docker ps --format '{{.Names}}' | grep -qx 'nbaiot-gateway'; then
    echo "[*] starting gateway ..."
    docker compose --profile gateway up -d gateway
fi

echo "[*] device-${DEVICE_ID}: ${DURATION}s attack_hard capture"
echo "    pcap  : $PCAP"
echo "    log   : $LOG"

docker compose --profile devices run --rm --no-deps \
    --entrypoint bash \
    -v "$PWD/attack_hard.py:/app/attack_hard.py:ro" \
    -e MODE=attack_hard \
    -e DEVICE_ID="${DEVICE_ID}" \
    -e TARGET_IP=192.168.100.3 \
    -e PYTHONUNBUFFERED=1 \
    -e ATTACK_LOG_PATH="$LOG" \
    -e BURST_SECS="$BURST_SECS" \
    -e GAP_SECS="$GAP_SECS" \
    -e RATE="$RATE" \
    device-"${DEVICE_ID}" -lc "
        set -u
        tcpdump -i eth0 -w '$PCAP' -c $MAX_PACKETS 2>/tmp/tcpdump.log &
        TP=\$!
        sleep 1
        python3 /app/attack_hard.py > '$RUNLOG' 2>&1 &
        AP=\$!
        sleep $DURATION
        kill \$AP 2>/dev/null || true
        sleep 2
        kill \$TP 2>/dev/null || true
        wait \$TP 2>/dev/null || true
        echo '--- tcpdump stats ---'
        cat /tmp/tcpdump.log 2>/dev/null || true
    "

echo "[*] done"
ls -lh "output/instrumented/device${DEVICE_ID}_attack_hard_instrumented_"* 2>/dev/null || true
