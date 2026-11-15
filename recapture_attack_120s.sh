#!/bin/bash
# Re-capture a full-duration standard attack for one device.
#
# Why: the canonical attack pcaps are cut at tcpdump's `-c 25000` packet cap
# (entrypoint.sh) after 0.1-5 s, so they do not cover the 120 s attack cycle.
# This script runs the same generator for the full duration with a high packet
# cap; tcpdump is stopped by the entrypoint's time-based kill, so the pcap
# spans the whole attack.
#
# Usage:
#   DEVICE_ID=1 ./recapture_attack_120s.sh            # pilot
#   DEVICE_ID=1 MAX_PACKETS=400000 ./recapture_attack_120s.sh
#
# Requirements:
#   - nbaiot-gateway running;
#   - nbaiot-identifier STOPPED (otherwise it may quarantine the device);
#   - ~2 GB free disk for the pilot (feature + results CSVs).
set -euo pipefail

DEVICE_ID="${DEVICE_ID:-1}"
DURATION="${DURATION:-120}"
MAX_PACKETS="${MAX_PACKETS:-2000000}"
RUN_TAG="${RUN_TAG:-recapture120}"
OUT="${OUT:-./output/recapture120}"

mkdir -p "${OUT}"

if ! docker ps --format '{{.Names}}' | grep -qx 'nbaiot-gateway'; then
    echo "[ERROR] nbaiot-gateway is not running. Start it first."
    exit 1
fi
if docker ps --format '{{.Names}}' | grep -qx 'nbaiot-identifier'; then
    echo "[ERROR] nbaiot-identifier is running. Stop it before an attack capture,"
    echo "        otherwise it may quarantine the device and truncate the run."
    exit 1
fi

LOG="/shared/device${DEVICE_ID}_attack_${RUN_TAG}_attack_packets.csv"
PCAP="${OUT}/device${DEVICE_ID}_attack_${RUN_TAG}_capture.pcap"
CSV="${OUT}/device${DEVICE_ID}_attack_${RUN_TAG}_capture_features.csv"
RES="${OUT}/device${DEVICE_ID}_attack_${RUN_TAG}_results.csv"

echo "[*] device-${DEVICE_ID}: ${DURATION}s attack"
echo "    max packets : ${MAX_PACKETS}"
echo "    output dir  : ${OUT}"
echo "    pcap        : ${PCAP}"
echo "    features    : ${CSV}"
echo "    results     : ${RES}"
echo "    attack log  : ${LOG}"

OUTPUT_DIR="${OUT}" docker compose --profile devices run --rm --no-deps \
    --entrypoint /app/entrypoint.sh \
    -v "$PWD/entrypoint.sh:/app/entrypoint.sh:ro" \
    -v "$PWD/pcap_csv_fast.py:/app/pcap_csv_fast.py:ro" \
    -v "$PWD/attack_traffic.py:/app/attack_traffic.py:ro" \
    -e MODE=attack \
    -e DEVICE_ID="${DEVICE_ID}" \
    -e TARGET_IP=192.168.100.3 \
    -e TRAFFIC_DURATION="${DURATION}" \
    -e MAX_PACKETS="${MAX_PACKETS}" \
    -e WAIT_QUARANTINE=0 \
    -e USE_FAST_EXTRACTOR=1 \
    -e OVERHEAD_LOG=1 \
    -e PYTHONUNBUFFERED=1 \
    -e RUN_TAG="${RUN_TAG}" \
    -e ATTACK_LOG_PATH="${LOG}" \
    device-"${DEVICE_ID}"

echo "[*] done"
ls -lh "${OUT}" | tail -20
