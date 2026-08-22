#!/bin/bash
# ============================================================
# entrypoint.sh - N-BaIoT traffic pipeline orchestrator
#
# Usage:
#   docker compose run --rm -e DEVICE_ID=1 pipeline
#   docker compose run --rm -e DEVICE_ID=3 -e MODE=attack pipeline
# ============================================================

# NOTE: do not use `set -e` - grep can return 1 (no match) in the wait loop.
set -uo pipefail

MODE="${MODE:-normal}"
DEVICE_ID="${DEVICE_ID:-1}"
SHARED_DIR="/shared"
MODEL_DIR="/model"

# ── Validate DEVICE_ID (1-9) ──────────────────────────────
if [ "${DEVICE_ID}" -lt 1 ] || [ "${DEVICE_ID}" -gt 9 ] 2>/dev/null; then
    echo "[ERROR] DEVICE_ID must be between 1 and 9. Got: ${DEVICE_ID}"
    exit 1
fi

# ── Device name lookup ────────────────────────────────────
case "${DEVICE_ID}" in
    1) DEVICE_NAME="Danmini_Doorbell" ;;
    2) DEVICE_NAME="Ecobee_Thermostat" ;;
    3) DEVICE_NAME="Ennio_Doorbell" ;;
    4) DEVICE_NAME="Philips_B120N10_Baby_Monitor" ;;
    5) DEVICE_NAME="Provision_PT_737E_Security_Camera" ;;
    6) DEVICE_NAME="Provision_PT_838_Security_Camera" ;;
    7) DEVICE_NAME="Samsung_SNH_1011_N_Webcam" ;;
    8) DEVICE_NAME="SimpleHome_XCS7_1002_WHT_Security_Camera" ;;
    9) DEVICE_NAME="SimpleHome_XCS7_1003_WHT_Security_Camera" ;;
esac

# ── Per-device benign-only threshold (median + 2.5*MAD of benign MSE) ──
case "${DEVICE_ID}" in
    1) THRESHOLD="8.6298"  ;;
    2) THRESHOLD="0.2253"  ;;
    3) THRESHOLD="2.1119"  ;;
    4) THRESHOLD="27.8245" ;;
    5) THRESHOLD="13.5779" ;;
    6) THRESHOLD="7.7438"  ;;
    7) THRESHOLD="4.2077"  ;;
    8) THRESHOLD="1.3466"  ;;
    9) THRESHOLD="3.7666"  ;;
esac

MODEL_FILE="${MODEL_DIR}/autoencoder_allinone.keras"
SCALER_FILE="${MODEL_DIR}/scaler_allinone.pkl"

echo ""
echo "======================================================"
echo "      N-BaIoT IoT Botnet Detection Pipeline"
echo "======================================================"
echo ""
echo "[Config] DEVICE    : ${DEVICE_ID} - ${DEVICE_NAME}"
echo "[Config] MODE      : ${MODE}"
echo "[Config] MODEL     : ${MODEL_FILE}"
echo "[Config] SCALER    : ${SCALER_FILE}"
echo "[Config] SHARED    : ${SHARED_DIR}"

# ── Wait for interface on the TARGET_IP subnet ────────────
# Authorized devices start on .100.x (immediate).
# Shadow devices start on .100.x and wait until the identity
# engine quarantines them to .200.x.
EXPECTED_PREFIX=$(echo "${TARGET_IP}" | sed 's/\.[0-9]*$/\./')

echo "[Config] Waiting for interface on subnet ${EXPECTED_PREFIX}0/24 ..."
echo "[Config] (Shadow devices wait here until identity engine quarantines them)"
INTERFACE=""
OWN_IP=""
WAIT_SECS=0
while true; do
    LINE=$(ip -o -4 addr show | awk '{print $2, $4}' | grep "^[^ ]* ${EXPECTED_PREFIX}" || true)
    if [ -n "${LINE}" ]; then
        INTERFACE=$(echo "${LINE}" | awk '{print $1}')
        OWN_IP=$(echo "${LINE}"    | awk '{print $2}' | cut -d/ -f1)
        echo "[Config] Interface ready: ${INTERFACE} / ${OWN_IP} (waited ${WAIT_SECS}s)"
        break
    fi
    if [ $((WAIT_SECS % 5)) -eq 0 ] && [ "${WAIT_SECS}" -gt 0 ]; then
        CURRENT=$(ip -o -4 addr show | awk '{print $2, $4}' | grep -v "^lo " | tr '\n' '  ' || true)
        echo "[Config] ${WAIT_SECS}s: waiting for ${EXPECTED_PREFIX}x ... current: ${CURRENT}"
    fi
    sleep 1
    WAIT_SECS=$((WAIT_SECS + 1))
done

# ── Fallback interface detection ──────────────────────────
if [ -z "${INTERFACE}" ]; then
    echo "[Config] WARNING: subnet ${EXPECTED_PREFIX}0/24 not found, using fallback."
    INTERFACE=$(ip -o -4 addr show | awk '{print $2, $4}' | \
        grep -E '192\.168\.(100|200)\.' | awk '{print $1}' | head -n1 || true)
    OWN_IP=$(ip -o -4 addr show dev "${INTERFACE}" 2>/dev/null | \
        awk '{print $4}' | cut -d/ -f1 | head -n1 || true)
fi
if [ -z "${INTERFACE}" ]; then
    INTERFACE=$(ip -o link show up | awk -F': ' '$2 != "lo" {print $2}' | head -n1 || true)
fi
if [ -z "${INTERFACE}" ]; then INTERFACE="any"; fi

# Safety: never target our own IP
if [ "${TARGET_IP}" = "${OWN_IP}" ]; then
    TARGET_IP=$(echo "${OWN_IP}" | sed 's/\.[0-9]*$/.1/')
fi
export TARGET_IP
export DEVICE_ID
echo "[Config] INTERFACE : ${INTERFACE}  (auto-detected)"
echo "[Config] OWN IP    : ${OWN_IP}"
echo "[Config] TARGET_IP : ${TARGET_IP}"
echo ""

mkdir -p "${SHARED_DIR}"

# ── Per-run file names ────────────────────────────────────
PREFIX="${SHARED_DIR}/device${DEVICE_ID}_${MODE}"
PCAP="${PREFIX}_capture.pcap"
CSV="${PREFIX}_capture_features.csv"
RESULTS="${PREFIX}_results.csv"

# ── Mode-dependent parameters ─────────────────────────────
case "${MODE}" in
    attack)
        TRAFFIC_DURATION="${TRAFFIC_DURATION:-120}"
        TRAFFIC_LABEL="ATTACK"
        TRAFFIC_SCRIPT="/app/attack_traffic.py"
        MAX_PACKETS="${MAX_PACKETS:-25000}"
        ;;
    attack_hard)
        # Stealthy intermittent attacks mixed with normal device traffic
        TRAFFIC_DURATION="${TRAFFIC_DURATION:-120}"
        TRAFFIC_LABEL="HARD-ATTACK"
        TRAFFIC_SCRIPT="/app/attack_hard.py"
        MAX_PACKETS="${MAX_PACKETS:-20000}"
        ;;
    stress)
        # Benign traffic with occasional natural bursts (FPR stress)
        TRAFFIC_DURATION=70
        TRAFFIC_LABEL="STRESS-BENIGN"
        TRAFFIC_SCRIPT="/app/device_traffic_stress.py"
        MAX_PACKETS="${MAX_PACKETS:-10000}"
        ;;
    normal)
        TRAFFIC_DURATION=70
        TRAFFIC_LABEL="BENIGN"
        TRAFFIC_SCRIPT="/app/device_traffic.py"
        MAX_PACKETS="${MAX_PACKETS:-10000}"
        ;;
esac

# ══════════════════════════════════════════════════════════
# STEP 1 - Generate traffic + capture PCAP with tcpdump
# ══════════════════════════════════════════════════════════
echo "------------------------------------------------------"
echo " STEP 1 -- ${DEVICE_NAME}"
echo "          ${TRAFFIC_LABEL} traffic for ${TRAFFIC_DURATION}s"
echo "------------------------------------------------------"

echo "[1/4] Starting tcpdump on '${INTERFACE}' -> ${PCAP}"
tcpdump -i "${INTERFACE}" -w "${PCAP}" -c "${MAX_PACKETS}" 2>/tmp/tcpdump.log &
TCPDUMP_PID=$!
sleep 1

echo "[1/4] Launching ${TRAFFIC_LABEL} traffic..."
python3 "${TRAFFIC_SCRIPT}" &
TRAFFIC_PID=$!

sleep "${TRAFFIC_DURATION}"

echo "[1/4] ${TRAFFIC_DURATION}s window complete -- stopping traffic..."
kill "${TRAFFIC_PID}" 2>/dev/null || true
wait "${TRAFFIC_PID}" 2>/dev/null || true
sleep 3

kill "${TCPDUMP_PID}" 2>/dev/null || true
wait "${TCPDUMP_PID}" 2>/dev/null || true

echo "[1/4] tcpdump stats:"
cat /tmp/tcpdump.log 2>/dev/null || true

PCAP_SIZE=$(stat -c%s "${PCAP}" 2>/dev/null || echo 0)
echo "[1/4] PCAP size: ${PCAP_SIZE} bytes"

if [ "${PCAP_SIZE}" -lt 100 ]; then
    echo "[ERROR] PCAP is empty. No traffic captured on '${INTERFACE}'."
    exit 1
fi
echo "[1/4] PCAP saved -> ${PCAP}"

# ══════════════════════════════════════════════════════════
# STEP 2 - Convert PCAP to 115 N-BaIoT features (CSV)
# ══════════════════════════════════════════════════════════
echo ""
echo "------------------------------------------------------"
echo " STEP 2 -- Convert PCAP -> CSV (N-BaIoT Features)"
echo "------------------------------------------------------"

echo "[2/4] Extracting 115 N-BaIoT features from PCAP..."
python3 - <<PYEOF
import sys
sys.path.insert(0, "/app")
from pcap_csv import extract_features_from_pcap
df = extract_features_from_pcap("${PCAP}", output_csv="${CSV}")
print(f"      Rows extracted : {len(df)}")
print(f"      Features       : {len(df.columns) - 3}  (+ 3 metadata cols)")
if len(df) == 0:
    print("[ERROR] Zero IP rows extracted from PCAP.")
    sys.exit(1)
PYEOF

echo "[2/4] CSV saved -> ${CSV}"

# ══════════════════════════════════════════════════════════
# STEP 3 - Score with the autoencoder model
# ══════════════════════════════════════════════════════════
echo ""
echo "------------------------------------------------------"
echo " STEP 3 -- Run Autoencoder Model (Device ${DEVICE_ID})"
echo "------------------------------------------------------"

echo "[3/4] Running testmodel.py with autoencoder_allinone.keras ..."
python3 /app/testmodel.py \
    --features  "${CSV}" \
    --model     "${MODEL_FILE}" \
    --scaler    "${SCALER_FILE}" \
    --threshold "${THRESHOLD}" \
    --device-name "${DEVICE_NAME}" \
    --output    "${RESULTS}"

if [ -f "results.csv" ]; then
    mv results.csv "${RESULTS}"
    echo "[3/4] Results saved (fallback mv) -> ${RESULTS}"
elif [ -f "${RESULTS}" ]; then
    echo "[3/4] Results saved -> ${RESULTS}"
fi

# ══════════════════════════════════════════════════════════
# STEP 4 - Wait for quarantine decision (attack mode only)
# ══════════════════════════════════════════════════════════
echo ""
echo "------------------------------------------------------"
echo " STEP 4 -- Wait for Quarantine Decision"
echo "------------------------------------------------------"

# Benign traffic won't trigger the identifier's anomaly watcher
if [ "${MODE}" != "attack" ]; then
    echo "[4/4] MODE=${MODE} -- skipping quarantine wait (live attack demo uses MODE=attack)."
else
    WAIT=0
    while [ $WAIT -lt 60 ]; do
        STILL_ON_IOT=$(ip -o -4 addr show | grep "192\.168\.100\." || true)
        if [ -z "${STILL_ON_IOT}" ]; then
            echo "[4/4] Identifier removed us from iot_net -- quarantine confirmed."
            break
        fi
        echo "[4/4] ${WAIT}s: still on iot_net, waiting for identifier..."
        sleep 3
        WAIT=$((WAIT + 3))
    done

    if [ -n "$(ip -o -4 addr show | grep '192\.168\.100\.' || true)" ]; then
        echo "[4/4] No quarantine action taken -- traffic classified as BENIGN."
    fi
fi

echo ""
echo "[4/4] Final network state:"
ip -o -4 addr show | grep -v "^.*lo " || true

echo ""
echo "======================================================"
echo "  Pipeline Complete! Device: ${DEVICE_NAME}"
echo "======================================================"
