# N-BaIoT Shadow IoT Detection Pipeline - Code Flow

This document describes how the code in this folder works together.

## 1. Architecture Overview

```
                     +-----------------------------------------------------------+
                     |                     IDENTITY ENGINE                        |
                     |   identification_engine.py  (nbaiot-identifier)           |
                     |   - whitelists / quarantines / blocks containers          |
                     +----------------------------+------------------------------+
                                                  | watches /output/ for results
                     +----------------------------v------------------------------+
   iot_net (192.168.100.0/24)        quarantine_network (192.168.200.0/24)
   +--------------------------------+        +--------------------------------+
   |  device-1..9  (authorized)     |        |  shadow-device-1..9 (rogue)    |
   |  gateway       (192.168.100.3) |        |  quarantine-gateway            |
   |  identifier    (192.168.100.2) |        |                                |
   +----------------+---------------+        +----------------+---------------+
                    |                                             |
                    | Docker bridge (tcpdump captures here)
                    v
   +--------------------------------------------------------------+
   |  entrypoint.sh  (runs inside every device container)         |
   |   1. generate traffic  2. capture pcap  3. extract features  |
   |   4. score with autoencoder  5. wait for quarantine          |
   +--------------------------------------------------------------+
```

## 2. Modules and Their Files

| Module | Files | Role |
|---|---|---|
| Orchestration | `compose.yml` | 22 services, 2 networks, static MAC/IP assignment, profiles |
| Images | `Dockerfile`, `Dockerfile.gateway`, `Dockerfile.identifier` | Build the pipeline / gateway / identity containers |
| Traffic generation | `device_traffic.py` (benign, per-device), `attack_traffic.py` (Mirai/BASHLITE), `benign_traffic.py` (generic) | Emit realistic IoT or botnet traffic |
| Capture | `entrypoint.sh` + `tcpdump` | Capture packets to a PCAP file |
| Feature extraction | `pcap_csv.py` | PCAP -> 115 N-BaIoT features (CSV) |
| ML scoring | `testmodel.py` + `model/autoencoder_allinone.keras`, `model/scaler_allinone.pkl` | MSE reconstruction error + anomaly rate |
| Threshold tuning | `threshold_finder.py` | Compute MSE threshold from benign data |
| Identity / quarantine | `identification_engine.py` + `whitelisted.json` | MAC whitelist, OUI spoof check, quarantine, blocklist, SIEM logs |
| Gateway sink | `gateway.sh`, `gateway_sink.py` | TCP/UDP sink so traffic has a valid target |

## 3. End-to-End Data Flow

```
device container (entrypoint.sh)
        |
        |-- MODE=normal  -> device_traffic.py  (realistic benign traffic)
        |-- MODE=attack  -> attack_traffic.py  (Mirai / BASHLITE floods)
        |
        |-- tcpdump -i <iface> -w device{N}_{mode}_capture.pcap
        |
        v
   pcap_csv.py  extract_features_from_pcap()
        |   115 N-BaIoT features per packet
        v
   device{N}_{mode}_capture_features.csv
        |
        |-- testmodel.py  (loads autoencoder + scaler, MinMax scale, predict)
        |-- MSE = mean((input - reconstructed)^2)
        |-- anomaly = MSE > threshold   (per-device ROC threshold)
        v
   device{N}_{mode}_results.csv   (anomaly 0/1 per row + mse + threshold)
        |
        |  written to ./output (shared volume /output)
        v
   identification_engine.py  _anomaly_watcher_loop()
        |  anomaly_rate > 30%  ?
        +-- YES -> _full_ban_container() -> disconnect + blocklist MAC/name
        +-- NO  -> device stays authorized
```

## 4. Identity Engine Flow (`identification_engine.py`)

```
start_identification()
  |
  +-- load_whitelist()            -> mac_index / prefix_index / name_index
  +-- load mac_blocklist.json     -> restore permanent bans
  +-- get_ieee_database()         -> OUI vendor DB (cached in /output)
  +-- ensure_quarantine_network() -> create 192.168.200.0/24 if missing
  |
  +-- thread: _quarantine_rescan_loop  (re-check quarantined every 60s)
  +-- thread: _anomaly_watcher_loop    (poll /output results CSVs every 15s)
  +-- scan existing containers        -> process_container()
  |
  +-- watch docker events (container start)
        |
        +-- _handle_start_event()
              | 1. brief grace + tc rate-limit on veth
              | 2. wait until iot_net appears
              | 3. process_container()
              |
              +-- auth check (Tier 1/2/3)
              |    1. MAC in whitelist?        -> AUTHORIZED
              |    2. name prefix (DYNAMIC)?   -> AUTHORIZED
              |    3. exact name (STATIC)?     -> AUTHORIZED
              |    else                        -> UNAUTHORIZED
              |
              +-- [FIX 1] OUI spoof check for authorized devices
              |    OUI vendor mismatch -> treated as UNAUTHORIZED
              |
              +-- authorized:
              |    record in device_types.json + inventory log
              |    remove tc limit
              |
              +-- unauthorized:
                    stop container
                    disconnect from iot_net
                    ban MAC + name prefix (persisted)
                    move to quarantine_network
```

## 5. Entry Point Flow (`entrypoint.sh`)

```
validate DEVICE_ID (1-9)
device name + threshold lookup
wait for interface on TARGET_IP subnet (shadow devices wait until quarantined)
detect interface + own IP (fallbacks)
STEP 1  generate traffic + tcpdump capture  (attack=3s / benign=70s)
STEP 2  pcap_csv -> 115 features CSV
STEP 3  testmodel.py -> results CSV (anomaly rate)
STEP 4  attack mode only: wait until removed from iot_net (= quarantine confirmed)
```

## 6. How to Run

```bash
# Build the base image (once)
docker compose --profile build build

# Full infra stack: gateway + identity engine
docker compose --profile gateway --profile identity up -d

# Authorized device, benign traffic
docker compose --profile devices up device-1

# Authorized device, attack traffic (autoencoder should flag it)
docker compose --profile devices -e MODE=attack up device-1

# Shadow (rogue) device - MAC not whitelisted -> auto-quarantined
docker compose --profile shadow run --rm shadow-device-1
```

## 7. Output Artifacts (./output, shared volume /output)

| File | Contents |
|---|---|
| `device{N}_{mode}_capture.pcap` | raw packet capture |
| `device{N}_{mode}_capture_features.csv` | 115 features per packet |
| `device{N}_{mode}_results.csv` | MSE + anomaly flags per packet |
| `device_inventory_log.json` | auth events (SIEM-ready, JSONL) |
| `security_alerts.json` | quarantine/ban/spoof alerts (SIEM-ready, JSONL) |
| `device_types.json` | IP/MAC -> device mapping for anomaly correlation |
| `mac_blocklist.json` | permanently banned MACs + name prefixes |
| `oui_cache.json` | cached OUI vendor database |
