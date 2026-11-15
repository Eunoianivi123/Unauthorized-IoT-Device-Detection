#!/usr/bin/env python3
"""MAC whitelist authorization + quarantine engine for the N-BaIoT pipeline.

Auth tiers (checked in order):
  1. Exact MAC match   - fixed-MAC real devices
  2. Name prefix match - mac="DYNAMIC" (compose run assigns a random MAC each run)
  3. Exact name match  - mac="STATIC"  (long-running services)
  Anything else       - UNAUTHORIZED -> quarantine

Extra defenses:
  [FIX 1] OUI spoof detection  - cross-check resolved OUI vs declared manufacturer
  [FIX 2] Quarantine rescan    - periodic re-check of quarantined containers
  [FIX 3] Fast auth window     - tc rate-limit during the container start grace period
  [FIX 4] Anomaly correlation  - quarantine on ML anomaly rate above threshold
"""

import datetime
import glob
import json
import math
import os
import subprocess
import threading
import time

import docker
import requests

# ──────────────────────────────────────────────────────────────────────────────
# Tunables
# ──────────────────────────────────────────────────────────────────────────────

STARTUP_GRACE_SECONDS        = 0.25   # FIX 3: was 1.0
QUARANTINE_RESCAN_INTERVAL   = 60     # FIX 2: seconds between quarantine rescans
ANOMALY_QUARANTINE_THRESHOLD = 30.0   # legacy anomaly % cutoff (superseded by CUSUM, kept for reference)
ANOMALY_POLL_INTERVAL        = 15     # FIX 4: seconds between results-file polls
OUI_MISMATCH_BLOCK           = True   # FIX 1: quarantine on OUI/manufacturer mismatch

# ── CUSUM (one-sided SPRT) anomaly decisioning ───────────────────────────────
# Every packet is a Bernoulli observation: anomalous with probability p_ben
# under H0 (benign) or p_atk under H1 (attack). Each packet contributes a
# log-likelihood-ratio weight and the statistic S = max(0, S + w) is accumulated;
# an ATTACK is declared the first time S >= CUSUM_A. Recalibrated 2026-09-17 with
# the stage-1 CUSUM winner model (median + 2.5*MAD threshold on benign r1, p_ben
# = r1 flag rate); p_atk=0.3, A=250: 9/9 attack TPR (~210 packets after onset),
# 0/9 benign-r2 FPR, 7/9 attack_hard TPR (devices 5,7 evade), 2/9 stress false
# alarms (devices 8,9). See output/cusum_recalibration_results.csv.
P_ATK = 0.3
CUSUM_A = 250.0
DEVICE_P_BEN = {
    1: 0.041, 2: 0.195, 3: 0.050, 4: 0.031, 5: 0.035,
    6: 0.044, 7: 0.061, 8: 0.046, 9: 0.047,
}

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

WHITELIST_FILE       = "/app/whitelisted.json"
OUI_CACHE            = "/output/oui_cache.json"
DEVICE_MAP_FILE      = "/output/device_types.json"
INVENTORY_LOG        = "/output/device_inventory_log.json"
ALERT_LOG            = "/output/security_alerts.json"
MAC_BLOCKLIST        = "/output/mac_blocklist.json"
PROCESSED_FILES_LOG  = "/output/processed_results.json"  # survives restarts
OUTPUT_DIR           = "/output"

# ──────────────────────────────────────────────────────────────────────────────
# Network names
# ──────────────────────────────────────────────────────────────────────────────

IOT_NETWORK_NAME          = "nbaiot-pipeline_iot_net"
IOT_NETWORK_SUBSTR        = "iot_net"
QUARANTINE_NETWORK_NAME   = "nbaiot-pipeline_quarantine_network"
QUARANTINE_NETWORK_SUBSTR = "quarantine"
QUARANTINE_SUBNET         = "192.168.200.0/24"
QUARANTINE_GATEWAY        = "192.168.200.1"

# ──────────────────────────────────────────────────────────────────────────────
# Shared state (guarded by _state_lock)
# ──────────────────────────────────────────────────────────────────────────────

_state_lock              = threading.Lock()
_quarantined_containers  = set()   # container names currently in soft-quarantine
_anomaly_checked_files   = set()   # results CSVs already acted upon
_banned_macs             = set()   # MACs permanently banned (anomaly confirmed)
_banned_name_prefixes    = set()   # Name prefixes banned (for DYNAMIC/random-MAC containers)


# ──────────────────────────────────────────────────────────────────────────────
# Alert logger
# ──────────────────────────────────────────────────────────────────────────────

def _write_alert(alert: dict):
    """Append a structured alert to security_alerts.json (one JSON obj per line)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    alert.setdefault("timestamp", datetime.datetime.now().isoformat())
    with open(ALERT_LOG, "a") as f:
        f.write(json.dumps(alert) + "\n")
        f.flush()


# ──────────────────────────────────────────────────────────────────────────────
# MAC blocklist — persistent banned MACs (anomaly-confirmed devices)
# ──────────────────────────────────────────────────────────────────────────────

def _load_mac_blocklist() -> tuple:
    """Load banned MACs and name prefixes from disk into memory. Called once at startup."""
    if not os.path.exists(MAC_BLOCKLIST):
        return set(), set()
    try:
        with open(MAC_BLOCKLIST) as f:
            data = json.load(f)
        macs     = set(data.get("banned_macs", []))
        prefixes = set(data.get("banned_name_prefixes", []))
        if macs:
            print(f"[STARTUP] {len(macs)} permanently banned MAC(s) restored from disk.", flush=True)
            for m in sorted(macs):
                print(f"          ✖  {m}", flush=True)
        if prefixes:
            print(f"[STARTUP] {len(prefixes)} permanently banned name prefix(es) restored from disk.", flush=True)
            for p in sorted(prefixes):
                print(f"          ✖  {p}", flush=True)
        return macs, prefixes
    except Exception as exc:
        print(f"[STARTUP] WARNING: could not load MAC blocklist: {exc}", flush=True)
        return set(), set()


def _save_mac_blocklist():
    """Persist the in-memory _banned_macs and _banned_name_prefixes sets to disk."""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with _state_lock:
            macs_copy     = set(_banned_macs)
            prefixes_copy = set(_banned_name_prefixes)
        # Merge with existing on-disk entries (handles concurrent engine restarts)
        if os.path.exists(MAC_BLOCKLIST):
            try:
                with open(MAC_BLOCKLIST) as f:
                    existing = json.load(f)
                macs_copy     |= set(existing.get("banned_macs", []))
                prefixes_copy |= set(existing.get("banned_name_prefixes", []))
            except Exception:
                pass
        with open(MAC_BLOCKLIST, "w") as f:
            json.dump({
                "description":          "Permanently banned MACs and name prefixes — anomaly confirmed",
                "updated":              datetime.datetime.now().isoformat(),
                "banned_macs":          sorted(macs_copy),
                "banned_name_prefixes": sorted(prefixes_copy),
            }, f, indent=2)
        print(f"[BLOCKLIST] Saved to {MAC_BLOCKLIST}  "
              f"({len(macs_copy)} MAC(s), {len(prefixes_copy)} prefix(es))", flush=True)
    except Exception as exc:
        print(f"[BLOCKLIST] ERROR — could not save blocklist to {MAC_BLOCKLIST}: {exc}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Processed results files — persisted so restarts don't reprocess old CSVs
# ──────────────────────────────────────────────────────────────────────────────

def _load_processed_files() -> set:
    """Load the set of already-processed results CSVs from disk."""
    if not os.path.exists(PROCESSED_FILES_LOG):
        return set()
    try:
        with open(PROCESSED_FILES_LOG) as f:
            data = json.load(f)
        files = set(data.get("processed", []))
        if files:
            print(f"[STARTUP] {len(files)} results file(s) already processed — will skip.", flush=True)
        return files
    except Exception as exc:
        print(f"[STARTUP] WARNING: could not load processed files log: {exc}", flush=True)
        return set()


def _mark_file_processed(fpath: str):
    """
    Add fpath to the in-memory set AND persist to disk immediately.
    Called only after a file has been fully acted upon.
    """
    with _state_lock:
        _anomaly_checked_files.add(fpath)

    # Merge with existing on-disk entries (safe across restarts)
    existing = set()
    if os.path.exists(PROCESSED_FILES_LOG):
        try:
            with open(PROCESSED_FILES_LOG) as f:
                existing = set(json.load(f).get("processed", []))
        except Exception:
            pass
    existing.add(fpath)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PROCESSED_FILES_LOG, "w") as f:
        json.dump({
            "description": "Results CSVs already processed by the anomaly watcher",
            "updated":     datetime.datetime.now().isoformat(),
            "processed":   sorted(existing),
        }, f, indent=2)


def _is_file_processed(fpath: str) -> bool:
    with _state_lock:
        return fpath in _anomaly_checked_files


def _is_mac_banned(mac: str) -> bool:
    with _state_lock:
        return mac.upper() in _banned_macs


def _is_name_prefix_banned(container_name: str) -> str | None:
    """Return the matching banned prefix if container_name starts with one, else None."""
    with _state_lock:
        for prefix in _banned_name_prefixes:
            if container_name.startswith(prefix):
                return prefix
    return None


def _ban_name_prefix(prefix: str, reason: str, container_name: str):
    """Add a name prefix to the permanent blocklist (for DYNAMIC/random-MAC containers)."""
    with _state_lock:
        _banned_name_prefixes.add(prefix)
    _save_mac_blocklist()
    print(f"[NAME BLOCKED] Name prefix '{prefix}' permanently banned — {reason}", flush=True)
    _write_alert({
        "event":     "NAME_PREFIX_PERMANENTLY_BANNED",
        "prefix":    prefix,
        "container": container_name,
        "reason":    reason,
    })


def _ban_mac(mac: str, reason: str, container_name: str):
    """Add a MAC to the permanent blocklist and persist it."""
    mac = mac.upper()
    with _state_lock:
        _banned_macs.add(mac)
    _save_mac_blocklist()
    print(f"[MAC BLOCKED] {mac} permanently banned — {reason}", flush=True)
    _write_alert({
        "event":     "MAC_PERMANENTLY_BANNED",
        "mac":       mac,
        "container": container_name,
        "reason":    reason,
    })

def _full_ban_container(container, mac: str, reason: str, quarantine_net, iot_net, name_prefix: str = None):
    """Tier 2 enforcement: disconnect from all networks and permanently ban by
    MAC and/or name prefix so any future container with them is rejected."""
    name = container.name
    print(f"[FULL BAN] Container : {name}", flush=True)
    print(f"[FULL BAN] MAC       : {mac}", flush=True)
    if name_prefix:
        print(f"[FULL BAN] Name prefix banned : {name_prefix}", flush=True)
    print(f"[FULL BAN] Reason    : {reason}", flush=True)

    for net, label in [(iot_net, "iot_net"), (quarantine_net, "quarantine_network")]:
        if net is None:
            continue
        try:
            net.disconnect(container)
            print(f"[FULL BAN] Disconnected from {label}.", flush=True)
        except docker.errors.APIError:
            pass  # already disconnected — not worth logging

    _ban_mac(mac, reason, name)
    if name_prefix:
        _ban_name_prefix(name_prefix, reason, name)

    _write_alert({
        "event":       "FULL_BAN",
        "container":   name,
        "mac":         mac,
        "name_prefix": name_prefix,
        "reason":      reason,
    })


# ──────────────────────────────────────────────────────────────────────────────
# Whitelist loader
# ──────────────────────────────────────────────────────────────────────────────

def load_whitelist():
    """
    Parse whitelisted.json into three lookup structures:

      mac_index    : { "AA:BB:CC:DD:EE:FF" : entry }  — exact MAC match
      prefix_index : { "nbaiot-pipeline"   : entry }  — name PREFIX match (DYNAMIC)
      name_index   : { "nbaiot-identifier" : entry }  — exact name match  (STATIC)
    """
    with open(WHITELIST_FILE) as f:
        data = json.load(f)

    mac_index    = {}
    prefix_index = {}
    name_index   = {}

    for entry in data.get("devices", []):
        mac  = entry.get("mac", "").upper()
        name = entry.get("device_name", "")

        if mac == "DYNAMIC":
            prefix_index[name] = entry
        elif mac == "STATIC":
            name_index[name] = entry
        else:
            mac_index[mac] = entry

    print(f"[STARTUP] Whitelist loaded — {len(mac_index)} fixed MACs, "
          f"{len(prefix_index)} dynamic name-prefix, "
          f"{len(name_index)} static name entries.", flush=True)

    return mac_index, prefix_index, name_index


# ──────────────────────────────────────────────────────────────────────────────
# Authorization check
# ──────────────────────────────────────────────────────────────────────────────

def check_whitelist(name: str, mac: str, mac_index: dict, prefix_index: dict, name_index: dict):
    """Return (authorized: bool, entry | None, reason: str).

    Priority: 1. exact MAC -> 2. name prefix (DYNAMIC) -> 3. exact name (STATIC).

    DYNAMIC containers get a fresh MAC every run, so they are matched by name.
    compose v2 names them "<project>-<service>-run-<hash>", so the whitelist
    entry "nbaiot-pipeline" (project="nbaiot", service="pipeline") yields the
    allowed patterns "nbaiot-pipeline", "nbaiot-pipeline-run-" and
    "nbaiot-pipeline-pipeline-run-". Shadow services ("nbaiot-pipeline-shadow-...")
    do not match any of these, so they correctly fall through to UNAUTHORIZED.
    """
    if mac in mac_index:
        return True, mac_index[mac], "MAC_MATCH"

    for wl_name, entry in prefix_index.items():
        # Pattern A — exact name
        if name == wl_name:
            return True, entry, "PREFIX_MATCH"

        # Pattern B — "<wl_name>-run-<hash>"  (older compose / standalone run)
        if name.startswith(wl_name + "-run-"):
            return True, entry, "PREFIX_MATCH"

        # Pattern C — "<wl_name>-<service>-run-<hash>"  (compose v2 project prefix)
        # service = last dash-token of wl_name  →  "nbaiot-pipeline" → "pipeline"
        service = wl_name.rsplit("-", 1)[-1]
        compose_v2_prefix = wl_name + "-" + service + "-run-"
        if name.startswith(compose_v2_prefix):
            return True, entry, "PREFIX_MATCH"

    if name in name_index:
        return True, name_index[name], "NAME_MATCH"

    return False, None, "UNAUTHORIZED"


# ──────────────────────────────────────────────────────────────────────────────
# [FIX 1] OUI spoofing detection
# ──────────────────────────────────────────────────────────────────────────────

# Docker/virtual OUI prefixes always acceptable for infrastructure containers
_VIRTUAL_OUI_PREFIXES = {
    "024200", "024201", "024202", "024203",   # Docker bridge MACs (02:42:xx)
    "000000",                                  # placeholder / unset
}

# Loose (case-insensitive, partial) keywords that the resolved OUI vendor must
# contain per declared manufacturer. Infrastructure entries skip this check.
_OUI_ACCEPT_KEYWORDS = {
    "Danmini":        ["danmini"],
    "ecobee Inc":     ["ecobee"],
    "Ennio":          ["ennio"],
    "Philips":        ["philips", "tp link"],   # Philips uses TP-Link silicon
    "Provision-ISR":  ["provision", "isr", "dahua"],
    "Samsung Electronics": ["samsung"],
    "SimpleHome":     ["simplehome", "simple home", "hangzhou"],
    "Infrastructure": [],   # skip check for infra containers
}


def check_oui_spoof(mac: str, whitelist_manufacturer: str, oui_db: dict) -> tuple[bool, str]:
    """FIX 1 - Cross-check the live OUI against the expected manufacturer.

    Returns (spoofed: bool, detail: str). Spoofing is suspected when the OUI
    is not a known virtual/Docker prefix AND the resolved vendor does not
    contain any accepted keyword. Infrastructure devices are skipped (they
    always get Docker-assigned virtual MACs).
    """
    if whitelist_manufacturer == "Infrastructure":
        return False, "skipped (infrastructure)"

    oui_prefix = mac.replace(":", "")[:6].upper()

    if oui_prefix in _VIRTUAL_OUI_PREFIXES:
        return False, f"virtual/docker OUI ({oui_prefix}) — acceptable"

    resolved = oui_db.get(oui_prefix, "").lower()

    keywords = _OUI_ACCEPT_KEYWORDS.get(whitelist_manufacturer, [])
    # If no keywords defined for this manufacturer, we cannot validate — pass.
    if not keywords:
        return False, f"no OUI keywords defined for '{whitelist_manufacturer}' — skipped"

    for kw in keywords:
        if kw in resolved:
            return False, f"OUI '{resolved}' matches keyword '{kw}'"

    if not resolved:
        # OUI not in database — could be a new device, don't block.
        return False, "OUI not in database — cannot verify, passing"

    detail = (
        f"OUI prefix {oui_prefix} resolves to '{resolved}', "
        f"expected keywords {keywords} for declared manufacturer '{whitelist_manufacturer}'"
    )
    return True, detail


# ──────────────────────────────────────────────────────────────────────────────
# OUI database
# ──────────────────────────────────────────────────────────────────────────────

def get_ieee_database() -> dict:
    if os.path.exists(OUI_CACHE):
        print("[STARTUP] Loading OUI database from cache ...", flush=True)
        with open(OUI_CACHE) as f:
            return json.load(f)

    url = "https://maclookup.app/downloads/csv-database/get-db?apiKey=free"
    print("[STARTUP] Downloading OUI database (first run) ...", flush=True)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        mapping = {}
        for line in r.text.splitlines()[1:]:
            parts = line.split(",")
            if len(parts) >= 2:
                raw = parts[0].strip().upper().replace(":", "").replace("-", "")
                org = parts[1].strip().replace('"', "")
                if len(raw) >= 6:
                    mapping[raw[:6]] = org
        if mapping:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(OUI_CACHE, "w") as f:
                json.dump(mapping, f)
            print(f"[STARTUP] OUI database ready — {len(mapping):,} entries cached.", flush=True)
            return mapping
    except Exception as exc:
        print(f"[STARTUP] WARNING: OUI download failed ({exc}) — spoof detection disabled.", flush=True)
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# Docker network helpers
# ──────────────────────────────────────────────────────────────────────────────

def _find_network(client, exact: str, substr: str):
    try:
        return client.networks.get(exact)
    except docker.errors.NotFound:
        pass
    for net in client.networks.list():
        if substr in net.name:
            return net
    return None


def ensure_quarantine_network(client):
    net = _find_network(client, QUARANTINE_NETWORK_NAME, QUARANTINE_NETWORK_SUBSTR)
    if net is None:
        print(f"[STARTUP] Creating quarantine network ({QUARANTINE_SUBNET}) ...", flush=True)
        net = client.networks.create(
            QUARANTINE_NETWORK_NAME,
            driver="bridge",
            internal=True,
            ipam=docker.types.IPAMConfig(
                pool_configs=[
                    docker.types.IPAMPool(
                        subnet=QUARANTINE_SUBNET,
                        gateway=QUARANTINE_GATEWAY,
                    )
                ]
            ),
        )
        print(f"[STARTUP] Quarantine network ready.", flush=True)
    return net


# ──────────────────────────────────────────────────────────────────────────────
# [FIX 3] Traffic-control shaping during the auth window
# ──────────────────────────────────────────────────────────────────────────────

def _get_container_veth(container_id: str) -> str | None:
    """
    Find the host-side veth interface for a container by reading its
    /sys/class/net/<iface>/ifindex and matching it via `ip link`.
    Returns the veth name (e.g. "veth3a4b5c6") or None.
    """
    try:
        result = subprocess.run(
            ["ip", "link", "show"],
            capture_output=True, text=True, timeout=3
        )
        # Docker veths are named vethXXXXXX
        for line in result.stdout.splitlines():
            if "veth" in line and "@" not in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    return parts[1].strip().split("@")[0]
    except Exception:
        pass
    return None


def _tc_rate_limit(iface: str, rate: str = "1kbit"):
    """Apply a near-zero tc rate limit on a host veth interface."""
    try:
        subprocess.run(
            ["tc", "qdisc", "add", "dev", iface, "root", "tbf",
             "rate", rate, "burst", "1600", "latency", "50ms"],
            capture_output=True, timeout=3
        )
    except Exception:
        pass


def _tc_remove_limit(iface: str):
    """Remove tc shaping from a host veth interface."""
    try:
        subprocess.run(
            ["tc", "qdisc", "del", "dev", iface, "root"],
            capture_output=True, timeout=3
        )
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Device map persistence
# ──────────────────────────────────────────────────────────────────────────────

def load_device_map() -> dict:
    if os.path.exists(DEVICE_MAP_FILE):
        try:
            with open(DEVICE_MAP_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_device_map(device_map: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(DEVICE_MAP_FILE, "w") as f:
        json.dump(device_map, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Quarantine action (shared by auth check and anomaly watcher)
# ──────────────────────────────────────────────────────────────────────────────

def _quarantine_container(container, reason: str, quarantine_net, iot_net, extra_fields: dict = None):
    """Move a container to quarantine_network and disconnect it from iot_net.
    Idempotent - safe to call repeatedly."""
    name = container.name
    with _state_lock:
        if name in _quarantined_containers:
            return  # already handled
        _quarantined_containers.add(name)

    print(f"[QUARANTINE] Container : {name}", flush=True)
    print(f"[QUARANTINE] Reason    : {reason}", flush=True)

    try:
        quarantine_net.connect(container)
        print(f"[QUARANTINE] Moved to quarantine network (192.168.200.0/24).", flush=True)
    except docker.errors.APIError as exc:
        print(f"[QUARANTINE] Connect skipped — {exc.explanation}", flush=True)

    try:
        if iot_net is not None:
            iot_net.disconnect(container)
            print(f"[QUARANTINE] Removed from iot_net.", flush=True)
    except docker.errors.APIError as exc:
        print(f"[QUARANTINE] Disconnect skipped — {exc.explanation}", flush=True)

    alert = {
        "event":     "QUARANTINE",
        "container": name,
        "reason":    reason,
    }
    if extra_fields:
        alert.update(extra_fields)
    _write_alert(alert)
    
# _____________________________________________________________________________
# Core: Disconnects the device from the network
# _____________________________________________________________________________

def disconnect_from_iot(container, iot_net):
    """Disconnect a container from iot_net only, without moving to quarantine."""
    try:
        if iot_net is not None:
            iot_net.disconnect(container)
            print(f"[DISCONNECT] {container.name} removed from iot_net.", flush=True)
    except docker.errors.APIError as exc:
        print(f"[DISCONNECT] Failed to disconnect {container.name} — {exc.explanation}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Core: process one container
# ──────────────────────────────────────────────────────────────────────────────

def process_container(container, oui_db, mac_index, prefix_index, name_index, quarantine_net, iot_net):
    try:
        # Always reload to get fresh NetworkSettings.
        # Catch NotFound in case the container exited between the start
        # event and this reload (e.g. --rm containers that exit quickly).
        try:
            container.reload()
        except docker.errors.NotFound:
            return  # container exited before auth check — silent skip
        name     = container.name
        networks = container.attrs["NetworkSettings"]["Networks"]

        # Find iot_net attachment
        iot_net_key = None
        for net_name in networks:
            if IOT_NETWORK_SUBSTR in net_name:
                iot_net_key = net_name
                break

        if iot_net_key is None:
            return  # not on iot_net — not our concern

        net_info = networks[iot_net_key]
        mac = net_info.get("MacAddress", "").upper()
        ip  = net_info.get("IPAddress", "")

        if not mac:
            return  # no MAC yet — Docker still wiring

        # ── Tier 0: Permanent MAC blocklist check ─────────────
        # If this MAC was previously banned (anomaly confirmed), immediately
        # disconnect from all networks — no further checks needed.
        if _is_mac_banned(mac):
            print(f"[BLOCKED] {name} — MAC {mac} is permanently banned, rejecting reconnect.", flush=True)
            _write_alert({
                "event":     "BANNED_MAC_RECONNECT_ATTEMPT",
                "container": name,
                "mac":       mac,
                "ip":        ip,
            })
            _full_ban_container(container, mac, "BANNED_MAC_RECONNECT", quarantine_net, iot_net)
            return

        # ── Tier 0b: Name prefix blocklist check (DYNAMIC containers) ──
        # Catches pipeline containers that get a new random MAC each run.
        banned_prefix = _is_name_prefix_banned(name)
        if banned_prefix:
            print(f"[BLOCKED] {name} — name matches banned prefix '{banned_prefix}', rejecting.", flush=True)
            _write_alert({
                "event":         "BANNED_PREFIX_RECONNECT_ATTEMPT",
                "container":     name,
                "mac":           mac,
                "ip":            ip,
                "banned_prefix": banned_prefix,
            })
            _full_ban_container(container, mac, "BANNED_NAME_PREFIX_RECONNECT",
                                quarantine_net, iot_net, name_prefix=banned_prefix)
            return

        # OUI lookup
        oui_prefix   = mac.replace(":", "")[:6]
        manufacturer = oui_db.get(oui_prefix, "Unknown / Virtual Device")

        # ── Whitelist check ───────────────────────────────────
        authorized, entry, reason = check_whitelist(name, mac, mac_index, prefix_index, name_index)

        # ── [FIX 1] OUI spoof check for authorized devices ───
        oui_spoofed    = False
        oui_spoof_detail = ""
        if authorized and entry and OUI_MISMATCH_BLOCK:
            declared_mfr = entry.get("manufacturer", "")
            oui_spoofed, oui_spoof_detail = check_oui_spoof(mac, declared_mfr, oui_db)
            if oui_spoofed:
                print(f"[OUI SPOOF] {name} — MAC {mac} passes whitelist but OUI mismatch detected.", flush=True)
                print(f"[OUI SPOOF] Expected manufacturer : {declared_mfr}", flush=True)
                print(f"[OUI SPOOF] OUI resolves to      : {manufacturer}", flush=True)
                print(f"[OUI SPOOF] Action               : Treating as UNAUTHORIZED → quarantine", flush=True)
                _write_alert({
                    "event":       "OUI_SPOOF_SUSPECTED",
                    "container":   name,
                    "mac":         mac,
                    "ip":          ip,
                    "oui_prefix":  oui_prefix,
                    "oui_resolved": manufacturer,
                    "declared_mfr": declared_mfr,
                    "detail":      oui_spoof_detail,
                })
                # Treat as unauthorized
                authorized = False
                reason     = "OUI_SPOOF_SUSPECTED"

        if authorized:
            device_id   = entry["device_id"]
            device_name = entry["device_name"]
            role        = entry["role"]
            is_dynamic  = (entry.get("mac", "").upper() == "DYNAMIC")
            status      = f"AUTHORIZED ({reason})"

            print(
                f"[AUTH] ✔  {device_name}  |  MAC: {mac}  |  IP: {ip}  |  Role: {role}",
                flush=True,
            )

            # Derive the name prefix that uniquely identifies this service across runs.
            # For DYNAMIC entries (pipeline), compose v2 names containers:
            #   <project>-<service>-run-<hash>  e.g. nbaiot-pipeline-pipeline-run-abc
            # The stable prefix is everything up to "-run-".
            name_prefix = None
            if is_dynamic and "-run-" in name:
                name_prefix = name.rsplit("-run-", 1)[0]  # "nbaiot-pipeline-pipeline"

            device_map      = load_device_map()
            device_map[mac] = {
                "device_id":    device_id,
                "device_name":  device_name,
                "role":         role,
                "manufacturer": manufacturer,
                "ip":           ip,
                "is_dynamic":   is_dynamic,
                "name_prefix":  name_prefix,
                "oui_verified": not oui_spoofed,
                "oui_detail":   oui_spoof_detail,
            }
            save_device_map(device_map)

        else:
            status = "UNAUTHORIZED"

            if reason != "OUI_SPOOF_SUSPECTED":
                print(f"[AUTH] ✖  UNAUTHORIZED  |  Container: {name}  |  MAC: {mac}  |  IP: {ip}", flush=True)
                print(f"[AUTH]    Action:  network.", flush=True)
                _write_alert({
                    "event":        "UNAUTHORIZED_DEVICE",
                    "container":    name,
                    "mac":          mac,
                    "ip":           ip,
                    "oui_prefix":   oui_prefix,
                    "manufacturer": manufacturer,
                    "match_reason": reason,
                })

            try:
                container.stop(timeout=5)
                print(f"[UNKNOWN MAC] Container {name} stopped.", flush=True)
            except docker.errors.APIError as exc:
                print(f"[UNKNOWN MAC] Failed to stop container {name} — {exc.explanation}", flush=True)
            
            disconnect_from_iot(container, iot_net)

            # Ban now so mac_blocklist.json exists even if the anomaly watcher never fires
            _ban_mac(mac, f"UNAUTHORIZED_{reason}", name)

            # Also ban by stable name prefix (shadow containers get a fresh MAC each run)
            shadow_name_prefix = None
            if "-run-" in name:
                shadow_name_prefix = name.rsplit("-run-", 1)[0]
            elif name.startswith("nbaiot-pipeline-shadow"):
                shadow_name_prefix = name
            elif name.startswith("shadow-device"):
                shadow_name_prefix = name

            # Never ban the authorised pipeline prefix by accident
            _SAFE_PREFIXES = {"nbaiot-pipeline-pipeline", "nbaiot-pipeline"}
            if shadow_name_prefix and shadow_name_prefix not in _SAFE_PREFIXES:
                _ban_name_prefix(shadow_name_prefix, f"UNAUTHORIZED_{reason}", name)

            # Extract DEVICE_ID from env so the anomaly watcher can map results -> container
            shadow_device_id = None
            try:
                env_vars = container.attrs.get("Config", {}).get("Env", []) or []
                for env in env_vars:
                    if env.startswith("DEVICE_ID="):
                        shadow_device_id = int(env.split("=", 1)[1])
                        break
            except Exception:
                pass

            # After quarantine, get the new IP on quarantine_network
            quarantine_ip = ip
            try:
                container.reload()
                for net_name, net_info in container.attrs["NetworkSettings"]["Networks"].items():
                    if QUARANTINE_NETWORK_SUBSTR in net_name:
                        quarantine_ip = net_info.get("IPAddress", ip)
                        break
            except Exception:
                pass

            device_map      = load_device_map()
            device_map[mac] = {
                "device_id":    shadow_device_id,
                "device_name":  name,
                "role":         "SHADOW",
                "manufacturer": manufacturer,
                "ip":           quarantine_ip,
                "oui_verified": False,
                "oui_detail":   oui_spoof_detail or "unauthorized",
            }
            save_device_map(device_map)

        # Append to inventory log
        log_entry = {
            "timestamp":    datetime.datetime.now().isoformat(),
            "mac":          mac,
            "ip":           ip,
            "device_id":    entry["device_id"]   if entry else None,
            "device_name":  entry["device_name"] if entry else name,
            "status":       status,
        }
        with open(INVENTORY_LOG, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            f.flush()

    except Exception as exc:
        print(f"[ID] Error processing {getattr(container, 'name', '?')}: {exc}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# [FIX 2] Periodic quarantine re-check thread
# ──────────────────────────────────────────────────────────────────────────────

def _quarantine_rescan_loop(client, oui_db, mac_index, prefix_index, name_index, quarantine_net):
    """FIX 2 - Periodically rescans quarantined containers:
    skip whitelisted ones (e.g. the identifier lives on both networks),
    re-disconnect any that escaped back to iot_net, and re-log an alert."""
    print(f"[STARTUP] Quarantine rescan thread active (every {QUARANTINE_RESCAN_INTERVAL}s).", flush=True)

    while True:
        time.sleep(QUARANTINE_RESCAN_INTERVAL)
        try:
            quarantine_net.reload()
            q_containers = quarantine_net.containers
        except Exception as exc:
            print(f"[RESCAN] ERROR: could not list quarantine containers — {exc}", flush=True)
            continue

        if not q_containers:
            continue

        iot_net = _find_network(client, IOT_NETWORK_NAME, IOT_NETWORK_SUBSTR)

        for container in q_containers:
            try:
                container.reload()
                cname = container.name

                # Skip whitelisted containers — the identifier is intentionally
                # on both networks and must never be treated as an escape attempt
                nets_now = container.attrs["NetworkSettings"]["Networks"]
                q_mac = ""
                for net_name, info in nets_now.items():
                    if QUARANTINE_NETWORK_SUBSTR in net_name:
                        q_mac = info.get("MacAddress", "").upper()
                        break

                authorized, _, _ = check_whitelist(
                    cname, q_mac, mac_index, prefix_index, name_index
                )
                if authorized:
                    continue   # legitimate dual-network container — skip silently

                attached_nets = nets_now

                # Check if unauthorized container is back on iot_net
                for net_name in attached_nets:
                    if IOT_NETWORK_SUBSTR in net_name:
                        print(
                            f"[RESCAN] {cname} escaped quarantine — re-joined iot_net, disconnecting again.",
                            flush=True
                        )
                        _write_alert({
                            "event":     "QUARANTINE_ESCAPE_ATTEMPT",
                            "container": cname,
                            "detail":    "Container re-joined iot_net while quarantined",
                        })
                        try:
                            if iot_net:
                                iot_net.disconnect(container)
                        except docker.errors.APIError as exc:
                            print(f"[RESCAN] Disconnect error — {exc.explanation}", flush=True)
                        break

                # Re-log presence in quarantine for audit trail (file only, no console)
                mac = q_mac
                ip  = ""
                for net_name, info in attached_nets.items():
                    if QUARANTINE_NETWORK_SUBSTR in net_name:
                        ip = info.get("IPAddress", "")
                        break

                _write_alert({
                    "event":     "QUARANTINE_STILL_ACTIVE",
                    "container": cname,
                    "mac":       mac,
                    "ip":        ip,
                })

            except docker.errors.NotFound:
                with _state_lock:
                    _quarantined_containers.discard(container.name)
            except Exception as exc:
                print(f"[RESCAN] Error for {container.name}: {exc}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# [FIX 4] Anomaly-score cross-correlation watcher thread
# ──────────────────────────────────────────────────────────────────────────────

def _anomaly_watcher_loop(client, quarantine_net):
    """FIX 4 - Poll /output/ for testmodel results CSVs (device*_results.csv)
    and quarantine any container whose anomaly rate exceeds the threshold.
    Container lookup: device IP in device_types.json -> running container."""
    # Check pandas availability once - don't spam per poll
    try:
        import pandas as _pd  # noqa: F401
        pandas_available = True
    except ImportError:
        pandas_available = False
        print("[STARTUP] WARNING: pandas not available — ML anomaly cross-correlation disabled.", flush=True)

    if not pandas_available:
        return   # nothing to do — exit thread cleanly instead of looping forever

    while True:
        time.sleep(ANOMALY_POLL_INTERVAL)
        try:
            pattern = os.path.join(OUTPUT_DIR, "device*_results.csv")
            result_files = glob.glob(pattern)
        except Exception:
            continue

        if not result_files:
            continue   # nothing to process — don't log anything

        for fpath in result_files:
            if _is_file_processed(fpath):
                continue

            try:
                done = _process_results_file(fpath, client, quarantine_net)
                # Only mark as processed if fully acted upon.
                # done=False means device_id not in device_types.json yet — retry next poll.
                if done:
                    _mark_file_processed(fpath)
            except Exception as exc:
                print(f"[ANOMALY] ERROR processing {os.path.basename(fpath)}: {exc}", flush=True)
                _mark_file_processed(fpath)  # don't retry on hard errors


def _cusum_alarm(flags, device_id: int):
    """One-sided CUSUM over per-packet anomaly flags.

    Returns (alarm: bool, packets_to_alarm: int | None, S_final: float).
    Attack is declared the first time the cumulative evidence reaches CUSUM_A;
    negative evidence is reset to zero so benign bursts never accumulate.
    """
    p_ben = DEVICE_P_BEN.get(int(device_id), 0.05)
    p_ben = max(1e-12, min(p_ben, 1.0 - 1e-12))
    p_atk = max(1e-12, min(P_ATK, 1.0 - 1e-12))
    w_anom   = math.log(p_atk / p_ben)
    w_normal = math.log((1.0 - p_atk) / (1.0 - p_ben))
    s = 0.0
    for i, flag in enumerate(flags):
        s = max(0.0, s + (w_anom if flag else w_normal))
        if s >= CUSUM_A:
            return True, i + 1, s
    return False, None, s


def _process_results_file(fpath: str, client, quarantine_net) -> bool:
    """Parse a results CSV and ban the device if its CUSUM evidence crosses A.

    Returns True  - done (act or skip), mark file processed.
    Returns False - device not in device_types.json yet, retry next poll.
    """
    import pandas as pd

    df = pd.read_csv(fpath)

    if "anomaly" not in df.columns:
        print(f"[ANOMALY] {os.path.basename(fpath)} missing 'anomaly' column — skipping.", flush=True)
        return True

    n_total   = len(df)
    flags     = df["anomaly"].astype(int).tolist()
    n_anomaly = int(sum(flags))
    rate      = n_anomaly / n_total * 100 if n_total else 0.0
    mse_mean  = float(df["mse"].mean()) if "mse" in df.columns else -1.0
    threshold = float(df["threshold"].iloc[0]) if "threshold" in df.columns else -1.0

    fname = os.path.basename(fpath)

    # Extract device ID from filename (device3_attack_results.csv → 3)
    try:
        device_id = int(fname.split("_")[0].replace("device", ""))
    except Exception:
        print(f"[ANOMALY] Cannot parse device_id from filename '{fname}' — skipping.", flush=True)
        return True  # unparseable filename, don't retry

    # Sequential decisioning: CUSUM over per-packet anomaly flags.
    # Replaces the legacy whole-capture "anomaly rate > 30%" rule.
    alarm, packets_to_alarm, s_final = _cusum_alarm(flags, device_id)

    if not alarm:
        print(
            f"[ANOMALY] {fname}  →  BENIGN  "
            f"(CUSUM S={s_final:.1f} < A={CUSUM_A:.0f}; "
            f"{n_anomaly}/{n_total} packets anomalous, {rate:.1f}%)",
            flush=True,
        )
        return True

    print(
        f"[ANOMALY] {fname}  →  ATTACK DETECTED (CUSUM)  "
        f"evidence S={s_final:.1f} ≥ A={CUSUM_A:.0f} after {packets_to_alarm} packets",
        flush=True,
    )
    print(f"[ANOMALY] {n_anomaly}/{n_total} packets anomalous ({rate:.1f}%)  "
          f"|  MSE mean: {mse_mean:.6f}  |  Threshold: {threshold:.6f}", flush=True)

    # Find the container IP/MAC/name_prefix from device_types.json
    device_map  = load_device_map()
    target_ip   = None
    target_mac  = None
    name_prefix = None
    for mac, info in device_map.items():
        if not isinstance(info, dict):
            continue
        if info.get("device_id") == device_id:
            target_ip   = info.get("ip")
            target_mac  = mac.upper()
            name_prefix = info.get("name_prefix")   # set for DYNAMIC containers, None for fixed-MAC
            break

    if not target_ip:
        print(
            f"[ANOMALY] device_id={device_id} not in device_types.json yet — "
            f"will retry next poll.",
            flush=True,
        )
        return False

    # Find the running container with that IP on iot_net OR quarantine_network
    # (shadow devices are on quarantine_network by the time results are written)
    target_container = None
    try:
        for container in client.containers.list():
            container.reload()
            nets = container.attrs["NetworkSettings"]["Networks"]
            for net_name, net_info in nets.items():
                if net_info.get("IPAddress") == target_ip:
                    target_container = container
                    break
            if target_container:
                break
    except Exception as exc:
        print(f"[ANOMALY] ERROR looking up container for IP {target_ip}: {exc}", flush=True)
        return True

    if not target_container:
        if target_mac:
            print(
                f"[ANOMALY] Device {device_id} container already stopped — "
                f"banning MAC {target_mac} based on results file.",
                flush=True,
            )
            _ban_mac(target_mac, f"CUSUM_ATTACK_S={s_final:.1f} (container already stopped)", "offline")
            if name_prefix:
                _ban_name_prefix(name_prefix, f"CUSUM_ATTACK_S={s_final:.1f} (container already stopped)", "offline")
            _write_alert({
                "event":           "ANOMALY_FULL_BAN_OFFLINE",
                "device_id":       device_id,
                "ip":              target_ip,
                "mac":             target_mac,
                "name_prefix":     name_prefix,
                "results_file":    fname,
                "anomaly_rate":    round(rate, 2),
                "mse_mean":        round(mse_mean, 6),
                "threshold":       round(threshold, 6),
                "cusum_s":         round(s_final, 2),
                "packets_to_alarm": packets_to_alarm,
            })
            return True
        else:
            print(
                f"[ANOMALY] No container found at {target_ip} and no MAC known — cannot ban device {device_id}.",
                flush=True,
            )
        return True

    iot_net = _find_network(client, IOT_NETWORK_NAME, IOT_NETWORK_SUBSTR)

    # Get MAC directly from the container's current networks
    try:
        for net_name, info in target_container.attrs["NetworkSettings"]["Networks"].items():
            found_ip = info.get("IPAddress", "")
            if found_ip == target_ip:
                target_mac = info.get("MacAddress", "").upper() or target_mac
                break
    except Exception:
        pass

    print(f"[ANOMALY] Blocking device {device_id} — {target_container.name}  |  MAC: {target_mac}  |  IP: {target_ip}", flush=True)

    _write_alert({
        "event":           "ANOMALY_FULL_BAN",
        "container":       target_container.name,
        "device_id":       device_id,
        "ip":              target_ip,
        "mac":             target_mac,
        "name_prefix":     name_prefix,
        "results_file":    fname,
        "anomaly_rate":    round(rate, 2),
        "mse_mean":        round(mse_mean, 6),
        "threshold":       round(threshold, 6),
        "cusum_s":         round(s_final, 2),
        "packets_to_alarm": packets_to_alarm,
    })

    ban_reason = (f"CUSUM ATTACK DETECTED: evidence S={s_final:.1f} ≥ A={CUSUM_A:.0f} "
                  f"after {packets_to_alarm} packets "
                  f"(anomaly rate {rate:.1f}%)")
    _full_ban_container(
        target_container,
        mac=target_mac,
        reason=ban_reason,
        quarantine_net=quarantine_net,
        iot_net=iot_net,
        name_prefix=name_prefix,
    )
    return True


# ──────────────────────────────────────────────────────────────────────────────
# [FIX 3] Fast startup — apply tc shaping before auth check
# ──────────────────────────────────────────────────────────────────────────────

def _handle_start_event(event, client, oui_db, mac_index, prefix_index, name_index, quarantine_net):
    """FIX 3 - Fast auth on container start:
    1. brief grace so Docker wires the network
    2. apply near-zero tc rate limit on the veth immediately
    3. retry reload until iot_net appears (closes the empty-NetworkSettings race)
    4. run the auth check, 5. lift the tc limit if authorized."""
    NETWORK_WIRE_RETRIES    = 10
    NETWORK_WIRE_RETRY_DELAY = 0.3   # seconds between retries = max 3 s total

    container_id = event.get("Actor", {}).get("ID")
    if not container_id:
        return

    time.sleep(STARTUP_GRACE_SECONDS)

    try:
        container = client.containers.get(container_id)
    except docker.errors.NotFound:
        return

    # Apply tc rate limit as early as possible
    veth = _get_container_veth(container_id[:12])
    if veth:
        _tc_rate_limit(veth)

    # Retry until iot_net appears in the container's network list
    for attempt in range(1, NETWORK_WIRE_RETRIES + 1):
        try:
            container.reload()
            nets = container.attrs["NetworkSettings"]["Networks"]
            if any(IOT_NETWORK_SUBSTR in n for n in nets):
                break
        except docker.errors.NotFound:
            return
        time.sleep(NETWORK_WIRE_RETRY_DELAY)

    iot_net = _find_network(client, IOT_NETWORK_NAME, IOT_NETWORK_SUBSTR)

    process_container(container, oui_db, mac_index, prefix_index, name_index, quarantine_net, iot_net)

    # Lift tc limit if container was authorized
    with _state_lock:
        still_quarantined = container.name in _quarantined_containers

    if veth and not still_quarantined:
        _tc_remove_limit(veth)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def start_identification():
    print("=" * 55, flush=True)
    print("  N-BaIoT Identity Engine  |  Two-Tier MAC Enforcement", flush=True)
    print("=" * 55, flush=True)
    print(f"  OUI spoof detection : {'enabled' if OUI_MISMATCH_BLOCK else 'disabled'}", flush=True)
    print(f"  Anomaly detection   : CUSUM (A={CUSUM_A:.0f}, p_atk={P_ATK}, per-device p_ben)", flush=True)
    print(f"  Quarantine subnet   : {QUARANTINE_SUBNET}", flush=True)
    print("=" * 55, flush=True)
    print("", flush=True)

    # Guarantee /output exists before any thread tries to write to it.
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"[STARTUP] Output directory ready: {OUTPUT_DIR}", flush=True)
    except Exception as exc:
        print(f"[STARTUP] WARNING: could not create output directory {OUTPUT_DIR}: {exc}", flush=True)

    try:
        mac_index, prefix_index, name_index = load_whitelist()
    except Exception as exc:
        print(f"[FATAL] Cannot load whitelist: {exc}", flush=True)
        raise

    banned_macs, banned_prefixes = _load_mac_blocklist()
    with _state_lock:
        _banned_macs.update(banned_macs)
        _banned_name_prefixes.update(banned_prefixes)
        _anomaly_checked_files.update(_load_processed_files())

    oui_db = get_ieee_database()
    print(f"[STARTUP] OUI database: {len(oui_db):,} entries.", flush=True)

    client = docker.from_env()

    quarantine_net = ensure_quarantine_network(client)
    iot_net = _find_network(client, IOT_NETWORK_NAME, IOT_NETWORK_SUBSTR)
    if iot_net is None:
        print("[STARTUP] iot_net not found yet — will pick it up on first container event.", flush=True)

    rescan_thread = threading.Thread(
        target=_quarantine_rescan_loop,
        args=(client, oui_db, mac_index, prefix_index, name_index, quarantine_net),
        daemon=True,
        name="QuarantineRescan",
    )
    rescan_thread.start()

    anomaly_thread = threading.Thread(
        target=_anomaly_watcher_loop,
        args=(client, quarantine_net),
        daemon=True,
        name="AnomalyWatcher",
    )
    anomaly_thread.start()

    print("[STARTUP] Scanning existing containers ...", flush=True)
    all_containers = client.containers.list()
    for container in all_containers:
        process_container(container, oui_db, mac_index, prefix_index, name_index, quarantine_net, iot_net)

    print("[WATCH] Listening for container events ...\n", flush=True)
    for event in client.events(decode=True, filters={"type": "container"}):
        try:
            if event.get("Action") != "start":
                continue

            # [FIX 3] Handle in a thread so the event loop isn't blocked
            # during tc shaping + auth check
            t = threading.Thread(
                target=_handle_start_event,
                args=(event, client, oui_db, mac_index, prefix_index,
                      name_index, quarantine_net),
                daemon=True,
            )
            t.start()

        except Exception as exc:
            print(f"[WATCH] Error: {exc}", flush=True)


if __name__ == "__main__":
    start_identification()
