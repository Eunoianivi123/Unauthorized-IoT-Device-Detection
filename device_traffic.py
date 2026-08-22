#!/usr/bin/env python3
"""Per-device N-BaIoT realistic benign traffic simulator.

Usage:
  DEVICE_ID=1 python3 device_traffic.py
"""
import os
import random
import socket
import struct
import sys
import threading
import time

TARGET_IP = os.environ.get("TARGET_IP", "192.168.100.3")
DEVICE_ID = int(os.environ.get("DEVICE_ID", "1"))

# Device types matching the N-BaIoT dataset model numbering
DEVICES = {
    1: "Danmini_Doorbell",
    2: "Ecobee_Thermostat",
    3: "Ennio_Doorbell",
    4: "Philips_B120N10_Baby_Monitor",
    5: "Provision_PT_737E_Security_Camera",
    6: "Provision_PT_838_Security_Camera",
    7: "Samsung_SNH_1011_N_Webcam",
    8: "SimpleHome_XCS7_1002_WHT_Security_Camera",
    9: "SimpleHome_XCS7_1003_WHT_Security_Camera",
}

if DEVICE_ID not in DEVICES:
    print(f"[ERROR] DEVICE_ID must be 1-9, got {DEVICE_ID}")
    sys.exit(1)

DEVICE_NAME = DEVICES[DEVICE_ID]
print(f"\n[Device] Simulating: {DEVICE_NAME} (ID={DEVICE_ID})")
print(f"[Device] Target    : {TARGET_IP}\n")

# ── Low-level socket helpers ──────────────────────────────
def send_udp(ip, port, payload):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1)
        s.sendto(payload, (ip, port))
        s.close()
    except Exception:
        pass


def send_tcp(ip, port, payload):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((ip, port))
        s.send(payload)
        try:
            s.recv(256)
        except Exception:
            pass
        s.close()
    except Exception:
        pass


def build_dns_query(domain):
    tid = random.randint(0, 65535)
    header = struct.pack(">HHHHHH", tid, 0x0100, 1, 0, 0, 0)
    qname = b""
    for part in domain.split("."):
        qname += bytes([len(part)]) + part.encode()
    qname += b"\x00"
    return header + qname + struct.pack(">HH", 1, 1)


def ntp_packet():
    return b'\x1b' + 47 * b'\0'


# ── Shared thread builders ────────────────────────────────
def t_ntp(interval=30):
    """NTP sync - all devices do this."""
    while True:
        send_udp(TARGET_IP, 123, ntp_packet())
        time.sleep(interval + random.uniform(-2, 2))


def t_dns(domains, interval_range=(8, 20)):
    while True:
        send_udp(TARGET_IP, 53, build_dns_query(random.choice(domains)))
        time.sleep(random.uniform(*interval_range))


def t_heartbeat(port, payload_fn, interval=5):
    while True:
        send_udp(TARGET_IP, port, payload_fn())
        time.sleep(interval + random.uniform(-0.5, 0.5))


def t_http_poll(path, user_agent, interval_range=(10, 20)):
    while True:
        payload = (
            f"GET {path} HTTP/1.0\r\n"
            f"Host: {TARGET_IP}\r\n"
            f"User-Agent: {user_agent}\r\n\r\n"
        ).encode()
        send_tcp(TARGET_IP, 80, payload)
        time.sleep(random.uniform(*interval_range))


def t_rtsp_stream(fps=15, chunk_lo=800, chunk_hi=1400):
    """UDP-based video stream (RTSP-like)."""
    seq = 0
    while True:
        size = random.randint(chunk_lo, chunk_hi)
        payload = struct.pack(">HH", seq % 65535, size) + \
                  bytes([random.randint(0, 255) for _ in range(size - 4)])
        send_udp(TARGET_IP, 554, payload)
        seq += 1
        time.sleep(1.0 / fps + random.uniform(-0.005, 0.005))


def t_motion_alert(img_lo=5000, img_hi=15000, interval_range=(15, 40)):
    while True:
        time.sleep(random.uniform(*interval_range))
        burst = random.randint(3, 10)
        for _ in range(burst):
            size = random.randint(img_lo, img_hi)
            payload = (
                f"POST /alert HTTP/1.0\r\nHost: {TARGET_IP}\r\n"
                f"Content-Type: image/jpeg\r\nContent-Length: {size}\r\n\r\n"
            ).encode() + bytes(random.randint(64, 256))
            send_tcp(TARGET_IP, 80, payload)
            time.sleep(0.05)


def t_cloud_report(path, interval_range=(8, 15)):
    while True:
        body = f'{{"ts":{int(time.time())},"ok":true}}'
        payload = (
            f"POST {path} HTTP/1.0\r\nHost: {TARGET_IP}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n{body}"
        ).encode()
        send_tcp(TARGET_IP, 80, payload)
        time.sleep(random.uniform(*interval_range))


# ── Per-device thread profiles ────────────────────────────
def threads_danmini_doorbell():
    """WiFi video doorbell - high-rate stream, motion alerts, cloud check-ins."""
    return [
        threading.Thread(target=t_rtsp_stream,    kwargs={"fps": 20, "chunk_lo": 1000, "chunk_hi": 1400}, name="VideoStream"),
        threading.Thread(target=t_motion_alert,   kwargs={"img_lo": 8000, "img_hi": 18000, "interval_range": (10, 25)}, name="MotionAlert"),
        threading.Thread(target=t_cloud_report,   kwargs={"path": "/api/doorbell/status", "interval_range": (5, 10)}, name="CloudReport"),
        threading.Thread(target=t_ntp,            kwargs={"interval": 30}, name="NTP"),
        threading.Thread(target=t_dns,            kwargs={"domains": ["cloud.danmini.com", "ntp.danmini.com", "ota.danmini.com"]}, name="DNS"),
        threading.Thread(target=t_heartbeat,      kwargs={"port": 9999, "payload_fn": lambda: b"RING|" + str(int(time.time())).encode(), "interval": 3}, name="Heartbeat"),
    ]


def threads_ecobee_thermostat():
    """Smart thermostat - low bandwidth, HTTP polling, no video/burst traffic."""

    def temp_report():
        temp = round(random.uniform(18.0, 26.0), 1)
        body = f'{{"temp":{temp},"humidity":{random.randint(30,60)},"ts":{int(time.time())}}}'
        return (
            f"POST /api/thermostat/report HTTP/1.0\r\nHost: {TARGET_IP}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n{body}"
        ).encode()

    return [
        threading.Thread(target=t_http_poll,    kwargs={"path": "/api/schedule", "user_agent": "Ecobee/4.8", "interval_range": (30, 60)}, name="SchedulePoll"),
        threading.Thread(target=t_http_poll,    kwargs={"path": "/api/weather",  "user_agent": "Ecobee/4.8", "interval_range": (60, 120)}, name="WeatherPoll"),
        threading.Thread(target=lambda: [send_tcp(TARGET_IP, 80, temp_report()) or time.sleep(random.uniform(25, 35)) for _ in iter(int, 1)], name="TempReport"),
        threading.Thread(target=t_ntp,          kwargs={"interval": 60}, name="NTP"),
        threading.Thread(target=t_dns,          kwargs={"domains": ["my.ecobee.com", "api.ecobee.com"], "interval_range": (30, 60)}, name="DNS"),
        threading.Thread(target=t_heartbeat,    kwargs={"port": 8883, "payload_fn": lambda: b'\x10\x1a\x00\x04MQTT\x04\x02\x00\x3c\x00\x08ecobee01', "interval": 20}, name="MQTTKeepAlive"),
    ]


def threads_ennio_doorbell():
    """Cheaper doorbell - moderate stream rate, door-ring bursts, basic cloud sync."""
    return [
        threading.Thread(target=t_rtsp_stream,  kwargs={"fps": 10, "chunk_lo": 600, "chunk_hi": 900}, name="VideoStream"),
        threading.Thread(target=t_motion_alert, kwargs={"img_lo": 4000, "img_hi": 10000, "interval_range": (15, 35)}, name="MotionAlert"),
        threading.Thread(target=t_cloud_report, kwargs={"path": "/api/status", "interval_range": (10, 20)}, name="CloudReport"),
        threading.Thread(target=t_ntp,          kwargs={"interval": 60}, name="NTP"),
        threading.Thread(target=t_dns,          kwargs={"domains": ["cloud.ennio.io", "ntp.pool.org"]}, name="DNS"),
        threading.Thread(target=t_heartbeat,    kwargs={"port": 9999, "payload_fn": lambda: b"HB|ENNIO|" + str(int(time.time())).encode(), "interval": 5}, name="Heartbeat"),
    ]


def threads_philips_baby_monitor():
    """Baby monitor - constant audio+video stream, high packet rate."""

    def audio_stream():
        while True:
            payload = bytes([random.randint(0, 255) for _ in range(random.randint(160, 320))])
            send_udp(TARGET_IP, 5004, payload)   # RTP audio port
            time.sleep(0.020 + random.uniform(-0.002, 0.002))  # 50 Hz

    return [
        threading.Thread(target=t_rtsp_stream,  kwargs={"fps": 25, "chunk_lo": 1100, "chunk_hi": 1400}, name="VideoStream"),
        threading.Thread(target=audio_stream,   name="AudioStream"),
        threading.Thread(target=t_cloud_report, kwargs={"path": "/api/monitor/ping", "interval_range": (15, 25)}, name="CloudPing"),
        threading.Thread(target=t_ntp,          kwargs={"interval": 30}, name="NTP"),
        threading.Thread(target=t_dns,          kwargs={"domains": ["cloud.philips.com", "myuremote.com"], "interval_range": (15, 30)}, name="DNS"),
        threading.Thread(target=t_heartbeat,    kwargs={"port": 9999, "payload_fn": lambda: b"BABY|ALIVE|" + str(int(time.time())).encode(), "interval": 4}, name="Heartbeat"),
    ]


def threads_provision_pt737e():
    """Outdoor pan-tilt camera - continuous HD stream, motion + NVR snapshots."""

    def snapshot_upload():
        while True:
            time.sleep(random.uniform(10, 20))
            size = random.randint(20000, 50000)
            payload = (
                f"POST /nvr/snapshot HTTP/1.0\r\nHost: {TARGET_IP}\r\n"
                f"Content-Type: image/jpeg\r\nContent-Length: {size}\r\n\r\n"
            ).encode() + bytes(random.randint(128, 512))
            send_tcp(TARGET_IP, 80, payload)

    return [
        threading.Thread(target=t_rtsp_stream,   kwargs={"fps": 25, "chunk_lo": 1200, "chunk_hi": 1400}, name="VideoStream"),
        threading.Thread(target=snapshot_upload, name="SnapshotUpload"),
        threading.Thread(target=t_motion_alert,  kwargs={"img_lo": 15000, "img_hi": 40000, "interval_range": (8, 20)}, name="MotionAlert"),
        threading.Thread(target=t_ntp,           kwargs={"interval": 30}, name="NTP"),
        threading.Thread(target=t_dns,           kwargs={"domains": ["nvr.local", "provision-isr.com", "time.nist.gov"]}, name="DNS"),
        threading.Thread(target=t_heartbeat,     kwargs={"port": 37777, "payload_fn": lambda: b'\xff\x01\x00\x04\x00\x00\x00\x00', "interval": 20}, name="DVRHeartbeat"),
    ]


def threads_provision_pt838():
    """Fixed indoor camera - like PT-737E, slightly lower rate, more snapshots."""

    def snapshot_upload():
        while True:
            time.sleep(random.uniform(8, 15))
            size = random.randint(15000, 35000)
            payload = (
                f"POST /nvr/snapshot HTTP/1.0\r\nHost: {TARGET_IP}\r\n"
                f"Content-Type: image/jpeg\r\nContent-Length: {size}\r\n\r\n"
            ).encode() + bytes(random.randint(128, 512))
            send_tcp(TARGET_IP, 80, payload)

    return [
        threading.Thread(target=t_rtsp_stream,   kwargs={"fps": 20, "chunk_lo": 1000, "chunk_hi": 1300}, name="VideoStream"),
        threading.Thread(target=snapshot_upload, name="SnapshotUpload"),
        threading.Thread(target=t_motion_alert,  kwargs={"img_lo": 10000, "img_hi": 30000, "interval_range": (10, 25)}, name="MotionAlert"),
        threading.Thread(target=t_ntp,           kwargs={"interval": 30}, name="NTP"),
        threading.Thread(target=t_dns,           kwargs={"domains": ["nvr.local", "provision-isr.com"]}, name="DNS"),
        threading.Thread(target=t_heartbeat,     kwargs={"port": 37777, "payload_fn": lambda: b'\xff\x01\x00\x04\x00\x00\x00\x00', "interval": 20}, name="DVRHeartbeat"),
    ]


def threads_samsung_webcam():
    """Cloud webcam - cloud-relayed stream, frequent HTTP polling, XMPP keep-alive."""

    def xmpp_keepalive():
        while True:
            send_tcp(TARGET_IP, 5222, b' ')   # XMPP whitespace keep-alive
            time.sleep(random.uniform(25, 35))

    return [
        threading.Thread(target=t_rtsp_stream,   kwargs={"fps": 15, "chunk_lo": 700, "chunk_hi": 1100}, name="VideoStream"),
        threading.Thread(target=t_http_poll,     kwargs={"path": "/api/stream/token", "user_agent": "Samsung-SmartCam/2.0", "interval_range": (5, 10)}, name="TokenRefresh"),
        threading.Thread(target=t_cloud_report,  kwargs={"path": "/api/cam/status", "interval_range": (10, 20)}, name="CloudStatus"),
        threading.Thread(target=xmpp_keepalive,  name="XMPPKeepAlive"),
        threading.Thread(target=t_ntp,           kwargs={"interval": 60}, name="NTP"),
        threading.Thread(target=t_dns,           kwargs={"domains": ["smartcam.samsung.com", "xmpp.samsungsmartcam.com"], "interval_range": (10, 20)}, name="DNS"),
    ]


def threads_simplehome_1002():
    """Budget camera - low-res stream, infrequent cloud sync, basic heartbeat."""
    return [
        threading.Thread(target=t_rtsp_stream,  kwargs={"fps": 10, "chunk_lo": 400, "chunk_hi": 700}, name="VideoStream"),
        threading.Thread(target=t_cloud_report, kwargs={"path": "/api/cam/ping", "interval_range": (15, 30)}, name="CloudPing"),
        threading.Thread(target=t_motion_alert, kwargs={"img_lo": 3000, "img_hi": 8000, "interval_range": (20, 50)}, name="MotionAlert"),
        threading.Thread(target=t_ntp,          kwargs={"interval": 60}, name="NTP"),
        threading.Thread(target=t_dns,          kwargs={"domains": ["cloud.simplehome.com", "ntp.pool.org"], "interval_range": (20, 40)}, name="DNS"),
        threading.Thread(target=t_heartbeat,    kwargs={"port": 9999, "payload_fn": lambda: b"HB|1002|" + str(int(time.time())).encode(), "interval": 8}, name="Heartbeat"),
    ]


def threads_simplehome_1003():
    """Upgraded 1002 - higher stream rate, more frequent cloud reports."""
    return [
        threading.Thread(target=t_rtsp_stream,  kwargs={"fps": 15, "chunk_lo": 600, "chunk_hi": 1000}, name="VideoStream"),
        threading.Thread(target=t_cloud_report, kwargs={"path": "/api/cam/ping", "interval_range": (10, 20)}, name="CloudPing"),
        threading.Thread(target=t_motion_alert, kwargs={"img_lo": 5000, "img_hi": 12000, "interval_range": (15, 40)}, name="MotionAlert"),
        threading.Thread(target=t_ntp,          kwargs={"interval": 60}, name="NTP"),
        threading.Thread(target=t_dns,          kwargs={"domains": ["cloud.simplehome.com", "ntp.pool.org"], "interval_range": (15, 30)}, name="DNS"),
        threading.Thread(target=t_heartbeat,    kwargs={"port": 9999, "payload_fn": lambda: b"HB|1003|" + str(int(time.time())).encode(), "interval": 6}, name="Heartbeat"),
    ]


# ── Dispatch table: DEVICE_ID -> thread profile ───────────
DEVICE_THREADS = {
    1: threads_danmini_doorbell,
    2: threads_ecobee_thermostat,
    3: threads_ennio_doorbell,
    4: threads_philips_baby_monitor,
    5: threads_provision_pt737e,
    6: threads_provision_pt838,
    7: threads_samsung_webcam,
    8: threads_simplehome_1002,
    9: threads_simplehome_1003,
}


# ── Main: start the device's traffic threads ──────────────
def main():
    threads = DEVICE_THREADS[DEVICE_ID]()
    for t in threads:
        t.daemon = True
        t.start()
        time.sleep(0.1)

    print(f"[{DEVICE_NAME}] All {len(threads)} traffic threads running.\n")
    start = time.time()
    while True:
        elapsed = int(time.time() - start)
        alive = sum(1 for t in threads if t.is_alive())
        print(f"[Status] {DEVICE_NAME} | Uptime: {elapsed}s | Threads: {alive}/{len(threads)}")
        time.sleep(10)


if __name__ == "__main__":
    main()
