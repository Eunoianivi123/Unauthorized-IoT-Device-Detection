#!/usr/bin/env python3
"""Realistic benign IoT traffic generator (video, heartbeat, DNS, NTP, etc.)."""
import random
import socket
import struct
import threading
import time

CLOUD_SERVER = "192.168.100.3"
HTTP_PORT = 80
RTSP_PORT = 554
UDP_PORT = 9999


def send_udp(dst_ip, dst_port, payload, count=1):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1)
        for _ in range(count):
            s.sendto(payload, (dst_ip, dst_port))
            time.sleep(0.001)
        s.close()
    except Exception:
        pass


def send_tcp(dst_ip, dst_port, payload=b"GET / HTTP/1.0\r\n\r\n"):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((dst_ip, dst_port))
        s.send(payload)
        s.recv(1024)
        s.close()
    except Exception:
        pass


# Continuous low-latency video stream (RTSP-like UDP)
def video_stream_thread():
    print("[Thread-VideoStream] Started")
    seq = 0
    while True:
        chunk_size = random.randint(1200, 1400)
        payload = struct.pack(">HH", seq % 65535, chunk_size) + \
                  bytes([random.randint(0, 255) for _ in range(chunk_size - 4)])
        send_udp(CLOUD_SERVER, RTSP_PORT, payload)
        seq += 1
        time.sleep(0.033 + random.uniform(-0.005, 0.005))


# Periodic small heartbeat packets
def heartbeat_thread():
    print("[Thread-Heartbeat] Started")
    device_id = random.randint(1000, 9999)
    while True:
        payload = f"HB|ID:{device_id}|TS:{int(time.time())}|OK".encode()
        send_udp(CLOUD_SERVER, UDP_PORT, payload)
        time.sleep(5)


# Periodic HTTP sensor reports
def http_report_thread():
    print("[Thread-HTTPReport] Started")
    while True:
        try:
            payload = (
                f"POST /api/sensor HTTP/1.0\r\n"
                f"Host: {CLOUD_SERVER}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: 64\r\n\r\n"
                f'{{"temp":22.5,"motion":false,"ts":{int(time.time())}}}'
            ).encode()
            send_tcp(CLOUD_SERVER, HTTP_PORT, payload)
        except Exception:
            pass
        time.sleep(10 + random.uniform(-1, 1))


# NTP sync
def ntp_sync_thread():
    print("[Thread-NTPSync] Started")
    ntp_packet = b'\x1b' + 47 * b'\0'
    while True:
        send_udp(CLOUD_SERVER, 123, ntp_packet)
        time.sleep(30 + random.uniform(-2, 2))


# Bursty motion-detection alert uploads
def motion_event_thread():
    print("[Thread-MotionEvent] Started")
    while True:
        time.sleep(20 + random.uniform(-5, 5))
        burst_count = random.randint(5, 15)
        print(f"  [Motion] Triggered -- sending {burst_count} alert packets")
        for _ in range(burst_count):
            img_size = random.randint(8000, 15000)
            payload = (
                f"POST /api/alert HTTP/1.0\r\n"
                f"Host: {CLOUD_SERVER}\r\n"
                f"Content-Type: image/jpeg\r\n"
                f"Content-Length: {img_size}\r\n\r\n"
            ).encode() + bytes(random.randint(100, 500))
            send_tcp(CLOUD_SERVER, HTTP_PORT, payload)
            time.sleep(0.05)


# Periodic DNS queries
def dns_query_thread():
    print("[Thread-DNS] Started")

    def build_dns_query(domain):
        transaction_id = random.randint(0, 65535)
        header = struct.pack(">HHHHHH", transaction_id, 0x0100, 1, 0, 0, 0)
        qname = b""
        for part in domain.split("."):
            qname += bytes([len(part)]) + part.encode()
        qname += b"\x00"
        return header + qname + struct.pack(">HH", 1, 1)

    domains = ["cloud.vendor.io", "ntp.pool.org", "updates.firmware.io"]
    while True:
        domain = random.choice(domains)
        pkt = build_dns_query(domain)
        send_udp(CLOUD_SERVER, 53, pkt)
        time.sleep(random.uniform(8, 15))


# Rare firmware update checks
def firmware_check_thread():
    print("[Thread-Firmware] Started")
    while True:
        time.sleep(random.uniform(45, 90))
        payload = (
            f"GET /firmware/latest HTTP/1.0\r\n"
            f"Host: {CLOUD_SERVER}\r\n"
            f"User-Agent: IoTCam/2.1\r\n\r\n"
        ).encode()
        send_tcp(CLOUD_SERVER, HTTP_PORT, payload)
        print("  [Firmware] Checked for update")


# ── Main: launch all traffic threads ──────────────────────
def main():
    print("\n[IoT Device] Launching N-BaIoT style traffic threads...\n")
    threads = [
        threading.Thread(target=video_stream_thread,   daemon=True, name="VideoStream"),
        threading.Thread(target=heartbeat_thread,      daemon=True, name="Heartbeat"),
        threading.Thread(target=http_report_thread,    daemon=True, name="HTTPReport"),
        threading.Thread(target=ntp_sync_thread,       daemon=True, name="NTPSync"),
        threading.Thread(target=motion_event_thread,   daemon=True, name="MotionEvent"),
        threading.Thread(target=dns_query_thread,      daemon=True, name="DNS"),
        threading.Thread(target=firmware_check_thread, daemon=True, name="Firmware"),
    ]
    for t in threads:
        t.start()
        time.sleep(0.2)

    print("\n[IoT Device] All traffic threads running.\n")
    start_time = time.time()
    while True:
        elapsed = int(time.time() - start_time)
        active = sum(1 for t in threads if t.is_alive())
        print(f"[Status] Uptime: {elapsed}s | Active threads: {active}/7")
        time.sleep(15)


if __name__ == "__main__":
    main()
