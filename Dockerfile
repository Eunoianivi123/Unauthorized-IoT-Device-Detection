# ============================================================
# Dockerfile — N-BaIoT IoT Traffic Detection Pipeline
# ============================================================
FROM python:3.11-slim

LABEL description="N-BaIoT IoT Botnet Detection — Traffic Gen + PCAP→CSV + Model Test"

# System tools: tcpdump for capture, net-tools for interface info
RUN apt-get update && apt-get install -y --no-install-recommends \
        tcpdump \
        iproute2 \
        iputils-ping \
        netcat-openbsd \
        curl \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir \
        scapy \
        pandas \
        numpy \
        tensorflow \
        joblib \
        scikit-learn

WORKDIR /app

# Copy pipeline scripts
COPY benign_traffic.py  /app/benign_traffic.py
COPY attack_traffic.py  /app/attack_traffic.py
COPY attack_hard.py     /app/attack_hard.py
COPY device_traffic.py  /app/device_traffic.py
COPY device_traffic_stress.py /app/device_traffic_stress.py
COPY pcap_csv.py        /app/pcap_csv.py
COPY testmodel.py       /app/testmodel.py
COPY entrypoint.sh      /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
