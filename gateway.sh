#!/bin/sh
# gateway.sh - traffic sink for the IoT simulation
# Serves HTTP on :80 and forwards everything else to the UDP sink.
python3 -m http.server 80 &
python3 /app/gateway_sink.py
