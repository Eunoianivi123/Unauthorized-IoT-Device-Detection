import sys
from collections import Counter
from scapy.all import PcapReader, IP, TCP, UDP

pcap = '/work/output/device1_attack_hard_capture.pcap'
n = 0
secs = Counter()
sizes = []
proto = Counter()
sports = Counter()
dports = Counter()
with PcapReader(pcap) as r:
    for pkt in r:
        if pkt.haslayer(IP):
            n += 1
            t = int(float(pkt.time))
            secs[t] += 1
            sizes.append(len(pkt))
            if pkt.haslayer(TCP):
                proto['tcp'] += 1
            elif pkt.haslayer(UDP):
                proto['udp'] += 1
                if pkt.haslayer(UDP):
                    sports[pkt[UDP].sport] += 1
                    dports[pkt[UDP].dport] += 1
print(f"total IP packets: {n}")
print("proto:", dict(proto))
print("top src ports:", sports.most_common(8))
print("top dst ports:", dports.most_common(8))
print("top seconds by pps:")
for s, c in secs.most_common(10):
    print(f"  t={s} pkts={c}")
print("seconds with >100 pkts:", sum(1 for c in secs.values() if c > 100))
