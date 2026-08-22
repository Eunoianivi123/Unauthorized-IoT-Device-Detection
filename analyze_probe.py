import csv
import numpy as np
from collections import defaultdict

feat_rows = list(csv.DictReader(open('/work/output/device1_attack_hard_capture_features.csv')))
res_rows = list(csv.DictReader(open('/work/output/device1_attack_hard_results.csv')))
n = min(len(feat_rows), len(res_rows))
ts = np.array([float(feat_rows[i]['timestamp']) for i in range(n)])
ms = np.array([float(res_rows[i]['mse']) for i in range(n)])
anom = ms > 8.6298

t0 = ts[0]
secs = (ts - t0).astype(int)
pps = np.bincount(secs, minlength=secs.max() + 1)
nsec = len(pps)

# burst periods: seconds with packet rate > 2.5x the median rate
med = np.median(pps[pps > 0])
burst_sec = pps > max(30, 2.5 * med)
print(f"rows={n}  overall anom={anom.mean():.1%}  median pps={med:.0f}  max pps={pps.max()}")

# label each packet's second as in-burst or not
in_burst = np.zeros(n, dtype=bool)
for i in range(n):
    in_burst[i] = burst_sec[secs[i]]

q_burst = anom[in_burst].mean() if in_burst.any() else 0.0
q_gap = anom[~in_burst].mean() if (~in_burst).any() else 0.0
print(f"in-burst packets={in_burst.sum()}  q_burst={q_burst:.2f}")
print(f"gap packets={(~in_burst).sum()}      q_gap={q_gap:.2f}")

# group bursts: find contiguous runs of burst seconds
runs = []
s = 0
while s < nsec:
    if burst_sec[s]:
        e = s
        while e + 1 < nsec and burst_sec[e + 1]:
            e += 1
        runs.append((s, e, e - s + 1))
        s = e + 1
    else:
        s += 1

print("\nper-burst stats (start_s, dur_s, pkts_in_sec, anom_frac):")
for (s, e, dur) in runs:
    sel = np.where((secs >= s) & (secs <= e))[0]
    q = anom[sel].mean() if len(sel) else 0.0
    print(f"  t={s:>4}s dur={dur:>2}s pkts={len(sel):>4} anom_frac={q:.2f}")

# CUSUM peak for reference
p_ben = 0.039
p_atk = 0.3
w_a = np.log(p_atk / p_ben)
w_n = np.log((1 - p_atk) / (1 - p_ben))
s = peak = 0.0
for f in anom:
    s = max(0.0, s + (w_a if f else w_n))
    peak = max(peak, s)
print(f"\nCUSUM peak-S (d1, p_atk=0.3): {peak:.0f}")
