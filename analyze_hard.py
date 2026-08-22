import csv
import numpy as np

rows = list(csv.DictReader(open('/work/output/device1_attack_hard_results.csv')))
ms = np.array([float(r['mse']) for r in rows])
anom = ms > 8.6298
print(f"rows={len(rows)} anom={anom.sum()} ({100*anom.mean():.1f}%)")

i = 0
runs = []
while i < len(anom):
    if anom[i]:
        j = i
        while j + 1 < len(anom) and anom[j + 1]:
            j += 1
        runs.append((i, j, j - i + 1))
        i = j + 1
    else:
        i += 1
print(f"anomalous runs: {len(runs)}")
for (s, e, n) in runs:
    print(f"  [{s},{e}] len={n}")

# sliding-window anomaly fraction over 200-packet windows
w = 200
fracs = []
for start in range(0, len(anom) - w, 50):
    fracs.append((start, anom[start:start + w].mean()))
print("windows with anom frac > 0.15:")
for (s, f) in fracs:
    if f > 0.15:
        print(f"  [{s},{s+w}] frac={f:.2f}")
print("max window frac:", max((f for _, f in fracs), default=0))
