import csv
import numpy as np

feat_rows = list(csv.DictReader(open('/work/output/device1_attack_hard_capture_features.csv')))
res_rows = list(csv.DictReader(open('/work/output/device1_attack_hard_results.csv')))
n = min(len(feat_rows), len(res_rows))
ts = np.array([float(feat_rows[i]['timestamp']) for i in range(n)])
ms = np.array([float(res_rows[i]['mse']) for i in range(n)])
anom = ms > 8.6298

t0 = ts[0]
secs = (ts - t0).astype(int)
pps = np.bincount(secs, minlength=secs.max() + 1)
a_s = np.bincount(secs[anom].astype(int), minlength=secs.max() + 1)
nsec = len(pps)

print(f"rows={n}  overall anom={anom.mean():.1%}  legacy(>30%): {'ATTACK' if anom.mean()>0.3 else 'miss'}")

print("second-by-second profile (t, pps, anom):")
for s in range(nsec):
    if pps[s] > 0:
        mark = "*" if pps[s] > 45 else " "
        print(f"  {mark} t={s:>3}s pps={pps[s]:>3} anom={a_s[s]:>3} frac={a_s[s]/pps[s]:.2f}")

# CUSUM peak
p_ben, p_atk = 0.039, 0.3
w_a = np.log(p_atk / p_ben)
w_n = np.log((1 - p_atk) / (1 - p_ben))
s = peak = 0.0
hit = None
for i, f in enumerate(anom):
    s = max(0.0, s + (w_a if f else w_n))
    if s > peak:
        peak = s
    if s >= 250 and hit is None:
        hit = i + 1
print(f"\nCUSUM peak-S={peak:.0f}  alarm_at={hit}  (A=250, p_atk=0.3, p_ben=0.039)")
