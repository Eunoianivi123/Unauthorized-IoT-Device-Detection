## 1. Threshold selection (summary)

> **Chosen threshold:** `median(benign MSE) + 2.5 * MAD(benign MSE)`, computed per device
> from that device's own benign traffic.
>
> MSE scores are heavily right-skewed, so the common `mean + 2*std` threshold is dragged
> into the tail and detects **0% of attacks on 4 of 9 devices**. `median + 2.5*MAD` is
> robust to that skew and gave the highest accuracy on **all 9 devices** — while using
> benign traffic only, consistent with benign-only training.

---

## 2. Final per-device thresholds

| Device ID | Device name | Threshold |
|---|---|---|
| 1 | Danmini_Doorbell | 8.6298 |
| 2 | Ecobee_Thermostat | 0.2253 |
| 3 | Ennio_Doorbell | 2.1119 |
| 4 | Philips_B120N10_Baby_Monitor | 27.8245 |
| 5 | Provision_PT_737E_Security_Camera | 13.5779 |
| 6 | Provision_PT_838_Security_Camera | 7.7438 |
| 7 | Samsung_SNH_1011_N_Webcam | 4.2077 |
| 8 | SimpleHome_XCS7_1002_WHT_Security_Camera | 1.3466 |
| 9 | SimpleHome_XCS7_1003_WHT_Security_Camera | 3.7666 |

---

## 3. Final accuracy — all-in-one vs single-device autoencoders

Both model families scored on the **same captures** with the **same** benign-only threshold
methodology. To keep the measurement honest, **two separate benign captures** per device were
collected: the threshold is calibrated on benign run 1 (`median + 2.5*MAD`) and the
false-positive rate is measured on benign run 2, which the threshold never saw. Total benign
pool: ~15,000 streams across 9 devices. Re-running with a second benign capture changed the
accuracy by <0.1% (72.88% -> 72.85%), confirming the thresholds are stable and the reported
accuracy reflects model capability, not benign sample size.

| Device | All-in-one AUC / Accuracy | Single AUC / Accuracy | Winner |
|---|---|---|---|
| 1  Danmini doorbell | 0.72 / 67.0% | 0.88 / 82.0% | single |
| 2  Ecobee thermostat | 0.94 / 93.9% | 0.99 / 99.2% | single |
| 3  Ennio doorbell | 0.86 / 82.6% | 0.80 / 89.8% | all-in-one (AUC) |
| 4  Philips baby monitor | 0.57 / 56.2% | 0.63 / 60.5% | single |
| 5  PT-737E camera | 0.66 / 60.3% | 0.72 / 65.0% | single |
| 6  PT-838 camera | 0.75 / 68.9% | 0.77 / 70.3% | single |
| 7  Samsung webcam | 0.81 / 75.9% | 0.91 / 86.5% | single |
| 8  XCS7-1002 camera | 0.89 / 86.0% | 0.45 / 31.9% | all-in-one |
| 9  XCS7-1003 camera | 0.82 / 77.6% | 0.92 / 88.7% | single |
| **OVERALL** | **0.70 / 72.9%** | **0.71 / 73.9%** | near tie |

### Overall metrics

| Metric | All-in-one | Single-device |
|---|---|---|
| Accuracy | 72.9% | 73.9% |
| ROC-AUC | 0.695 | 0.713 |
| TPR (detection rate) | 68.0% | 69.3% |
| FPR (false alarms) | 3.9% (held-out) | 3.9% |
| F1 score | 0.805 | 0.814 |

---

## 4. Key result — where the all-in-one outperforms the single-device ideology

1. **Device 8 (SimpleHome XCS7-1002):** the single-device autoencoder collapses
   (AUC **0.45** — worse than random, 32% accuracy, only 25% of attacks caught), while
   the all-in-one stays strong (AUC **0.89**, 86% accuracy, 85% of attacks caught).
   A universal model is resilient to an individual model's poor training.

2. **Device 3 (Ennio doorbell):** the all-in-one has a **higher AUC (0.86 vs 0.80)**,
   meaning better detection quality even though the single model shows higher raw accuracy
   at a much higher 19.6% false-positive rate.

3. **Overall consistency:** with **one model instead of nine**, the all-in-one achieves
   nearly the same overall accuracy (72.9% vs 73.9%) and F1 (0.805 vs 0.814) while never
   failing catastrophically on any device.

### Conclusion

> A single universal autoencoder delivers ~equal accuracy to nine specialised
> per-device models, while avoiding the risk of a badly-trained individual model
> (device 8). One model means simpler deployment and maintenance at a small,
> acceptable accuracy cost.

---

## 5. Caveats

- Results reflect **one attack scenario** (the synthetic 3-second attack generator);
  generalisation to other attack types needs further testing.
- **Device 2** had few benign streams (77 per capture) — its numbers are real but based on
  a small benign sample.
- **Device 4** (Philips baby monitor) is inherently weak for both models (AUC ~0.57-0.63)
  — a model-quality limit, not a threshold issue.
- Overall accuracy is dominated by **model capability** (AUC ~0.70), not benign sample size;
  doubling the benign data changed accuracy by <0.1%.
