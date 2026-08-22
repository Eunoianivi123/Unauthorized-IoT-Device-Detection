#!/usr/bin/env python3
"""Fit per-device MSE density models for SPRT sequential decisioning.

Reads benign and attack MSE samples from output/device{N}_results.csv and
fits a probability density model for each class per device, selecting the
best family (gamma / log-normal) by log-likelihood. Writes the result to
output/sprt_densities.json, consumed by the identification engine and by
sprt_accumulator.py.

Usage:
    python fit_densities.py
    python fit_densities.py --devices 1 2 3 --out-dir output
"""
import argparse
import json
import os

from sprt_accumulator import fit_density, read_mse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--devices", type=int, nargs="+", default=list(range(1, 10)))
    ap.add_argument("--benign-suffix", default="normal_r1", help="benign capture suffix (normal_r1 = threshold-calibration run)")
    ap.add_argument("--attack-suffix", default="attack")
    args = ap.parse_args()

    devices = {}
    print(f"  {'device':<7}{'ben_n':>7}{'ben_family':<12}{'attack_n':>9}{'attack_family':<12}")
    print("  " + "-" * 48)
    for d in args.devices:
        ben_path = os.path.join(args.out_dir, f"device{d}_{args.benign_suffix}_results.csv")
        atk_path = os.path.join(args.out_dir, f"device{d}_{args.attack_suffix}_results.csv")
        if not (os.path.exists(ben_path) and os.path.exists(atk_path)):
            print(f"  device{d:<5}  missing files -- skipped")
            continue
        ben_mse = read_mse(ben_path)
        atk_mse = read_mse(atk_path)
        if len(ben_mse) == 0 or len(atk_mse) == 0:
            print(f"  device{d:<5}  empty MSE column -- skipped")
            continue
        d_ben = fit_density(ben_mse)
        d_atk = fit_density(atk_mse)
        devices[str(d)] = {
            "benign": d_ben.to_dict(),
            "attack": d_atk.to_dict(),
        }
        print(f"  device{d:<5}{len(ben_mse):>7}{d_ben.family:<12}{len(atk_mse):>9}{d_atk.family:<12}")

    payload = {
        "version": 1,
        "description": "Per-device P(mse|benign) and P(mse|attacked) density models for SPRT decisioning",
        "note": "fitted from MSE of benign (normal_r1) and attack captures using the all-in-one autoencoder",
        "benign_suffix": args.benign_suffix,
        "attack_suffix": args.attack_suffix,
        "devices": devices,
    }
    out_path = os.path.join(args.out_dir, "sprt_densities.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[OK] Wrote {out_path}  ({len(devices)} device models)")


if __name__ == "__main__":
    main()
