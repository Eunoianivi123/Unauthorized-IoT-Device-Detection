# AGENTS.md

Docker-based IoT security testbed (FYP project): 9 simulated devices generate benign/Mirai traffic, PCAPs are converted to 115 N-BaIoT features, scored by an all-in-one autoencoder, and a MAC-whitelist identity engine quarantines unauthorized/anomalous containers. This work targets publication at least as a conference paper with an authorized publisher (IEEE / Springer / Elsevier / ACM). The ultimate goal is resolving all 10 reviewer comments in "Review Comments FYP_Paper.pdf" (`/home/neelavathy/Downloads/Review Comments FYP_Paper.pdf`, outside this repo) — as of Aug 2026, comments 1, 2, and 10 are addressed; comment 6 (baseline comparison vs per-device N-BaIoT autoencoders) is the current work item. Not a git repo; no test suite, linter, or CI — verification means running scripts inside the Docker image.

## Running things

- TensorFlow-dependent scripts run inside the pipeline image, not on the host:
  `docker run --rm -v "$PWD:/work" -w /work nbaiot-pipeline:latest python3 <script.py>`
- Compose services only start via profiles (`compose.yml`): `build`, `devices`, `gateway`, `identity`, `shadow`.
  - Build once: `docker compose --profile build build`
  - Infra stack: `docker compose --profile gateway --profile identity up -d`
  - One benign device run: `docker compose --profile devices up device-1`
  - Attack run: prefix with `-e MODE=attack` (modes: `normal | attack | attack_hard | stress`)
  - Rogue device (auto-quarantine demo): `docker compose --profile shadow run --rm shadow-device-1`
- Teardown: `docker compose --profile identity --profile devices --profile gateway down --remove-orphans`
- Identity engine logs: `docker logs -f nbaiot-identifier`

## Volume-mount quirks

- Pipeline/device containers see `./output` at `/shared` and `./model` at `/model` (read-only); the identifier sees the same output dir at `/output`. Scripts hardcode these paths.
- The identifier mounts `/var/run/docker.sock` — it reconfigures container networks itself (quarantine = move off iot_net onto quarantine_network).

## Data contracts — do not break silently

- Output filenames follow `device{N}_{mode}[_rN]_capture{,_features,_results}.{pcap,csv}` in `./output`. `identification_engine.py` polls `*_results.csv` and declares an attack via one-sided CUSUM over per-packet `anomaly` flags (S ≥ `CUSUM_A=250`) → quarantine/ban. The legacy "anomaly rate > 30%" rule is superseded — do not reintroduce it.
- CUSUM parameters (`P_ATK`, `CUSUM_A`, per-device `DEVICE_P_BEN` dict) live near the top of `identification_engine.py`; they are duplicated/calibrated via the `CUSUM` class in `sprt_accumulator.py`. If you recalibrate, update both.
- The 115-feature N-BaIoT column list lives in `pcap_csv.py` and is duplicated as `FEATURE_NAMES` in consumers (e.g. `compare_models.py`). Keep them in sync; import/reuse rather than retyping subsets.
- Per-device MSE thresholds are hardcoded in `entrypoint.sh`. If you recalibrate thresholds in an analysis script, update `entrypoint.sh` too.
- Device IDs 1–9 map to fixed device names (Danmini doorbell … SimpleHome cameras) via a `case` in `entrypoint.sh`; static IP/MAC pairs live in `compose.yml` and must stay consistent with `whitelisted.json`.

## Evaluation methodology (paper-facing)

- Threshold rule everywhere: `median(benign MSE) + 2.5 * MAD`, per device — NOT mean+std (rationale documented in `THRESHOLD_MODEL_COMPARISON.md`).
- Calibrate thresholds on benign run-1 only (`device{N}_normal_r1_capture_features.csv`); measure FPR on held-out benign run-2; AUC pools r1+r2+attack. Never calibrate and report FPR on the same capture.
- `final_evaluation.py` is the reference implementation of this protocol; `compare_models.py` compares the unified ("allinone") model vs per-device baseline autoencoders on identical captures.
- Reviewer-comment artifacts: `baseline_comparison.py` (comment 6 — unified vs per-device N-BaIoT baseline across normal/attack/attack_hard/stress) and `detector_comparison.py` (CUSUM vs EWMA vs Shewhart justification) write their result tables to `./output/*_results.csv`. Both run in the pipeline image; mount `-v /home/neelavathy/model:/work/models:ro -e MODEL_DIR=models` because `./model` is a host symlink invisible inside containers.
- Everything here feeds a peer-reviewed submission: reported metrics, `./output` artifacts, and methodology docs (`THRESHOLD_MODEL_COMPARISON.md`) are citable evidence — never overwrite or regenerate them casually, and keep any new analysis consistent with the r1/r2 protocol.
