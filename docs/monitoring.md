# Monitoring, drift detection and retraining

Drift features, detectors, both scorecards, the Prometheus/Grafana stack, alert rules, the demos and the drift-triggered retraining loop. Back to the [README](../README.md).

## Monitoring and drift (Module 5)

Code lives in the top-level `monitoring/` package (dependency group `monitor`, pinned).
The runtime image adds only `prometheus-client`.

**Features per image:** the 2048-d penultimate embedding of the shipped
`resnet50_best.pt` through the shared eval transform (CUDA); pixel statistics with
numpy (brightness = mean luminance, contrast = luminance std, sharpness = variance of
a 4-neighbour Laplacian); and the calibrated max probability from `PetBreedPredictor`.

**Detectors per batch of 200 images:**

| Detector | Statistic | Alarm rule | Threshold |
|---|---|---|---|
| Embedding drift | domain classifier (standardize -> PCA-50 -> logistic regression), 5-fold CV ROC-AUC, reference vs batch | AUC > threshold | 0.5399 (99th percentile over 50 clean val batches) |
| Pixel statistics | two-sample KS per statistic vs reference; score = -log10(min p) | min p < 0.01/3 | 2.4771 (fixed Bonferroni rule; clean 99th percentile was 1.99) |
| Confidence drift (tracked separately) | KS statistic of calibrated max probability vs val reference | KS > threshold | 0.0977 (99th percentile over 50 clean val batches) |

A fourth, **post-hoc** detector, `embedding_valref`, was added on 2026-10-13 after the
scorecard below showed the embedding detector's clean-test false alarms. It is the
same domain classifier, but its reference is the seeded val half used by the
confidence detector (n=368), and its threshold, **0.5517**, is the 99th percentile
over 50 batches of 200 drawn from the other val half (seed 42). It is scored in the
scorecard only; the live drift check still runs the three detectors above.

References: clean train images for embedding and pixel statistics. For confidence, a
seeded half of val (n=368), because training-image confidences are inflated; the
other half supplied the clean threshold batches. Thresholds were set from clean data
only (`reports/drift_thresholds.json`) and were not changed after the scorecard ran.

**Scorecard** (`reports/drift_scorecard.csv`): 10 seeded batches of 200 per level.
Corrupted rows show detection rates; the clean test row shows false-alarm rates.

| Corruption | Severity | Batches | Embedding | Pixel stats | Confidence | Abstain rate | Embedding, val ref (post-hoc) |
|---|---|---|---|---|---|---|---|
| clean (control) | 0 | 10 | 80% | 0% | 10% | 0.161 | 20% |
| brightness_down | 1 | 10 | 40% | 100% | 60% | 0.170 | 20% |
| brightness_down | 2 | 10 | 50% | 100% | 100% | 0.190 | 70% |
| brightness_down | 3 | 10 | 100% | 100% | 100% | 0.232 | 100% |
| brightness_up | 1 | 10 | 80% | 100% | 50% | 0.175 | 40% |
| brightness_up | 2 | 10 | 100% | 100% | 100% | 0.201 | 100% |
| brightness_up | 3 | 10 | 100% | 100% | 100% | 0.262 | 100% |
| downscale_upscale | 1 | 10 | 100% | 100% | 100% | 0.330 | 100% |
| downscale_upscale | 2 | 10 | 100% | 100% | 100% | 0.442 | 100% |
| downscale_upscale | 3 | 10 | 100% | 100% | 100% | 0.645 | 100% |
| gaussian_blur | 1 | 10 | 100% | 100% | 100% | 0.222 | 100% |
| gaussian_blur | 2 | 10 | 100% | 100% | 100% | 0.400 | 100% |
| gaussian_blur | 3 | 10 | 100% | 100% | 100% | 0.788 | 100% |
| jpeg_compression | 1 | 10 | 100% | 10% | 100% | 0.232 | 100% |
| jpeg_compression | 2 | 10 | 100% | 100% | 100% | 0.281 | 100% |
| jpeg_compression | 3 | 10 | 100% | 100% | 100% | 0.472 | 100% |
| motion_blur | 1 | 10 | 100% | 100% | 100% | 0.242 | 100% |
| motion_blur | 2 | 10 | 100% | 100% | 100% | 0.415 | 100% |
| motion_blur | 3 | 10 | 100% | 100% | 100% | 0.697 | 100% |

How to read it:
- The pixel-statistics detector is the most useful alarm: 0% false alarms on clean
  test, and every level detected except JPEG severity 1 (10%), which barely moves
  brightness, contrast or sharpness.
- The embedding detector is not usable at its current threshold. It fired on 80% of
  clean test batches (mean AUC 0.562 vs threshold 0.540). Its threshold came from
  val, whose 736 images were resampled into heavily overlapping batches, and that
  did not capture how far clean test embeddings sit from train. Below 100% its
  detection rates are indistinguishable from its false-alarm rate. The threshold was
  not tuned on these results.
- Post-hoc, after seeing the clean-test row above: the `embedding_valref` variant,
  whose reference is val instead of the memorized train images, cuts clean-test false
  alarms from 80% to 20% (mean AUC 0.542 vs threshold 0.552) and still detects every
  level at 100% except mild brightness shifts (brightness_down 20% / 70% at severity
  1 / 2, brightness_up 40% at severity 1). Because this revision was motivated by
  test batches, its numbers are not an unbiased estimate. The rule set before seeing
  them was to switch the live check only at <= 10% clean false alarms; 20% misses
  that, so the live check was not switched (this decision was also made after seeing
  test batches). Pixel statistics stays the primary alarm.
- The confidence detector had 10% false alarms and caught everything except mild
  brightness shifts. The abstain rate rises with severity (0.16 clean -> 0.79 for
  blur severity 3).

**Drift check after every batch run:** `pet_breed_classification.batch` runs the
check by default (`--no-drift` skips it). It writes `reports/drift/latest.json`,
appends to `reports/drift/history.jsonl`, and saves an Evidently DataDrift report
on brightness, contrast, sharpness and max_confidence (latest batch vs clean val) to
`reports/evidently_latest_batch.html`. When `PETBREED_PUSHGATEWAY_URL` is set, it
pushes `petbreed_drift_score{detector}`, `petbreed_drift_alarm{detector}`,
`petbreed_drift_threshold{detector}`, `petbreed_confidence_drift_ks` and
`petbreed_batch_abstain_rate`. The Pushgateway adds its own `job`/`instance` grouping
labels.

**API metrics:** `GET /metrics` exposes `petbreed_request_duration_seconds`
(histogram, labels `endpoint` and `status_class`), `petbreed_prediction_confidence`
(histogram) and `petbreed_predictions_total{decision}`.

**Stack:** `docker compose -f docker/docker-compose.monitoring.yml up -d` starts the
v2 API (8000), Prometheus (9090), Pushgateway (9091) and Grafana (3001), all on
127.0.0.1. Grafana's admin password comes from `GRAFANA_ADMIN_PASSWORD` in
`docker/.env`. Prometheus config is hand-written in `monitoring/prometheus/` and
includes a relabel config (`instance` = compose service name, `python_gc_*` dropped)
and `honor_labels` for the Pushgateway. Alert rules:
- `PetBreedDriftDetected`: `petbreed_drift_alarm == 1`, fires when any detector's
  latest-batch score exceeds its threshold above.
- `PetBreedHighP95`: `/predict` p95 > 0.5 s for 5 minutes. That is 10x the 48 ms
  p95 measured under the 50-user Locust load.

The dashboard "Pet Breed Service" is provisioned from `monitoring/grafana/` with 8
panels: p95 latency with its 0.5 s line, request rate, one drift-score panel per
detector with its threshold, abstain rate, a calibrated-confidence heatmap, and alarm
status. After `down -v` and `up -d`, which removed both volumes, the dashboard (8
panels, provisioned), the datasource and both alert rules came back.
Screenshot (taken by the owner on 2026-10-13; range "Last 30 minutes", about
20:47-21:17, 10 s refresh) during live `/predict` traffic and a pushed
`brightness_down` severity-3 batch of 200 images. `/predict` 2xx runs at about
4.3-4.7 req/s from about 21:04 to 21:14, and its p95 stays at about 50-70 ms, well
under the 0.5 s alert line. Every drift score is far above its threshold: embedding
AUC 0.617 vs 0.540, pixel score 274.16 vs 2.48, confidence KS 0.184 vs 0.098 (exact
values in `reports/drift/latest.json`), and the alarm-status panel shows ALARM for
all three detectors. The live abstain rate moves between about 9% and 18%; the
batch's abstain rate is 0.21. All 8 panels are in the frame:

![Grafana dashboard](../reports/grafana_dashboard.png)

**Demo (2026-10-11):** about 3 minutes of val-image Locust traffic (10 users), then
three batch runs with `PETBREED_PUSHGATEWAY_URL=http://127.0.0.1:9091`:

| Batch (200 images) | Embedding AUC | Pixel score | Confidence KS | Alarms |
|---|---|---|---|---|
| trainval, seeded shuffle | 0.490 | 0.60 | 0.112 | confidence only: about 80% of trainval is training data, which the model is overconfident on (abstain rate 0.035) |
| clean val images (`--input-dir`) | 0.464 | 1.18 | 0.057 | none |
| `brightness_down` severity 3 (`--input-dir`) | 0.617 | 274.16 | 0.184 | all three |

After the last run, Prometheus `/api/v1/alerts` showed three firing alerts:
`PetBreedDriftDetected` with `detector` = `pixel`, `confidence` and `embedding`
(`job="petbreed_batch_drift"`, `severity="warning"`, active since
2026-10-11T17:44:23Z). The Evidently report and `reports/drift/latest.json` now
come from the later retraining demo (1,000 `brightness_down` severity-3 images, see
[Retraining loop](#retraining-loop)). `reports/drift/history.jsonl` holds every batch
from all four runs, and `reports/drift/prometheus_alerts_demo.json` holds the alerts
response.

## Retraining loop

Airflow is not used. The loop is `monitoring/retrain_on_drift.py`, called by the batch
drift check whenever a batch raises any drift alarm.

1. **Storm guard** (state in the git-ignored `data/monitoring/retrain_state.json`): a
   24 h cooldown, at most 1 retrain per UTC day, and at least 1,000 images scored since
   the last retrain. The attempt is recorded before training starts, so a crash cannot
   cause a retrain storm.
2. **Fine-tune** the Production checkpoint (after checking that the registry's
   Production sha256 equals the shipped model's) for 2 epochs, lr 1e-5, AdamW and
   cosine. It trains on TRAIN images; with probability 0.5 each image gets a random
   corruption at a random severity from the existing corruption functions. The
   corrupted test sets are never used. MLflow tracking is on.
3. **Evaluate** on val through the served path (export to ONNX, parity check,
   temperature and threshold fitted on val), then register the run.
4. **Gate:** `registry.promote_if_better(run_id, metric="top1", margin=0.01,
   alias="challenger")` compares the candidate with the current `production` version
   and sets alias `challenger` only if candidate top-1 >= Production top-1 - 0.01. The
   gate never moves `production`: following the handbook's "never auto-promote to
   Production", that needs a human. The candidate is never copied into `artifacts/`,
   and no image is rebuilt.

A human approves a challenger by moving the alias:

```python
from mlflow import MlflowClient
client = MlflowClient("http://localhost:5000")
client.set_registered_model_alias("PetBreedClassifier", "production", client.get_model_version_by_alias("PetBreedClassifier", "challenger").version)
```

Deploying it (new release, `artifacts/` and image) is a separate step. Until then the
next retrain refuses to start, because it checks that the registry's Production
sha256 equals the shipped model's.

**Triggered run (2026-10-11):**
`PETBREED_PUSHGATEWAY_URL=http://127.0.0.1:9091 uv run --group monitor python -m pet_breed_classification.batch --input-dir data/corrupted/brightness_down/severity_3 --limit 1000`
scored 1,000 images. All 5 drift batches alarmed on all three detectors (latest batch:
AUC 0.634, pixel score 273.4, confidence KS 0.209). The guard allowed the retrain
(1,000 images scored, no earlier retrain), and the whole command took 97 s.

| Item | Value |
|---|---|
| MLflow run | `retrain-drift-20261011-180926`, run_id `ba7419ee9d2f4803a6efe44f73ac0b46`, FINISHED, tag `trigger=drift_alarm` |
| Fine-tuned from | `PetBreedClassifier@production` = version 1 (run `d05c2f2b`) |
| Training-loop val accuracy | epoch 1: 0.9484; epoch 2: 0.9497 |
| Served-path val (candidate) | top-1 0.9470, macro-F1 0.9472, ECE 0.0265, T 0.6794, threshold 0.8701, coverage 0.9063, selective accuracy 0.9805 |
| Production val top-1 | 0.9497 |
| Gate | 0.9470 >= 0.9497 - 0.01 = 0.9397, so the candidate passed |
| Registry after the run | version 2 registered (`onnx_sha256` `93f77e33...d523`). The run used the old gate code, which moved `production` from version 1 to version 2 |
| ONNX parity | `max_logit_parity_diff` 7.6e-6 (< 1e-4) |

Caveats, stated plainly:
- The run is tagged `git_dirty=true` because the retraining code was not yet
  committed when the demo ran (it is committed in the phase-8 commit).
- On 2026-10-13 the alias was moved back by hand: `production` = version 1 (run
  `d05c2f2b`, the shipped model), `challenger` = version 2 (run `ba7419ee`), and
  version 2 is tagged `approval=pending` and `note=git_dirty run, not deployed`. The
  gate code now sets only `challenger`. The shipped service (`artifacts/`, release
  `model-v2`, Docker image) serves version 1.
- The candidate was trained on corrupted train images, and it was gated only on
  clean val, where it is 0.27 points lower. Its robustness gain was not measured on
  the corruption suite.
- The storm guard now blocks another retrain until 2026-10-12T18:09Z.
