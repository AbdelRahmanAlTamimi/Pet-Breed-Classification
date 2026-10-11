# Pet Breed Classifier

Run the published CPU inference service and classify a local pet photo in three
commands. Replace `/path/to/pet.jpg` with the image you want to classify.

```bash
docker pull abdelrahmanaltamimi/pet-breed:latest
docker compose up -d
curl --fail-with-body --retry 30 --retry-delay 1 --retry-all-errors \
  -X POST http://localhost:8000/predict \
  -F "file=@/path/to/pet.jpg"
```

The response includes the predicted breed, species, confidence, top three
breeds, and whether the model is confident or uncertain.

> Status (work in progress): the image tagged `latest` on Docker Hub was pushed on
> 2026-10-05 and predates model v2. CI pushes a v2 image when the pull request is
> merged to `main`. Until then, build v2 locally with `docker build -t pet-breed .`.

## Shipped model (v2)

| Item | Value |
|---|---|
| MLflow run | `r50-lr-half-final`, run_id `d05c2f2b9ca7481bb657acb5cc2b29f6` (experiment `pet-breed-classification`) |
| Registry | `PetBreedClassifier` version 1, alias `production` |
| Backbone | ResNet-50, lr 5e-5, batch 64, 10 epochs, AdamW, cosine, AMP |
| `model.onnx` sha256 | `58165e216e4a5e8945d6763801c31eef1bfc9a2a5ae348b607bb1098f0a88f72` |
| Val top-1 / macro-F1 | 0.9497 / 0.9499 (served path, n=736) |
| Temperature | 0.678530 (ECE 0.0568 -> 0.0321) |
| Abstain threshold | 0.918166 (target selective accuracy 0.98; coverage 0.8736, selective accuracy 0.9813 on val) |

MLflow 3 replaced the deprecated registry stages with aliases. In this project the
alias `production` on `PetBreedClassifier` is what "promoted to Production" means.
The previous shipped model (v1, ONNX sha256 `2695971d...d14a5d`) is kept locally in
`artifacts/previous/`.

## Data versioning and CI

- `dvc repro train` (GPU, MLflow disabled) reproduced the shipped model's val metrics
  exactly (top-1 0.9497282608695652, difference 0.0); a second `dvc repro train` is a
  no-op and `dvc status` reports "Data and pipelines are up to date."
- The DVC remote is local to the owner's machine (git-ignored `.dvc/config.local`).
  Reviewers rebuild the data from the official archives; see `data/README.md`.
- CI (`.github/workflows/ci.yml`): lint -> test (pytest, 70% coverage gate) ->
  quality gate -> docker. The quality gate fails if val top-1 in
  `reports/metrics.json` < top-1 in `reports/baseline_metrics.json` - 0.01. The
  baseline is the shipped run's served-path val metrics, so for now the two files
  are equal. Pull requests only build the image. Pushes to `main` also push
  `latest` and `sha-<short>` to Docker Hub, but only when the repository secrets
  `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` exist.
- First green PR run: https://github.com/AbdelRahmanAlTamimi/Pet-Breed-Classification/actions/runs/38157794666

## Serving measurements (model v2)

Hardware: Intel Core i5-12500 (6 cores / 12 threads), CPU only. Locust ran on the
same host as the service, so client and server shared the CPU. The workload used
`loadtest/locustfile.py`: 736 distinct val images, 90% `/predict` and 10% `/health`,
`between(1, 3)` wait, 50 users, spawn rate 5/s. Each run lasted 190 s with
`--reset-stats`, so the numbers cover about 3 minutes after the ramp. One service
ran at a time with one worker. ONNX Runtime used its default thread settings in
both services (no explicit `intra_op`/`inter_op` configuration, no container CPU
limit).

| Service | Workers | `/predict` p50 / p95 / p99 (ms) | `/predict` RPS | All RPS | Failures | Report |
|---|---|---|---|---|---|---|
| FastAPI (container `pet-breed:0.2.0`) | 1 | 28 / 48 / 68 | 21.93 | 24.52 | 0 / 4,417 (0%) | `reports/locust_fastapi.html` |
| BentoML, first run (`max_latency_ms=50`, per-image loop) | 1 + 1 runner | 31 / 58 / 75 | 22.31 | 24.80 | 346 / 4,468 (7.7%), all `503 Service Busy` on `/predict` | `reports/locust_bentoml_run1_latency50.html` |
| BentoML, fixed (`max_latency_ms=500`, one ORT call per batch) | 1 + 1 runner | 33 / 110 / 230 | 22.04 | 24.44 | 0 / 4,405 (0%) | `reports/locust_bentoml.html` |

1-minute load average before -> after each run: FastAPI 0.18 -> 5.93; BentoML
first run 1.47 -> 6.66; BentoML fixed 2.98 -> 7.42. In the first BentoML run the
runnable scored a batch image by image, and the 50 ms latency budget was below
the latency of one inference under load, so the adaptive batcher rejected 8.6% of
`/predict` requests. After the fix BentoML has no failures, but its p95 is higher
than FastAPI's at this load (110 ms vs 48 ms): batching adds queueing time, and
with 50 users there is little concurrency to amortize it.

**nginx canary** (`docker/docker-compose.canary.yml`, both backends `pet-breed:0.2.0`,
nginx `1.27.5-alpine@sha256:65645c7b...2a10`): `nginx -t` reported "syntax is ok" and
"test is successful". With weights 95/5, 1,000 `/predict` requests split
**940 stable / 60 canary** (all HTTP 200). `serving/rollback.sh` took **0.147 s**
wall-clock. After the rollback, 1,000 requests went **1,000 stable / 0 canary**. The
first rollback attempt did not take effect: the script used `sed -i`, which the
single-file bind mount does not see, and 50 of 1,000 requests still reached the
canary. The script now rewrites the file in place. Rollout stages are listed in
`serving/README.md`. The automatic latency-based rollback guard is not implemented.

**Batch scoring** (`uv run python -m pet_breed_classification.batch --limit 1000`):
1,000 trainval images in 22.75 s (43.96 images/s), peak RSS 1,403,888 KiB,
1,000 rows in `data/scoring/output/batch_predictions.csv`. The 1-minute load
average was 7.50 when the run started, which may have slowed it.
`--input-dir PATH` scores any unlabeled image directory instead.

## Final test evaluation (one shot, model frozen)

`uv run python -m pet_breed_classification.evaluate_test` ran once on the official
test split through the served path: ONNX Runtime on CPU, the shipped temperature
0.678530 and the shipped threshold 0.918166. Nothing was changed after it ran. The
module refuses to overwrite `reports/final_test_evaluation.json`.

| Metric | Val (n=736) | Test (n=3,669) | Why they differ |
|---|---|---|---|
| top-1 | 0.9497 | 0.9182 | Val chose the winner among 6 runs, so its number is optimistically biased; test is unseen and 5x larger. |
| macro-F1 | 0.9499 | 0.9173 | Same selection bias as top-1. |
| ECE (calibrated) | 0.0321 | 0.0352 | The temperature was fitted on val, so it fits val slightly better. |
| coverage | 0.8736 | 0.8449 | The model is less accurate on test, so fewer predictions clear the val-chosen threshold. |
| selective accuracy | 0.9813 | 0.9739 | The threshold was set to reach 0.98 on val; on test it reaches 0.974, which misses the target by 0.6 points. |

The 10 most-confused pairs (true -> predicted, count) on test:
american_pit_bull_terrier -> staffordshire_bull_terrier 30;
Egyptian_Mau -> Bengal 14; Ragdoll -> Birman 14;
staffordshire_bull_terrier -> american_pit_bull_terrier 11; Abyssinian -> Bengal 10;
american_pit_bull_terrier -> american_bulldog 10; Russian_Blue -> Bombay 10;
Maine_Coon -> Bengal 9; staffordshire_bull_terrier -> american_bulldog 9;
Birman -> Ragdoll 8. These are look-alike breeds, and they are why the service
abstains rather than guessing. Confusion matrix: `reports/confusion_matrix.png`.

### Robustness (corrupted test copies)

For each corruption x severity, a seeded sample (seed 42) of n=1,000 of the 3,669
corrupted test images went through the same served path. Full table, including ECE
and macro-F1: `reports/robustness.csv`.

| Corruption | Sev 1 top-1 / cov / sel-acc | Sev 2 top-1 / cov / sel-acc | Sev 3 top-1 / cov / sel-acc |
|---|---|---|---|
| brightness_down | 0.910 / 0.825 / 0.973 | 0.901 / 0.812 / 0.973 | 0.903 / 0.766 / 0.980 |
| brightness_up | 0.911 / 0.824 / 0.967 | 0.906 / 0.799 / 0.973 | 0.879 / 0.751 / 0.968 |
| downscale_upscale | 0.811 / 0.687 / 0.913 | 0.698 / 0.553 / 0.890 | 0.560 / 0.365 / 0.827 |
| gaussian_blur | 0.880 / 0.769 / 0.969 | 0.788 / 0.606 / 0.932 | 0.516 / 0.203 / 0.897 |
| jpeg_compression | 0.856 / 0.765 / 0.948 | 0.821 / 0.727 / 0.930 | 0.680 / 0.536 / 0.860 |
| motion_blur | 0.866 / 0.752 / 0.967 | 0.794 / 0.588 / 0.947 | 0.625 / 0.300 / 0.920 |

Brightness shifts cost at most 4 points of top-1. Downscaling, blur and heavy JPEG
cost up to 40 points. Abstention absorbs part of the damage, since coverage falls
faster than selective accuracy, but selective accuracy still drops to 0.83-0.92 at
severity 3, so the threshold does not hold its 0.98 promise under these shifts.

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

References: clean train images for embedding and pixel statistics. For confidence, a
seeded half of val (n=368), because training-image confidences are inflated; the
other half supplied the clean threshold batches. Thresholds were set from clean data
only (`reports/drift_thresholds.json`) and were not changed after the scorecard ran.

**Scorecard** (`reports/drift_scorecard.csv`): 10 seeded batches of 200 per level.
Corrupted rows show detection rates; the clean test row shows false-alarm rates.

| Corruption | Severity | Batches | Embedding | Pixel stats | Confidence | Abstain rate |
|---|---|---|---|---|---|---|
| clean (control) | 0 | 10 | 80% | 0% | 10% | 0.161 |
| brightness_down | 1 | 10 | 40% | 100% | 60% | 0.170 |
| brightness_down | 2 | 10 | 50% | 100% | 100% | 0.190 |
| brightness_down | 3 | 10 | 100% | 100% | 100% | 0.232 |
| brightness_up | 1 | 10 | 80% | 100% | 50% | 0.175 |
| brightness_up | 2 | 10 | 100% | 100% | 100% | 0.201 |
| brightness_up | 3 | 10 | 100% | 100% | 100% | 0.262 |
| downscale_upscale | 1 | 10 | 100% | 100% | 100% | 0.330 |
| downscale_upscale | 2 | 10 | 100% | 100% | 100% | 0.442 |
| downscale_upscale | 3 | 10 | 100% | 100% | 100% | 0.645 |
| gaussian_blur | 1 | 10 | 100% | 100% | 100% | 0.222 |
| gaussian_blur | 2 | 10 | 100% | 100% | 100% | 0.400 |
| gaussian_blur | 3 | 10 | 100% | 100% | 100% | 0.787 |
| jpeg_compression | 1 | 10 | 100% | 10% | 100% | 0.232 |
| jpeg_compression | 2 | 10 | 100% | 100% | 100% | 0.281 |
| jpeg_compression | 3 | 10 | 100% | 100% | 100% | 0.472 |
| motion_blur | 1 | 10 | 100% | 100% | 100% | 0.241 |
| motion_blur | 2 | 10 | 100% | 100% | 100% | 0.415 |
| motion_blur | 3 | 10 | 100% | 100% | 100% | 0.697 |

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
Screenshot: `reports/grafana_dashboard.png` (taken by the owner).

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
2026-10-11T17:44:23Z). The Evidently report and `reports/drift/latest.json` are
from that run.
