# Pet Breed Classifier

A calibrated 37-breed classifier for user-submitted pet photos (Oxford-IIIT Pet), served
on CPU with ONNX Runtime. It returns the top-3 breeds with calibrated probabilities and
answers `"uncertain"` instead of guessing when the calibrated confidence is below the
abstention threshold.

## Run it (3 commands)

```bash
docker pull abdelrahmanaltamimi/pet-breed:latest
docker compose up -d
curl --retry 30 --retry-delay 1 --retry-all-errors -F "file=@/path/to/pet.jpg" http://localhost:8000/predict
```

`docker compose up -d` uses `docker-compose.yml` in the repository root and serves on
port 8000. Expected response shape (this one is model v2 on the val image
`Abyssinian_10.jpg`):

```json
{"breed":"Abyssinian","species":"cat","confidence":0.9793733225963039,
 "top_3":[{"breed":"Abyssinian","probability":0.9793733225963039},
          {"breed":"Egyptian_Mau","probability":0.006867481899213621},
          {"breed":"Russian_Blue","probability":0.00637048607088967}],
 "decision":"confident","model_version":"v2",
 "correlation_id":"5fb84e84-49e7-411c-8347-0e2cc45d7e3d","latency_ms":24.92}
```

The image is `abdelrahmanaltamimi/pet-breed`, pushed by CI from `main` as `latest` and
`sha-<short>` (linux/amd64). On 2026-10-13, after removing every local `pet-breed` image,
these three commands pulled `latest` (digest
`sha256:6598f14b8649bc41b04ac7428743ada14be800c4be43467372cf38417e7af766`, the same as
`sha-64b0774`) and returned the response above for `Abyssinian_10.jpg`.

Other endpoints: `GET /health` returns `{"status":"healthy","model_version":"v2","num_classes":37}`.
`GET /metadata` returns the temperature, threshold and sha256. `POST /predict/batch`
accepts up to 32 files. `GET /metrics` serves Prometheus metrics.

## Architecture

```mermaid
flowchart LR
  subgraph M1["Module 1-2: data and training"]
    RAW["Oxford-IIIT Pet archives<br/>(DVC-tracked .tar.gz)"] --> SPLIT["split_index.json<br/>label_map.json"]
    SPLIT --> CORR["dvc stage: generate_corruptions<br/>6 corruptions x 3 severities"]
    CORR --> MAN["dvc stage: build_manifest<br/>+ validate_manifest"]
    MAN --> TRAIN["dvc stage: train<br/>ResNet-50, export ONNX,<br/>temperature + threshold on val"]
    TRAIN --> MLF[("MLflow 3.17<br/>PostgreSQL + MinIO")]
    MLF --> REG["Registry: PetBreedClassifier<br/>aliases production, challenger"]
  end
  subgraph M2["Module 2: CI/CD"]
    GH["GitHub Actions<br/>lint -> test -> quality gate -> docker"] --> REL["release model-v2<br/>(model.onnx + meta, sha256-checked)"]
    GH --> HUB["Docker Hub<br/>pet-breed:latest / sha-*"]
  end
  subgraph M3["Module 3-4: serving"]
    API["FastAPI + ONNX Runtime<br/>(OpenVINO switch)"]
    BENTO["BentoML batchable Runner"]
    NGINX["nginx canary 95/5"] --> API
    BATCH["batch scoring CLI"]
  end
  subgraph M5["Module 5: monitoring"]
    DRIFT["drift check: embedding,<br/>pixel stats, confidence<br/>+ Evidently report"]
    PGW["Pushgateway"] --> PROM["Prometheus<br/>alerts"]
    PROM --> GRAF["Grafana dashboard"]
    RETRAIN["retrain_on_drift<br/>storm guard -> fine-tune<br/>-> gate -> challenger"]
  end
  REG --> REL --> HUB --> API
  REG --> BENTO
  API -- /metrics --> PROM
  BATCH --> DRIFT --> PGW
  DRIFT -- alarm --> RETRAIN --> MLF
```

## Data versioning and CI

- `dvc repro train` (GPU, MLflow disabled) reproduced the shipped model's val metrics
  exactly (top-1 0.9497282608695652, difference 0.0); a second `dvc repro train` is a
  no-op and `dvc status` reports "Data and pipelines are up to date."
- Phase-10.1 edited `src/pet_breed_classification/registry.py`, a dependency of the
  `train` stage, so the owner re-ran `dvc repro train` on 2026-10-13 (commit
  `42607b8`). `reports/metrics.json` is unchanged (top-1 0.9497282608695652, same md5
  in `dvc.lock`); `dvc status` reports "Data and pipelines are up to date." and
  `dvc status -c -r storage` reports "Cache and remote 'storage' are in sync."
- DVC remotes: `public` (default, committed) is the read-only `r2.dev` HTTP URL of a
  Cloudflare R2 bucket; `storage` (S3 endpoint, used by the owner to push) and its
  credentials are only in the git-ignored `.dvc/config.local`. On 2026-10-13
  `dvc push -r storage` uploaded 66,051 files in 1 h 11 min, then the two archives in
  89 s (see the next point); the bucket holds 66,053 objects, 4,352,746,975 bytes, and
  `dvc status -c -r storage` reports "Cache and remote 'storage' are in sync."
- From a fresh clone without `config.local`, `dvc pull` of the two archives through
  `public` took 83.9 s (811,092,049 bytes, both md5 values match the `.dvc` files), and
  `dvc pull data/processed/manifest.json` took 4.6 s (22,964,628 bytes, md5 matches).
  The full 66,042-image pull was not run; `dvc status -c` for the whole project
  against `public` listed every other tracked file as present on the remote and none
  as missing.
- Bug fixed on 2026-10-13: the `.gitignore` rule `data/raw/*` made DVC treat
  `data/raw/` as ignored, so workspace-wide `dvc push`/`status`/`pull` never saw
  `data/raw/*.dvc` and the archives had never been pushed to any remote. The rule now
  names the extracted directories and the archives explicitly.
- CI (`.github/workflows/ci.yml`): lint -> test (pytest, 70% coverage gate) ->
  quality gate -> docker. The quality gate fails if val top-1 in
  `reports/metrics.json` < top-1 in `reports/baseline_metrics.json` - 0.01. The
  baseline is the shipped run's served-path val metrics, so for now the two files
  are equal. Pull requests only build the image. Pushes to `main` also push
  `latest` and `sha-<short>` to Docker Hub (secrets `DOCKERHUB_USERNAME` and
  `DOCKERHUB_TOKEN`).
- First green PR run: https://github.com/AbdelRahmanAlTamimi/Pet-Breed-Classification/actions/runs/38157794666
- First push from `main` (CI #5, merge commit `64b0774`):
  https://github.com/AbdelRahmanAlTamimi/Pet-Breed-Classification/actions/runs/38265330455.
  `docker manifest inspect` shows the same digest for `latest` and `sha-64b0774`:
  `sha256:6598f14b8649bc41b04ac7428743ada14be800c4be43467372cf38417e7af766`.

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

![Grafana dashboard](reports/grafana_dashboard.png)

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

## Final verification

Run on 2026-10-13 at commit `defd96f`; the DVC line was re-checked at `42607b8`.
Later commits change only documentation, `.env.example`, `dvc.lock` and the Grafana
screenshot:

- Fresh `git clone` of `final-fixes` (default groups, no data or model artifacts):
  `uv run pytest` gave 215 passed and 8 skipped (optional groups, or local
  data/artifacts absent), with coverage 83.85% against the 70% gate. `ruff check`,
  `uv lock --check` and actionlint 1.7.7 on `.github/workflows/ci.yml` were clean.
- Docker Hub: `latest` and `sha-64b0774` have the same digest
  `sha256:6598f14b…7af766`. With every local `pet-breed` image removed, the three
  commands at the top returned Abyssinian 0.9793733225963039 "confident",
  `model_version` `v2`, and `GET /metadata` reported sha256 `58165e21…f88f72`.
- Registry: `get_model_version_by_alias` returns version 1 (run `d05c2f2b…`) for
  `production` and version 2 (run `ba7419ee…`, `approval=pending`) for `challenger`;
  the retrain loop's Production check against `artifacts/` passes.
- DVC: `dvc status` reports "Data and pipelines are up to date." and
  `dvc status -c -r storage` reports in sync; see
  [Data versioning and CI](#data-versioning-and-ci).
- No `.env`, `config.local`, `.onnx`, `.pt`, `.parquet` or `output.md` files are
  tracked, and no secret value from the local env files or `.dvc/config.local`
  appears in any of the 138 tracked files (the public `r2.dev` URL is in
  `.dvc/config` by design).

The previous verification (2026-10-11, commit `146a496`) also re-ran
`dvc repro train` (metrics identical, top-1 0.9497282608695652) and installed the
package with `uv pip install -e .` into a fresh venv (predicts Abyssinian 0.979373
"confident").

## Reproduce

**Data and training (DVC).** Pull the data from the public remote (details, md5
checks and the fallback from the official archives are in `data/README.md`), then run:

```bash
uv sync
dvc pull data/raw/annotations.tar.gz.dvc data/raw/images.tar.gz.dvc
(cd data/raw && tar xzf images.tar.gz && tar xzf annotations.tar.gz)
dvc pull             # corrupted sets, manifest, train output (r2.dev is rate-limited)
dvc repro            # re-runs only stages whose dependencies changed (train: GPU; MLflow off)
```

**MLflow stack.**

```bash
cp docker/.env.example docker/.env   # then change the passwords
docker compose -f docker/docker-compose.mlflow.yml up -d   # UI on http://127.0.0.1:5000
PETBREED_MLFLOW_ENABLED=true uv run python -m pet_breed_classification.train \
  --run-name my-run --lr 0.00005 --batch-size 64 --epochs 10
```

**Build the service from source.**

```bash
mkdir -p artifacts
gh release download model-v2 --pattern model.onnx --pattern model_meta.json --dir artifacts
sha256sum --check ci/model_sha256.txt
docker build -t pet-breed .
docker run --rm -p 8000:8000 pet-breed
```

**Quality gate.** CI fails if top-1 < baseline - 0.01, where top-1 comes from
`reports/metrics.json` (written by `dvc repro train`) and the baseline from
`reports/baseline_metrics.json`:
`python scripts/quality_gate.py reports/metrics.json reports/baseline_metrics.json`.

**Serving, monitoring and evaluation commands.**

```bash
uv run --group serve python serving/import_model.py      # BentoML model store
BENTOML_CONFIG=serving/bentoml-config.yaml uv run --group serve bentoml serve serving/service.py:svc --port 8000
uv run --group monitor python -m monitoring.thresholds    # clean-data thresholds
uv run --group monitor python -m monitoring.scorecard     # detector scorecard
uv run --group monitor python -m pet_breed_classification.batch --limit 1000   # scoring + drift check
docker compose -f docker/docker-compose.monitoring.yml up -d                  # Grafana on 127.0.0.1:3001
```

## Dataset and license

Oxford-IIIT Pet Dataset: O. M. Parkhi, A. Vedaldi, A. Zisserman and C. V. Jawahar,
"Cats and Dogs", IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
2012. https://www.robots.ox.ac.uk/~vgg/data/pets/. The dataset page (checked on
2026-10-13) states: "The dataset is available to download for commercial/research
purposes under a Creative Commons Attribution-ShareAlike 4.0 International License.
The copyright remains with the original owners of the images."
