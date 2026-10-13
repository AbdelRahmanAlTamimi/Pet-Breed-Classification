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

## Status

Evidence paths are relative to the repository root. "Owner" marks items only the
repository owner can do.

### Rubric

| # | Area | Status | Evidence |
|---|---|---|---|
| 01 | Code & packaging | done | `pyproject.toml`, `src/pet_breed_classification/`; `uv pip install -e .` into a fresh venv on 2026-10-11, see [Final verification](#final-verification) |
| 02 | API endpoint | done | `src/pet_breed_classification/api/`, `tests/test_api.py`; calibrated `/predict` with abstention |
| 03 | Docker | done | `Dockerfile`, `docker-compose.yml`; [CI #5](https://github.com/AbdelRahmanAlTamimi/Pet-Breed-Classification/actions/runs/38265330455) pushed `latest` and `sha-64b0774` (digest `sha256:6598f14b…7af766`), serving v2; the 3 commands verified on 2026-10-13 |
| 04 | MLflow tracking | done | 6 selection runs + final run + retrain run; `PetBreedClassifier`: alias `production` = version 1 (the shipped model), `challenger` = version 2 (`approval=pending`, see [Retraining loop](#retraining-loop)); comparison screenshot `reports/mlflow_comparison.png` |
| 05 | DVC data versioning | partial (train re-record owner) | `dvc.yaml`, `dvc.lock`; public read-only remote on Cloudflare R2: `dvc pull` of the two archives and the manifest from a fresh clone verified on 2026-10-13, the full corrupted-set pull was not run (`data/README.md`). `dvc repro train` reproduced top-1 exactly on 2026-10-11; on this branch the `train` stage is stale until the owner re-runs it, see [Data versioning and CI](#data-versioning-and-ci) |
| 06 | GitHub Actions CI/CD | done | `.github/workflows/ci.yml`; [CI #5](https://github.com/AbdelRahmanAlTamimi/Pet-Breed-Classification/actions/runs/38265330455) on `main` after merging PR #4: lint, test, quality-gate and docker all green, `latest` and `sha-64b0774` pushed (digest `sha256:6598f14b…7af766`) |
| 07 | Production serving | done | BentoML service `serving/`, Locust reports `reports/locust_*.html`, p95 below |
| 08 | Monitoring | done | `monitoring/`, `reports/drift_scorecard.csv`, alert rules; Grafana screenshot `reports/grafana_dashboard.png` in [Monitoring](#monitoring-and-drift-module-5) |
| 09 | Peer review | not done (owner) | written by the owner, not part of this repository |
| 10 | README & architecture | done | this file |

### Track C checklist: code, packaging, tracking, versioning

| Item | Status | Evidence |
|---|---|---|
| `src/` layout with `pyproject.toml`, clean-checkout tests pass with uv | done | see [Final verification](#final-verification) |
| Manifest JSON and corruption suite from committed scripts; documented | done | `src/pet_breed_classification/data/`, `data/README.md` |
| Label map committed and sorted; test asserts 37 entries | done | `data/label_map.json`, `tests/test_split_data.py` |
| Model in a typed class, loaded once at startup | done | `predict.py` `PetBreedPredictor`, API lifespan |
| Eval transform exported with the weights; train-path vs API-path logits agree to 1e-4 | done | `model_meta.json` `eval_transform`; `tests/test_predict.py`, `tests/test_serialization.py::test_real_validation_parity` |
| `/predict` returns `{breed, species, confidence, top_3, decision, model_version}` | done | `api/schemas.py` |
| Non-image, oversized upload and 4-channel PNG: 422 or 200, never 500, tested with curl | done | curl on the v2 container: 422, 422, 200; `tests/test_api.py` |
| Temperature scaling on val; reliability diagram before/after | done | `reports/calibration.png`, `reports/calibration.json` |
| Abstention threshold from calibrated scores; coverage and selective accuracy in README | done | [Calibration](#shipped-model-v2) |
| `/health` used by the Docker healthcheck | done | `Dockerfile` `HEALTHCHECK` |
| `docker build -t pet-breed .` passes; `docker compose up` serves on 8000 | done | root `Dockerfile` and `docker-compose.yml`; image `pet-breed:0.2.0` 1.58 GB |
| README: exactly 3 commands | done | above |
| MLflow tracking server running, UI accessible | done (local) | `docker/docker-compose.mlflow.yml` (PostgreSQL + MinIO + MLflow 3.17) |
| >= 5 runs across >= 3 backbones with the required params/metrics | done | [MLflow runs](#mlflow-runs-and-model-selection) |
| MLflow comparison screenshot | done | `reports/mlflow_comparison.png`, shown in [MLflow runs](#mlflow-runs-and-model-selection) |
| Best model registered as `PetBreedClassifier`, promoted to Production | done | version 1 (the shipped model) holds alias `production`; the retrain candidate, version 2, holds `challenger` with `approval=pending` |
| Images, manifest and corrupted sets tracked with DVC; `dvc status` clean | partial | `data/raw/*.dvc`, `dvc.lock`; all data pushed to the R2 remote (`dvc status -c -r storage`: in sync). `dvc status` lists only `train` (`changed deps: src`, from phase-10.1) until the owner re-runs `dvc repro train` |
| `dvc repro` reproduces training within ±0.005 top-1 | done (2026-10-11) | difference 0.0 at `146a496`; not re-run on this branch, where `train` is stale |
| GitHub Actions: lint -> test -> docker build -> push on main | done | [CI #5](https://github.com/AbdelRahmanAlTamimi/Pet-Breed-Classification/actions/runs/38265330455): `latest` and `sha-64b0774`, digest `sha256:6598f14b…7af766` |
| CI fails if top-1 drops below baseline; documented | done | `scripts/quality_gate.py`, [Data versioning and CI](#data-versioning-and-ci) |

### Track C checklist: serving, monitoring, optimization

| Item | Status | Evidence |
|---|---|---|
| BentoML service with `batchable=True` Runner; `bentoml models list` shows it | done | `serving/service.py`; model tag `pet_breed_classifier:n5x6k3wfsgmvhnk3` (v2) |
| `bentoml serve` runs `/predict` at localhost:8000 | done | Locust run against it |
| Locust report before optimization | done | `reports/locust_fastapi.html`, `reports/locust_bentoml.html` |
| ONNX export test: PyTorch vs ONNX logits within 1e-4 on 200 images | done | `tests/test_serialization.py::test_real_validation_parity` (needs local artifacts; skipped in CI) |
| ONNX Runtime and OpenVINO CPU paths, p95 for both | done | p95 13.13 ms (ORT) vs 17.57 ms (OpenVINO), `reports/runtime_benchmark.json` |
| TensorRT FP16 on Triton | not done (out of scope) | owner decision |
| Locust after TensorRT | not done (out of scope) | owner decision |
| Batch scoring >= 1,000 images to `data/scoring/output/` | done | 1,000 images, 43.96 images/s |
| nginx canary 5/95; rollout stages in README | done | `serving/nginx/nginx.conf`, `serving/README.md` |
| Embedding drift (domain classifier on penultimate features) logged per batch | done, but not usable as an alarm (80% false alarms on clean test; a post-hoc val-reference variant has 20%) | `reports/drift_scorecard.csv` |
| Cheap pixel-statistics detector logged per batch | done | same |
| Confidence-distribution drift tracked separately | done | same |
| Detector scorecard with false alarms on clean batches | done | [Monitoring](#monitoring-and-drift-module-5) |
| Evidently report on the latest batch | done | `reports/evidently_latest_batch.html` |
| Drift check after every batch scoring run | done | `batch.py` (`--no-drift` skips) |
| Prometheus exposes drift scores and p95 | done | API `/metrics` + Pushgateway gauges |
| Grafana >= 3 panels, screenshot in README | done | provisioned 8-panel dashboard; `reports/grafana_dashboard.png`, shown in [Monitoring](#monitoring-and-drift-module-5) |
| Drift alert -> Airflow retraining DAG -> triggered run in README | partial | drift alarm -> guarded retrain -> triggered run `ba7419ee` shown below; no Airflow DAG |
| Retraining quality gate: promote only if top-1 >= Production - 0.01 | done | `registry.promote_if_better`; a passing candidate gets `challenger` only, Production needs a human. The demo candidate (0.9470 vs 0.9497) passed |
| Structured pruning, INT8 PTQ, QAT, distillation, optimization benchmark table, worse-row explanations | not done (out of scope) | owner decision |
| Final architecture diagram covering all 5 sessions | done | below |

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

## MLflow runs and model selection

Selection rule, declared before results: the highest served-path val macro-F1 among the
six runs below (ONNX Runtime CPU, after temperature scaling). Experiment
`pet-breed-classification` on the local tracking server. All runs: batch 64, 10 epochs,
AdamW, cosine schedule, AMP, seed 42, `git_dirty=false`.

| Run | Backbone | lr | top-1 | macro-F1 | ECE | Temperature | Coverage | Selective acc. |
|---|---|---|---|---|---|---|---|---|
| r50-repro | resnet50 | 1e-4 | 0.9429 | 0.9436 | 0.0231 | 0.9148 | 0.8478 | 0.9808 |
| **r50-lr-half** | resnet50 | 5e-5 | **0.9497** | **0.9499** | 0.0321 | 0.6785 | 0.8736 | 0.9813 |
| r18-a | resnet18 | 1e-4 | 0.9185 | 0.9185 | 0.0405 | 0.6461 | 0.7731 | 0.9807 |
| r18-lr-half | resnet18 | 5e-5 | 0.9076 | 0.9072 | 0.0333 | 0.5062 | 0.7704 | 0.9806 |
| mnv3s-a | mobilenet_v3_small | 1e-4 | 0.8003 | 0.7991 | 0.0228 | 0.6502 | 0.4851 | 0.9804 |
| mnv3s-lr-half | mobilenet_v3_small | 5e-5 | 0.7106 | 0.7079 | 0.0477 | 0.4454 | 0.3139 | 0.9827 |

The winning configuration was retrained once on a clean tree as `r50-lr-half-final`
(run_id `d05c2f2b9ca7481bb657acb5cc2b29f6`, commit `e1844b7`). Its metrics match
`r50-lr-half` exactly, and it is the shipped model. Three earlier attempts named
`r50-lr-half-final` are recorded as FAILED: they hung before epoch 1. The cause was
not proven; tracking now fails fast when the server is unreachable. A
`backfill-resnet50-existing-run` run records the pre-MLflow v1 model from files on
disk.

MLflow UI comparison (taken by the owner; it lists the 6 selection runs, the final
run and the retrain run):

![MLflow run comparison](reports/mlflow_comparison.png)

## Shipped model (v2)

| Item | Value |
|---|---|
| MLflow run | `r50-lr-half-final`, run_id `d05c2f2b9ca7481bb657acb5cc2b29f6` (experiment `pet-breed-classification`) |
| Registry | `PetBreedClassifier` version 1, alias `production`. Version 2 (the drift-triggered retrain candidate) holds `challenger`; see [Retraining loop](#retraining-loop) |
| Backbone | ResNet-50, lr 5e-5, batch 64, 10 epochs, AdamW, cosine, AMP |
| `model.onnx` sha256 | `58165e216e4a5e8945d6763801c31eef1bfc9a2a5ae348b607bb1098f0a88f72` |
| Val top-1 / macro-F1 | 0.9497 / 0.9499 (served path, n=736) |
| Temperature | 0.678530 (ECE 0.0568 -> 0.0321) |
| Abstain threshold | 0.918166 (target selective accuracy 0.98; coverage 0.8736, selective accuracy 0.9813 on val) |

**Calibration** (`reports/calibration.json`, reliability diagram
`reports/calibration.png`): the temperature was fitted on val by minimizing NLL
(0.2207 -> 0.1803). The threshold is the lowest calibrated confidence whose selective
accuracy on val reaches 0.98. A 2-fold swap on val checks that this choice is not
overfitted: fit T and the threshold on one half (368 images), evaluate on the other.

| Fold | T | Threshold | Held-out coverage | Held-out selective accuracy |
|---|---|---|---|---|
| 0 | 0.6654 | 0.9099 | 0.8913 | 0.9756 |
| 1 | 0.6898 | 0.9282 | 0.8587 | 0.9873 |
| pooled | | | 0.8750 | 0.9814 |

The in-sample threshold gives coverage 0.8736 and selective accuracy 0.9813, and the
pooled held-out numbers are within 0.002 of them.

MLflow 3 replaced the deprecated registry stages with aliases. In this project the
alias `production` on `PetBreedClassifier` is what "promoted to Production" means, and
`challenger` marks a candidate that passed the retrain gate and waits for a human.
Naming: `model_version` `v2` in `model_meta.json` counts shipped artifacts (this is the
second one), and that artifact is registry version 1.
The previous shipped model (v1, ONNX sha256 `2695971d...d14a5d`) is kept locally in
`artifacts/previous/`.

## Data versioning and CI

- `dvc repro train` (GPU, MLflow disabled) reproduced the shipped model's val metrics
  exactly (top-1 0.9497282608695652, difference 0.0); a second `dvc repro train` is a
  no-op and `dvc status` reports "Data and pipelines are up to date."
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
- On this branch `dvc status` reports `train: changed deps: modified: src`, because
  phase-10.1 edited `src/pet_breed_classification/registry.py`, a dependency of the
  `train` stage. Re-running `dvc repro train` was not done in this session; the owner
  re-runs it, commits `dvc.lock` and pushes.
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
| BentoML, first run (`max_latency_ms=50`, per-image loop) | 1 + 1 runner | 31 / 58 / 75 | 22.31 | 24.80 | 346 / 4,468 (7.7%), all HTTP 503 on `/predict` (BentoML `ServiceUnavailable: Service Busy`) | `reports/locust_bentoml_run1_latency50.html` |
| BentoML, fixed (`max_latency_ms=500`, one ORT call per batch) | 1 + 1 runner | 33 / 110 / 230 | 22.04 | 24.44 | 0 / 4,405 (0%) | `reports/locust_bentoml.html` |

1-minute load average before -> after each run: FastAPI 0.18 -> 5.93; BentoML
first run 1.47 -> 6.66; BentoML fixed 2.98 -> 7.42. In the first BentoML run the
runnable scored a batch image by image, and the 50 ms latency budget was below
the latency of one inference under load, so the adaptive batcher rejected 8.6% of
`/predict` requests (346 of 4,019; 7.7% of all 4,468 requests). After the fix BentoML has no failures, but its p95 is higher
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
| brightness_up | 0.911 / 0.824 / 0.967 | 0.906 / 0.799 / 0.972 | 0.879 / 0.751 / 0.968 |
| downscale_upscale | 0.811 / 0.687 / 0.913 | 0.698 / 0.553 / 0.890 | 0.560 / 0.365 / 0.827 |
| gaussian_blur | 0.880 / 0.769 / 0.969 | 0.788 / 0.606 / 0.932 | 0.516 / 0.203 / 0.897 |
| jpeg_compression | 0.856 / 0.765 / 0.948 | 0.821 / 0.727 / 0.930 | 0.680 / 0.536 / 0.860 |
| motion_blur | 0.866 / 0.752 / 0.967 | 0.794 / 0.588 / 0.947 | 0.625 / 0.300 / 0.920 |

Brightness shifts cost at most 4 points of top-1. Downscaling, blur and heavy JPEG
cost up to 40.2 points (Gaussian blur severity 3: 0.516 vs 0.918 clean). Abstention absorbs part of the damage, since coverage falls
faster than selective accuracy, but for downscale, blur and JPEG selective accuracy still drops to 0.83-0.92 at
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
Screenshot (taken by the owner, time range 2026-10-11 20:00-21:59). The drift panels
show the last batch of the [retraining run](#retraining-loop) (AUC 0.634, pixel score
273.4, confidence KS 0.209); the demo batches, the `/predict` request rate and the p95
series are not visible in this window, only the 0.5 s threshold line. Seven of the 8
panels are in the frame; the alarm-status panel is not:

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

## ONNX Runtime vs OpenVINO (CPU)

`PETBREED_RUNTIME=openvino` (dependency group `openvino`, `openvino==2026.4.1`) makes
`PetBreedPredictor` run the sha256-verified ONNX file through OpenVINO: converted to
an OpenVINO IR model and compiled for CPU in FP32. Preprocessing, temperature and
threshold are unchanged. The default stays `onnxruntime`.
`uv run --group openvino python -m pet_breed_classification.benchmark_runtimes` also
writes the IR to `artifacts/openvino/model.xml` (git-ignored).

Protocol, identical for both engines: batch 1, 200 val images, 50 warm-up runs
discarded, 500 timed runs of the engine call on preprocessed tensors, 6 threads
(ORT `intra_op_num_threads=6`, `inter_op_num_threads=1`; OpenVINO
`INFERENCE_NUM_THREADS=6`). Hardware: Intel Core i5-12500 (6 cores / 12 threads).
ONNX Runtime 1.30.0, OpenVINO 2026.4.1. The 1-minute load average was 0.54 before
the run. Report: `reports/runtime_benchmark.json`.

| Engine | p50 (ms) | p95 (ms) | mean (ms) |
|---|---|---|---|
| ONNX Runtime (CPU EP) | 12.15 | 13.13 | 12.35 |
| OpenVINO (CPU, FP32) | 13.94 | 17.57 | 14.33 |

Top-1 agreement on the 200 images: 1.000. Maximum absolute difference in calibrated
probability: 2.4e-6. On this CPU, OpenVINO FP32 was slower than ONNX Runtime at batch
1 (p95 +4.4 ms), so the service keeps ONNX Runtime. No INT8 or other OpenVINO
optimization was tried, since that is out of scope.

**Why three ONNX Runtime p95 numbers differ.** They measure different things, and
the settings below were read from the source files that produced them:

| Source | p95 | Model | ORT threads | What is timed |
|---|---|---|---|---|
| Module 1, `reports/module-1.md` (`benchmark_serialization.py`) | 68.6 ms | v1 ResNet-50 (2026-10-05, before v2 existed) | `intra_op_num_threads=1`, `inter_op_num_threads=1` | engine call on preprocessed tensors, batch 1, 200 val images, 20 warm-up runs |
| Phase 7, `reports/runtime_benchmark.json` (`benchmark_runtimes.py`) | 13.13 ms | v2 | `intra_op_num_threads=6`, `inter_op_num_threads=1` | engine call on preprocessed tensors, batch 1, 500 timed runs after 50 warm-up runs |
| Locust, `reports/locust_fastapi.html` | 48 ms | v2 | ORT defaults (`predict.py` passes no `SessionOptions`) | end-to-end HTTP `/predict` (upload, decode, preprocess, inference, JSON) under 50 users, client on the same host |

Single-threaded vs 6 threads explains most of the gap between the first two, and the
Locust number adds HTTP, image decoding, preprocessing and queueing under concurrent
load.

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

Run on 2026-10-13 at commit `defd96f` (later commits change only documentation):

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
- DVC: see [Data versioning and CI](#data-versioning-and-ci). `dvc status` lists the
  `train` stage as changed (`src`) until the owner re-runs `dvc repro train`.
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

## Known limitations

- Airflow is not used. The retraining loop is a guarded Python call from the batch
  drift check, not a DAG.
- Optimization is out of scope by owner decision: no pruning, INT8 PTQ, QAT,
  distillation, TensorRT/Triton or TFLite, and no optimization benchmark table.
- MLflow runs locally (Docker Compose on 127.0.0.1); reviewers cannot open the
  owner's UI.
- The public DVC remote is an `r2.dev` URL, which Cloudflare rate-limits; a full
  pull of the 66,042 corrupted images may be throttled (`data/README.md` has the
  fallback). The `train` stage is stale on this branch until the owner re-runs
  `dvc repro train`.
- Both canary backends run the same v2 image, so the canary exercises routing and
  rollback, not a model comparison. The automatic latency-based rollback guard is not
  implemented.
- The embedding drift detector false-alarms on 80% of clean test batches at its
  clean-val threshold, so the pixel-statistics and confidence detectors are the
  usable alarms. The post-hoc val-reference variant lowers that to 20%, still above
  the 10% bar for switching the live check, and it was revised after seeing test
  batches.
- On test, selective accuracy at the val-chosen threshold is 0.974, below the 0.98
  target, and it drops to 0.83-0.92 under severity-3 downscale, blur and JPEG corruptions.
- Locust ran on the same host as the services, so client load affects the latencies.
- The Docker Hub image is built for linux/amd64 only.
- GitGuardian flags commit `548b1db` (`docker/docker-compose.mlflow.yml`) as a
  "Generic Password". The line is the PostgreSQL URI built from `$$POSTGRES_USER` and
  `$$POSTGRES_PASSWORD` environment references, with no literal secret; the owner
  should mark it as a false positive.

## Session changelog

| Module | What landed | Key commits |
|---|---|---|
| 1. Packaging, API, Docker | dataset split, corruption suite, manifest, first ResNet-50, FastAPI `/predict` and `/health`, Docker image, tests | `63ddec9`, `58d588c`, `c3abfb9`, `49971f8`, PR #2 `cca208b` |
| 2. MLflow, DVC, CI | MLflow stack; temperature scaling and abstention; ResNet-18 and MobileNetV3 backbones; isolated runs; registry helpers; DVC train stage; CI quality gate; final run shipped as v2; stable DVC hashes; CI on PRs | `548b1db`, `b4e2ea1`, `df36f60`, `25fe59f`, `1005f56`, `2b4c116`, `133fad7`, `ac409c1`, `7f13329`, `844ec8d`, `5818ba6`, `756da0b`, `e30a0c1` |
| 3. Serving | BentoML batchable Runner, Locust, batch scoring, nginx canary; batching and rollback fixes; measurements | `9d5bd4a`, `9340734`, `795727a`, `cdf317d`, `a8196ef`, `cafa030` |
| 4. Evaluation and CPU runtimes | one-shot final test evaluation and robustness; OpenVINO CPU path and benchmark; optimization out of scope | `78bea08`, `aab2b8d` |
| 5. Monitoring and retraining | drift features, three detectors, scorecard, Evidently, `/metrics`, Prometheus/Pushgateway/Grafana, alerts; drift-triggered retraining loop | `15abeed`, phase-8 commit |
| Final fixes (2026-10-13) | retrain gate sets `challenger`, Production back to version 1; post-hoc val-reference embedding detector and scorecard column; DVC sees `data/raw/*.dvc`; public DVC remote on R2; README: Docker Hub, registry, DVC, latency settings, screenshots, license | `7a44650`, `38e53a9`, `b801454`, `024f0f5`, `a16e7db`, `455a066`, `e051f60`-`defd96f` |

## Dataset and license

Oxford-IIIT Pet Dataset: O. M. Parkhi, A. Vedaldi, A. Zisserman and C. V. Jawahar,
"Cats and Dogs", IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
2012. https://www.robots.ox.ac.uk/~vgg/data/pets/. The dataset page (checked on
2026-10-13) states: "The dataset is available to download for commercial/research
purposes under a Creative Commons Attribution-ShareAlike 4.0 International License.
The copyright remains with the original owners of the images."
