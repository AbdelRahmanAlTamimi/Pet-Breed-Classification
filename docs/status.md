# Project status

Full rubric evidence, both Track C checklists, the detailed known limitations and the full session changelog. Back to the [README](../README.md).

## Status

Evidence paths are relative to the repository root. "Owner" marks items only the
repository owner can do.

### Rubric

| # | Area | Status | Evidence |
|---|---|---|---|
| 01 | Code & packaging | done | `pyproject.toml`, `src/pet_breed_classification/`; `uv pip install -e .` into a fresh venv on 2026-10-11, see [Final verification](../README.md#final-verification) |
| 02 | API endpoint | done | `src/pet_breed_classification/api/`, `tests/test_api.py`; calibrated `/predict` with abstention |
| 03 | Docker | done | `Dockerfile`, `docker-compose.yml`; [CI #5](https://github.com/AbdelRahmanAlTamimi/Pet-Breed-Classification/actions/runs/38265330455) pushed `latest` and `sha-64b0774` (digest `sha256:6598f14b…7af766`), serving v2; the 3 commands verified on 2026-10-13 |
| 04 | MLflow tracking | done | 6 selection runs + final run + retrain run; `PetBreedClassifier`: alias `production` = version 1 (the shipped model), `challenger` = version 2 (`approval=pending`, see [Retraining loop](../README.md#retraining-loop)); comparison screenshot `reports/mlflow_comparison.png` |
| 05 | DVC data versioning | done | `dvc.yaml`, `dvc.lock`; public read-only remote on Cloudflare R2: `dvc pull` of the two archives and the manifest from a fresh clone verified on 2026-10-13, the full corrupted-set pull was not run (`data/README.md`). `dvc repro train` reproduced top-1 exactly on 2026-10-11 and again at `42607b8` on 2026-10-13; `dvc status` is clean and `dvc status -c -r storage` is in sync, see [Data versioning and CI](../README.md#data-versioning-and-ci) |
| 06 | GitHub Actions CI/CD | done | `.github/workflows/ci.yml`; [CI #5](https://github.com/AbdelRahmanAlTamimi/Pet-Breed-Classification/actions/runs/38265330455) on `main` after merging PR #4: lint, test, quality-gate and docker all green, `latest` and `sha-64b0774` pushed (digest `sha256:6598f14b…7af766`) |
| 07 | Production serving | done | BentoML service `serving/`, Locust reports `reports/locust_*.html`, p95 below |
| 08 | Monitoring | done | `monitoring/`, `reports/drift_scorecard.csv`, alert rules; Grafana screenshot `reports/grafana_dashboard.png` in [Monitoring](../README.md#monitoring-and-drift-module-5) |
| 09 | Peer review | not done (owner) | written by the owner, not part of this repository |
| 10 | README & architecture | done | this file |

### Track C checklist: code, packaging, tracking, versioning

| Item | Status | Evidence |
|---|---|---|
| `src/` layout with `pyproject.toml`, clean-checkout tests pass with uv | done | see [Final verification](../README.md#final-verification) |
| Manifest JSON and corruption suite from committed scripts; documented | done | `src/pet_breed_classification/data/`, `data/README.md` |
| Label map committed and sorted; test asserts 37 entries | done | `data/label_map.json`, `tests/test_split_data.py` |
| Model in a typed class, loaded once at startup | done | `predict.py` `PetBreedPredictor`, API lifespan |
| Eval transform exported with the weights; train-path vs API-path logits agree to 1e-4 | done | `model_meta.json` `eval_transform`; `tests/test_predict.py`, `tests/test_serialization.py::test_real_validation_parity` |
| `/predict` returns `{breed, species, confidence, top_3, decision, model_version}` | done | `api/schemas.py` |
| Non-image, oversized upload and 4-channel PNG: 422 or 200, never 500, tested with curl | done | curl on the v2 container: 422, 422, 200; `tests/test_api.py` |
| Temperature scaling on val; reliability diagram before/after | done | `reports/calibration.png`, `reports/calibration.json` |
| Abstention threshold from calibrated scores; coverage and selective accuracy in README | done | [Calibration](../README.md#shipped-model-v2) |
| `/health` used by the Docker healthcheck | done | `Dockerfile` `HEALTHCHECK` |
| `docker build -t pet-breed .` passes; `docker compose up` serves on 8000 | done | root `Dockerfile` and `docker-compose.yml`; image `pet-breed:0.2.0` 1.58 GB |
| README: exactly 3 commands | done | above |
| MLflow tracking server running, UI accessible | done (local) | `docker/docker-compose.mlflow.yml` (PostgreSQL + MinIO + MLflow 3.17) |
| >= 5 runs across >= 3 backbones with the required params/metrics | done | [MLflow runs](../README.md#mlflow-runs-and-model-selection) |
| MLflow comparison screenshot | done | `reports/mlflow_comparison.png`, shown in [MLflow runs](../README.md#mlflow-runs-and-model-selection) |
| Best model registered as `PetBreedClassifier`, promoted to Production | done | version 1 (the shipped model) holds alias `production`; the retrain candidate, version 2, holds `challenger` with `approval=pending` |
| Images, manifest and corrupted sets tracked with DVC; `dvc status` clean | done | `data/raw/*.dvc`, `dvc.lock`; all data and the `train` outputs pushed to the R2 remote (`dvc status -c -r storage`: in sync); `dvc status`: "Data and pipelines are up to date." (2026-10-13, `42607b8`) |
| `dvc repro` reproduces training within ±0.005 top-1 | done | difference 0.0 at `146a496` (2026-10-11) and at `42607b8` (2026-10-13, `reports/metrics.json` unchanged) |
| GitHub Actions: lint -> test -> docker build -> push on main | done | [CI #5](https://github.com/AbdelRahmanAlTamimi/Pet-Breed-Classification/actions/runs/38265330455): `latest` and `sha-64b0774`, digest `sha256:6598f14b…7af766` |
| CI fails if top-1 drops below baseline; documented | done | `scripts/quality_gate.py`, [Data versioning and CI](../README.md#data-versioning-and-ci) |

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
| Detector scorecard with false alarms on clean batches | done | [Monitoring](../README.md#monitoring-and-drift-module-5) |
| Evidently report on the latest batch | done | `reports/evidently_latest_batch.html` |
| Drift check after every batch scoring run | done | `batch.py` (`--no-drift` skips) |
| Prometheus exposes drift scores and p95 | done | API `/metrics` + Pushgateway gauges |
| Grafana >= 3 panels, screenshot in README | done | provisioned 8-panel dashboard; `reports/grafana_dashboard.png` (2026-10-13, live `/predict` traffic and a pushed `brightness_down` severity-3 batch: p95 under the 0.5 s line, all three drift scores above threshold, all three alarms firing), shown in [Monitoring](../README.md#monitoring-and-drift-module-5) |
| Drift alert -> Airflow retraining DAG -> triggered run in README | partial | drift alarm -> guarded retrain -> triggered run `ba7419ee` shown below; no Airflow DAG |
| Retraining quality gate: promote only if top-1 >= Production - 0.01 | done | `registry.promote_if_better`; a passing candidate gets `challenger` only, Production needs a human. The demo candidate (0.9470 vs 0.9497) passed |
| Structured pruning, INT8 PTQ, QAT, distillation, optimization benchmark table, worse-row explanations | not done (out of scope) | owner decision |
| Final architecture diagram covering all 5 sessions | done | below |

## Known limitations

- Airflow is not used. The retraining loop is a guarded Python call from the batch
  drift check, not a DAG.
- Optimization is out of scope by owner decision: no pruning, INT8 PTQ, QAT,
  distillation, TensorRT/Triton or TFLite, and no optimization benchmark table.
- MLflow runs locally (Docker Compose on 127.0.0.1); reviewers cannot open the
  owner's UI.
- The public DVC remote is an `r2.dev` URL, which Cloudflare rate-limits; a full
  pull of the 66,042 corrupted images may be throttled (`data/README.md` has the
  fallback).
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
