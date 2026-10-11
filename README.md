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
