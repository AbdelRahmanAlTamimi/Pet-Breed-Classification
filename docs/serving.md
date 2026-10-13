# Serving measurements

Locust load tests, the latency-settings reconciliation, the nginx canary, batch scoring and ONNX Runtime vs OpenVINO. Back to the [README](../README.md).

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
