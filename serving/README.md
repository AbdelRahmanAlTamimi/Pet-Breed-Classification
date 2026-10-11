# Production Serving

The BentoML service uses BentoML `1.4.30`'s class-based `@bentoml.service` and
`@bentoml.api` decorators. `PetBreedPredictor` is loaded once in the service
constructor and is also the only predictor used by batch scoring and the FastAPI
service. The evaluator transform, temperature, abstention threshold, and ONNX
SHA-256 therefore come from the same metadata file.

## BentoML

Import the model into the local Bento model store. The command is idempotent for
the same ONNX SHA-256:

```bash
uv run python serving/import_model.py
uv run bentoml models list
BENTOML_CONFIG=serving/bentoml-config.yaml uv run bentoml serve serving/service.py:svc --port 8000
```

`POST /predict` accepts a multipart `file` upload and returns the FastAPI fields
`breed`, `species`, `confidence`, `top_3`, `decision`, and `model_version`, plus
the existing correlation and latency fields. `GET /health` only returns healthy
after the predictor has loaded and includes `model_version` and `num_classes`.
Invalid bytes and uploads over `PETBREED_MAX_UPLOAD_BYTES` (10 MiB by default)
are rejected as 4xx errors. Four-channel images are converted to RGB.

Adaptive batching is enabled with `max_batch_size=8` and `max_latency_ms=50`.
Eight is small enough to cap memory for 224x224 CPU inference, and 50 ms keeps
the batching wait below the latency target while still coalescing concurrent
requests.

The service uses the model paths resolved by `cfg`: `PETBREED_ARTIFACTS_DIR`,
`PETBREED_ONNX_PATH`, and `PETBREED_MODEL_META_PATH` can override them. No model
path is hardcoded in the service.

## Locust Baseline

`loadtest/locustfile.py` reads only the `val` list from `data/split_index.json`.
It validates a pool of 736 validation images and cycles through 200 or more
distinct image IDs before repeating. Tasks are weighted 90% `POST /predict` and
10% `GET /health`, with `between(1, 3)` seconds of wait time. The same file is
used for FastAPI and BentoML by changing only `--host`.

Run one service at a time, with Locust on this same machine:

```bash
uv run locust -f loadtest/locustfile.py --host http://127.0.0.1:8000 --users 50 --spawn-rate 5 --run-time 3m --headless --html reports/locust_fastapi.html --csv reports/locust_fastapi
uv run locust -f loadtest/locustfile.py --host http://127.0.0.1:8000 --users 50 --spawn-rate 5 --run-time 3m --headless --html reports/locust_bentoml.html --csv reports/locust_bentoml
```

The intended fairness settings are one process/worker per service, the same
ONNX Runtime defaults (no explicit intra-op or inter-op settings in
`PetBreedPredictor`), and the same 12-core host. Locust is colocated with the
server, so these are local baselines rather than isolated production latency.
The required pre-run gate is:

```bash
nvidia-smi
awk '{print $1}' /proc/loadavg
nproc
```

Do not measure if another training process is present or the one-minute load
average is above 25% of `nproc` (3.0 on this host). Record `not run: machine
busy` instead.

## Batch Scoring

Batch scoring reads only trainval image IDs by concatenating the `train` and
`val` arrays from `data/split_index.json`; it never reads the test list. It uses
the shared predictor in chunks and writes one CSV row per image:

```bash
uv run python -m pet_breed_classification.batch --limit 1000
```

Output is `data/scoring/output/batch_predictions.csv`. The command prints image
count, wall-clock seconds, images/second, peak RSS, and row count.

## Nginx Canary

`docker/docker-compose.canary.yml` runs two copies of the existing local API
image, `model-stable` and `model-canary`, behind Nginx on `127.0.0.1:8080`.
The default image is the local v2 build `pet-breed:0.2.0`; override it with
`PETBREED_API_IMAGE`. The compose file never builds or pulls an API image. The
default Nginx image is pinned to `nginx:1.27.5-alpine@sha256:65645c7bb6a0661892a8b03b89d0743208a18dd2f3f17a54ef4b76fb8e2f2a10`;
override it with `NGINX_IMAGE`.

The initial weights are exact stable 95 / canary 5:

```nginx
server model-stable:8000 weight=95;
server model-canary:8000 weight=5;
```

The proxy adds `X-Serving-Backend` containing the selected upstream address;
map the address to `model-stable` or `model-canary` in the compose network when
counting requests.

Rollout stages are applied by editing the two upstream lines in
`serving/nginx/nginx.conf` and running `nginx -t` and `nginx -s reload` in the
`nginx` container. Rewrite the file in place (as `rollback.sh` does), because a
single-file bind mount does not see a file replaced by `sed -i`.

| Stage | `model-stable` line | `model-canary` line |
|---|---|---|
| 5% | `server model-stable:8000 weight=95;` | `server model-canary:8000 weight=5;` |
| 25% | `server model-stable:8000 weight=75;` | `server model-canary:8000 weight=25;` |
| 50% | `server model-stable:8000 weight=50;` | `server model-canary:8000 weight=50;` |
| 100% | `server model-stable:8000 down;` | `server model-canary:8000 weight=100;` |

`serving/rollback.sh` sets stable to `weight=100`, marks canary `down`, runs
`nginx -t`, and reloads nginx. The automatic latency-based rollback guard is
**not implemented**; rollback is a manual command.

## Recorded Results

All results below use model v2 (`model.onnx` sha256
`58165e216e4a5e8945d6763801c31eef1bfc9a2a5ae348b607bb1098f0a88f72`) on an
Intel Core i5-12500 (6 cores, 12 threads). Locust ran on the same host as the
service. See the top-level README for the summary table.
