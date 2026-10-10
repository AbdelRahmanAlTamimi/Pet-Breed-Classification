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
uv run bentoml serve serving/service.py:PetBreedService --port 8000
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
The default image is `pet-breed:latest`; override it with
`PETBREED_API_IMAGE`. The compose file never builds or pulls an API image. The
default Nginx image is `nginx:1.27-alpine`; use a locally available image with
`NGINX_IMAGE` rather than pulling one.

The initial weights are exact stable 95 / canary 5:

```nginx
server model-stable:8000 weight=95;
server model-canary:8000 weight=5;
```

The proxy adds `X-Serving-Backend` containing the selected upstream address;
map the address to `model-stable` or `model-canary` in the compose network when
counting requests.

Rollout stages are applied by editing the two upstream lines and reloading
Nginx:

1. 5% canary: stable `weight=95`, canary `weight=5`.
2. 25% canary: stable `weight=75`, canary `weight=25`.
3. 50% canary: stable `weight=50`, canary `weight=50`.
4. 100% canary: stable `weight=0` is represented by removing or disabling the stable line, and canary remains active.

`serving/rollback.sh` edits stable to `weight=100`, marks canary `down`, runs
`nginx -t`, and reloads the running Nginx container. The automatic
latency-based rollback guard was **not implemented**.

The local API image may be older than the current calibrated artifact. If so,
the canary measurement tests routing only and does not claim current-model
calibration parity.
