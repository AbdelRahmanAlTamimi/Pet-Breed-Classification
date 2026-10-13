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
