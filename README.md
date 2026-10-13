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

## Dataset and license

Oxford-IIIT Pet Dataset: O. M. Parkhi, A. Vedaldi, A. Zisserman and C. V. Jawahar,
"Cats and Dogs", IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
2012. https://www.robots.ox.ac.uk/~vgg/data/pets/. The dataset page (checked on
2026-10-13) states: "The dataset is available to download for commercial/research
purposes under a Creative Commons Attribution-ShareAlike 4.0 International License.
The copyright remains with the original owners of the images."
