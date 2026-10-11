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
