# Project Spec — Track C (Computer Vision)

> Extracted verbatim from *The MLOps Practitioner — Mini Projects & Final Project Handbook*
> (Project 3 section, the shared 10-point rubric, and the timeline).
> The handbook describes Oxford-IIIT Pet; this project uses PlantVillage instead
> (an approved alternative). See `CLAUDE.md` for how the requirements map to this dataset.

---

## Project 3 — Computer Vision · Pet Breed Classification

User-submitted pet photos · PyTorch · ResNet / MobileNet · BentoML · TensorRT · Triton · ONNX Runtime · OpenVINO

### Three tracks, not two

This project was added after the first print of the handbook. If an earlier copy of the course guide says "choose 1 of 2 tracks", "Deep Learning or LLM" or "two projects, one standard", read those as three. Nothing else changes: the same 10-point rubric, the same deadline, the same certificate, the same referral route. The rubric needed no new rows for this track — Track C is a deep-learning project, so rows 07 and 08 apply to it exactly as written.

**The business problem.** A pet-care marketplace operating across Egypt, Saudi Arabia and the UAE lets users upload a photo of their animal when they create a profile. That one photo drives everything downstream: which food and accessory listings are recommended, which veterinary content is surfaced, and which adoption listings get matched to which household. Users type the breed wrong, in Arabic, in English, or not at all. The platform needs a service that takes a phone photo and returns a breed with a calibrated confidence — and that says "I am not sure" instead of guessing, because a confident wrong breed sends a large-breed food recommendation to a kitten owner.

**Why this project for this course.** This is the track with the smallest machine-learning surface and the largest engineering surface. Twenty lines of torchvision transfer learning gets you a working model in an afternoon, which means every remaining hour of the six weeks lands on the system. It is also the only track where the entire Module 4 toolbox applies to one model and produces numbers you can actually compare: pruning, post-training quantization, quantization-aware training, distillation, TensorRT on GPU, ONNX Runtime and OpenVINO on CPU, and TFLite at the edge. Image drift is real and measurable here — phone cameras, indoor lighting, motion blur, and the seasonal shift when everyone photographs their pets outdoors in winter. And a fine-grained CNN is the one model in this course where a distilled student and an INT8 engine give visibly different accuracy-latency trade-offs rather than a rounding error.

Best if you are newer to deep learning and want the full optimization story. No Arabic NLP, no tokenizers, no RAG evaluation theory — a pretrained backbone, a replaced classification head, and then six weeks of production engineering. A GPU helps for Session 4; Colab's free T4 is acceptable and is what most students on this track use.

### Your input data — Project 3

The dataset is Oxford-IIIT Pet — 7,349 images, 37 cat and dog breeds, roughly 200 images per class, about 800 MB. It ships with `torchvision.datasets.OxfordIIITPet(download=True)`, so you are training within ten minutes of starting.

**Why 37 breeds and not cats-vs-dogs.** Binary cats-vs-dogs reaches 99% accuracy with a frozen backbone and one epoch. That sounds like a gift and it is a trap: with no headroom, distillation shows no loss, quantization shows no loss, pruning shows no loss, and your entire Module 4 benchmark table is five identical numbers. You will have built the pipeline and measured nothing. Fine-grained breed classification sits around 88–93% with a ResNet-50, which leaves exactly enough room for every optimization in the course to move the number in a direction you have to explain.

**This dataset is not your production data.** It is studio-quality, centred, well-lit, and balanced. Real uploads are none of those things. Your first deliverable on this track is a script that turns it into a realistic, versioned, drifting dataset — committed to the repo and tracked with DVC.

Approved alternatives if you prefer another domain, cleared with the instructor in Session 1: Stanford Dogs (120 classes, harder), PlantVillage crop-disease leaves (agriculture, MENA-relevant), or a recyclable-waste classifier. The rules below apply unchanged. You may not use CIFAR-10 or the NYC TLC dataset — those are mini-project material, and your reviewer needs to see what you built rather than what the mini projects handed you.

---

## Step 0 for Project 3 — Build the dataset and the corruption suite

1. **Fix the split before anything else.** Oxford-IIIT Pet ships a trainval / test split. Use it, then carve a validation fold out of trainval with a fixed seed and commit the index files. Never touch test until Session 4. A test set you have looked at forty times is a validation set with a misleading name, and every number in your report inherits the lie.

2. **Build a corruption suite — this is your drift ground truth.** Write `data/corruptions.py` applying, at three severity levels each: Gaussian blur (out-of-focus phone), brightness shift up and down (indoor evening, direct sun), JPEG compression at quality 30 (messaging-app re-encode), downscale-then-upscale to 96×96 (old handset), and motion blur. Generate a corrupted copy of the test set per severity. You now own a labelled drift scenario: you know exactly which shift you applied and when, so in Module 5 you can score your detectors against the truth instead of guessing.

3. **Target manifest schema.** One record per image, committed as JSON and DVC-tracked:

   ```json
   {
     "image_id": "Abyssinian_100",
     "path": "data/images/Abyssinian_100.jpg",
     "breed": "Abyssinian",
     "species": "cat",
     "class_index": 0,
     "split": "train",
     "corruption": null,
     "severity": 0,
     "width": 394,
     "height": 500
   }
   ```

4. **What will actually break** — budget an evening, and write what you hit in your report:
   - **Train/serve preprocessing skew.** Your training transform resizes, centre-crops and normalizes with ImageNet mean and std. If you reimplement that by hand in the API and get the channel order or the normalization constants wrong, the model still returns a confident answer — a wrong one, with no error anywhere. Export the exact eval transform alongside the weights and load it from one place. Then write the test that proves it: the same image through the training path and through the API path must produce logits agreeing to 1e-4.
   - **Class imbalance is mild here, ordering is not.** Class indices must come from a committed, sorted label map — never from directory-listing order, which differs between your laptop and the CI runner. A shuffled label map is the bug that makes your API return "Bengal" for every Beagle and passes every test you wrote.
   - **Four of the 37 breeds are genuinely confusable** even for humans (the three spaniels; British Shorthair against Russian Blue). Do not chase them with more epochs. Put the confusion matrix in your report and use it to justify your abstention threshold instead — that is the production answer.
   - **Confidence out of a softmax is not calibrated.** A model that is 96% confident is right well below 96% of the time, and your "I am not sure" behaviour depends on that number meaning something. Apply temperature scaling on the validation fold, plot the reliability diagram before and after, and set your abstention threshold on the calibrated score. Log the temperature as an MLflow parameter and ship it in the artifact.
   - **Some images are CMYK, some are greyscale, one is a PNG with an alpha channel.** `Image.open(...).convert("RGB")` at both training and serving, and a test that feeds a 4-channel PNG to `/predict` and expects 200, not 500.
   - **Augmentation belongs to training only.** Random crops and flips in the eval or serving path produce different answers for the same image on two consecutive calls, and you will spend a day blaming the GPU.

5. **Validate before you train.** Write tests asserting: every manifest path exists and opens, no `image_id` appears in two splits, the label map has exactly 37 entries and matches the committed reference, every class has at least 50 training images, and no image is under 32×32. A silent manifest bug becomes a wrong prediction three steps later, and by then you will blame the model.

6. **Make it reproducible.** The manifest builder and the corruption generator are `dvc.yaml` stages. Images, manifest and corrupted sets are all DVC-tracked, so `dvc repro` rebuilds everything from the source archive and your reviewer regenerates your exact splits.

7. **Abstain, do not guess.** `/predict` returns the top-3 breeds with calibrated probabilities and a `decision` field that is `"uncertain"` when the top calibrated score falls below your threshold. Report coverage and selective accuracy — what fraction you answer, and how accurate you are on that fraction — not just top-1. That pair of numbers is what a product owner actually buys.

### Two things this dataset gives you for free

**A drift scenario with a known answer.** You generated the corruptions in Step 0, so when your Module 5 detectors fire you can check them against ground truth: which detector caught which corruption, at which severity, with how many false alarms on the clean batches. Nobody on Tracks A or B can do that — their drift is real but unlabelled. A detector scorecard in your report is one of the strongest things a reviewer can read.

**The complete optimization matrix on one model.** A 25M-parameter ResNet-50 and a 2.5M-parameter MobileNetV3-Small solving the same task, at the same input size, on data you already have. Every technique in Module 4 — pruning, PTQ, QAT, distillation, TensorRT, ONNX Runtime, OpenVINO, TFLite — applies to this pair without any adaptation, and each one moves accuracy and latency by an amount you can defend. No other track in this course gets a clean sweep of all eight.

---

## Output — Project 3 · Computer Vision

| Area | Deliverable |
|---|---|
| Package + API | PyTorch model class wrapping backbone + exported eval transform. `/predict`: multipart image → `{"breed":"Bengal","species":"cat","confidence":0.91,"top_3":[...],"decision":"confident","model_version":"v3"}` |
| Track + Version | MLflow: ≥ 5 runs across ≥ 3 backbones (ResNet-50, ResNet-18, MobileNetV3-Small) with top-1, macro-F1, ECE, coverage and selective accuracy. DVC: images, manifest, corrupted sets. GitHub Actions CI with a top-1 quality gate. |
| Serve | BentoML with a batchable Runner. TensorRT FP16 engine on Triton (GPU). ONNX Runtime + OpenVINO path (CPU, mandatory). Locust: p95 before vs after. |
| Optimize | Five techniques benchmarked: structured pruning, INT8 PTQ, QAT, ResNet-50 → MobileNetV3 distillation, TensorRT FP16. One table: top-1 × p95 × size × hardware. |
| Monitor | Embedding drift (MMD or a domain classifier on penultimate-layer features) plus a cheap pixel-statistics detector. Confidence-distribution drift. Evidently report per batch. Prometheus + Grafana. Alert wired to the retraining DAG. |

## Where Track C is deliberately harder

The model is easier to train than AraBERT, so four requirements are stricter than on Tracks A and B. This is what keeps "no track is easier than another" honest — read it before you choose this track thinking it is the light one.

1. **The full optimization matrix, not a subset.** Track A benchmarks two techniques. You benchmark five, in one table, with a hardware column, and you explain every row that moved the wrong way.
2. **Calibration is mandatory.** Temperature scaling, a reliability diagram before and after, and an abstention threshold defended with coverage and selective accuracy. Raw softmax confidence in your API loses points on rubric row 02.
3. **Two drift detectors, scored against ground truth.** One embedding-based, one cheap statistical, plus the detector scorecard across your corruption severities.
4. **The retraining loop must actually close and be demonstrated.** Drift alert → Airflow DAG → fine-tune on the corrupted distribution → quality gate against the Production model → promote or reject. Show the triggered run in your README.

---

## Completion checklist — Code, packaging, tracking, versioning

- [ ] `src/` layout with `pyproject.toml` — `pip install -e .` works
- [ ] Manifest JSON and corruption suite generated by committed scripts; corruptions and severities documented in `data/README.md`
- [ ] Label map committed and sorted; a test asserts it has 37 entries and matches the reference
- [ ] Model loaded in a class with type hints — weights and eval transform loaded once at startup, not per request
- [ ] Eval transform exported with the weights; a test asserts training-path and API-path logits agree to 1e-4
- [ ] FastAPI `/predict`: multipart image → `{breed, species, confidence, top_3, decision, model_version}`
- [ ] Pydantic and file validation reject a non-image, an oversized upload and a 4-channel PNG correctly — 422 or 200, never 500, tested with curl
- [ ] Temperature scaling fitted on validation; reliability diagram before and after in `/reports/calibration.png`
- [ ] Abstention threshold set from calibrated scores; coverage and selective accuracy in README
- [ ] `/health` returns `{status: healthy, model_version, num_classes}` — used by the Docker healthcheck
- [ ] `docker build -t pet-breed .` passes; `docker compose up` starts the service on port 8000
- [ ] README: exactly 3 commands to run on any machine
- [ ] MLflow tracking server running — UI accessible
- [ ] ≥ 5 runs across ≥ 3 backbones: `backbone`, `lr`, `batch_size`, `top1`, `f1_macro`, `ece`, `temperature`
- [ ] MLflow comparison screenshot in `/reports/mlflow_comparison.png`
- [ ] Best model registered as `PetBreedClassifier` → promoted to Production
- [ ] Images, manifest and corrupted sets tracked with DVC — `dvc status` clean
- [ ] `dvc repro` reproduces training with the same metrics (± 0.005 top-1)
- [ ] GitHub Actions: lint → test → docker build → push on main
- [ ] CI fails if top-1 drops below baseline — documented in README

## Completion checklist — Serving, monitoring, optimization

- [ ] BentoML service with `batchable=True` Runner — `bentoml models list` shows the model
- [ ] `bentoml serve` runs `/predict` at localhost:8000
- [ ] Locust report before optimization in `/reports/locust_fastapi.html`
- [ ] ONNX export with a test asserting PyTorch and ONNX logits agree to 1e-4 on 200 images
- [ ] ONNX Runtime and OpenVINO CPU path live — p95 for both in README
- [ ] TensorRT FP16 engine built and served on Triton with dynamic batching — Colab T4 acceptable, hardware documented
- [ ] Locust after TensorRT in `/reports/locust_trt.html` — p95 in README
- [ ] Batch scoring: ≥ 1,000 images, output to `/data/scoring/output/`
- [ ] nginx canary config: weight 5/95 — rollout stages in README
- [ ] Embedding drift: MMD or a domain classifier on penultimate-layer features — score logged per batch
- [ ] Cheap statistical detector on pixel brightness, contrast and sharpness — logged per batch
- [ ] Confidence-distribution drift tracked separately from input drift
- [ ] Detector scorecard: which detector caught which corruption, at which severity, with false alarms on clean batches
- [ ] Evidently report on the latest batch — saved in `/reports/`
- [ ] Drift check triggered after every batch scoring run
- [ ] Prometheus exposes drift scores and p95 at `/metrics`
- [ ] Grafana: ≥ 3 panels (p95, drift score, confidence distribution) — screenshot in README
- [ ] Alert: drift above threshold → Airflow retraining DAG triggered — triggered run shown in README
- [ ] Quality gate on retraining: promoted only if top-1 ≥ Production − 0.01
- [ ] Structured channel pruning applied — sparsity level and accuracy delta documented
- [ ] INT8 post-training quantization with a calibration set — `model_int8.onnx` committed
- [ ] Quantization-aware training run — accuracy compared against PTQ
- [ ] ResNet-50 distilled into MobileNetV3-Small — student weights committed
- [ ] Benchmark table: baseline vs pruned vs PTQ vs QAT vs distilled vs TensorRT FP16 (top-1, p95, size, hardware)
- [ ] Every row of the table that got worse is explained in one sentence — see the honesty clause
- [ ] Final architecture diagram in README covering all 5 sessions

## Failure modes — what costs Track C students the most hours

- **Chasing accuracy instead of building the system.** A backbone that is 1.5% better moves zero rubric points. Freeze your model by the end of Week 2 and spend the rest of the course on the eighteen boxes above that have nothing to do with training.
- **Reimplementing preprocessing in the API.** The single most common way this track fails silently. Export the transform, load it from one place, and write the agreement test.
- **Benchmarking a Colab T4 number against a laptop CPU number in the same column.** The hardware column exists for a reason, and a reviewer will spot it.
- **Quantizing before you have a working benchmark harness.** An unmeasured optimization is a rumour. Module 4 says this and this track is where it bites hardest, because five techniques with no harness is five unverifiable claims.
- **Assuming "no Arabic, no RAG" means less work.** Read the four stricter requirements again. The hours move from the model to the engineering; they do not disappear.

## A note on the honesty clause

It applies here with extra force, because five optimization techniques give you five chances to report something that did not happen. "INT8 PTQ cost 2.1% top-1 and QAT recovered 1.6% of it for four extra GPU hours, so I shipped FP16 TensorRT" is a better report than a table where every technique is free. Negative results, measured properly and reported clearly, are production engineering. Fabricated numbers are the one thing that fails a submission outright.

---

## The 10-point rubric (shared by all tracks)

| # | Area | Module | What you build | Pass condition |
|---|---|---|---|---|
| 01 | Code & packaging | M1 | `pyproject.toml`, `src/` layout, OOP model class, type hints | `pip install -e .` works. Model loads without errors. |
| 02 | API endpoint | M1 | `/predict` with Pydantic, `/health`, async handlers | `curl /predict` returns correct output. |
| 03 | Docker | M1 | Dockerfile, `docker-compose.yml`, 3-command README | `docker compose up && curl localhost:8000/predict` works on the reviewer's machine. |
| 04 | MLflow tracking | M2 | ≥ 5 runs, params + metrics + artifacts, best model in Registry | MLflow UI shows the comparison. Model promoted to Production. |
| 05 | DVC data versioning | M2 | Dataset tracked, `dvc.yaml` pipeline, `dvc repro` works | `git checkout` + `dvc pull` + `dvc repro` reproduces the result. |
| 06 | GitHub Actions CI/CD | M2 | Lint → test → build → push on every PR. Quality gate. | Merge a PR. Actions green. Docker image pushed to the registry. |
| 07 | Production serving | M3 | BentoML (DL) or vLLM + BentoML (LLM). Locust tested. | p95 latency documented. Locust report in `/reports/`. |
| 08 | Monitoring | M5 | ≥ 1 drift metric (DL) or RAGAS faithfulness (LLM). Grafana panel. | Screenshot in README. Alert threshold set. |
| 09 | Peer review | Post-course | Written review ≥ 300 words, all 5 areas | Submitted on time. Covers setup, code, strength, 2 improvements, extension. |
| 10 | README & architecture | M4/M5 | 3-command setup, architecture diagram, session changelog | The reviewer runs it without asking you anything. |

## Timeline and deadlines

| Checkpoint | What must exist |
|---|---|
| End of Session 1 | GitHub repo created. `/predict` running in Docker. 3-command README. |
| End of Session 2 | ≥ 5 MLflow runs logged. DVC pipeline clean. GitHub Actions green. |
| End of Session 3 | BentoML or vLLM service running. Locust report submitted. |
| End of Session 4 | Optimization complete. Benchmark table. |
| End of Session 5 | Drift detection or RAGAS running. Grafana dashboard live. Start final polish. |
| 2 weeks after Session 5 | Submit your GitHub repo AND your peer review together. No separate deadlines. |
