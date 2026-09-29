# AGENTS.md

The full requirements are in `docs/project_spec.md` — read it before starting any task.

This project follows the spec's dataset (**Oxford-IIIT Pet**). The decisions below are final;
where they are more specific than the spec, they win.

## Project decisions


- **Classes:** 37 breeds. `breed` and `species` (`cat` / `dog`) come from the annotation dir (`./data/raw/annotations`)
- **Source of truth for the image set:** `trainval.txt` and `test.txt`. Images in `images/`
  that are not listed there are ignored; non-image files there are ignored.
- **Split:**
  - `test` = official `test.txt`, used as-is (~3,669 images).
  - `train` / `val` = official `trainval.txt` split 80/20, **stratified by breed**, seed `42`
    (~2,944 / ~736).
  - Split index files are committed and DVC-tracked; the seed lives in `cfg`.
- **Label map:** committed, sorted alphabetically by breed, exactly 37 entries.
  Class indices come only from this file — never from directory-listing order.
- **Names:** registry model `PetBreedClassifier`; Docker image `pet-breed`
  (`docker build -t pet-breed .`).

## Scope

- **Optimization is optional.** Everything in the spec's optimization section — pruning,
  PTQ, QAT, distillation, TensorRT / Triton, the optimization benchmark table, and the
  "Locust after TensorRT" item — is out of scope unless the user explicitly asks for it.
  Do not start it, scaffold it, or add dependencies for it.
- Everything else in the spec is required: packaging, API, calibration and abstention,
  Docker, MLflow, DVC, CI with a quality gate, BentoML serving with a Locust baseline,
  batch scoring, nginx canary, drift detection, monitoring, and the retraining loop.

## Rules

- KISS and YAGNI: simplest working solution. No abstractions, options or files that the
  current task doesn't need.
- All code, comments and names in English.
- Package manager is `uv` only — never pip. `uv add` / `uv add --dev`.
- Package lives in `src/pet_breed_classification/`. Run modules with
  `uv run python -m pet_breed_classification.<module>`.
- Shared paths and params come from `cfg` in `config.py`.
- Paths stored in JSON files are relative to the project root, POSIX form.
- Never look at test-set metrics before the final evaluation (see spec, Step 0.1).
  Model selection, calibration and thresholds use `val` only.
- Report real numbers only (see the honesty clause in the spec).
- Don't create git commits; the user reviews and commits.