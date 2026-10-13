# Results: model selection, calibration and test evaluation

MLflow runs and model selection, the shipped model and its calibration, the one-shot test evaluation and robustness. Back to the [README](../README.md).

## MLflow runs and model selection

Selection rule, declared before results: the highest served-path val macro-F1 among the
six runs below (ONNX Runtime CPU, after temperature scaling). Experiment
`pet-breed-classification` on the local tracking server. All runs: batch 64, 10 epochs,
AdamW, cosine schedule, AMP, seed 42, `git_dirty=false`.

| Run | Backbone | lr | top-1 | macro-F1 | ECE | Temperature | Coverage | Selective acc. |
|---|---|---|---|---|---|---|---|---|
| r50-repro | resnet50 | 1e-4 | 0.9429 | 0.9436 | 0.0231 | 0.9148 | 0.8478 | 0.9808 |
| **r50-lr-half** | resnet50 | 5e-5 | **0.9497** | **0.9499** | 0.0321 | 0.6785 | 0.8736 | 0.9813 |
| r18-a | resnet18 | 1e-4 | 0.9185 | 0.9185 | 0.0405 | 0.6461 | 0.7731 | 0.9807 |
| r18-lr-half | resnet18 | 5e-5 | 0.9076 | 0.9072 | 0.0333 | 0.5062 | 0.7704 | 0.9806 |
| mnv3s-a | mobilenet_v3_small | 1e-4 | 0.8003 | 0.7991 | 0.0228 | 0.6502 | 0.4851 | 0.9804 |
| mnv3s-lr-half | mobilenet_v3_small | 5e-5 | 0.7106 | 0.7079 | 0.0477 | 0.4454 | 0.3139 | 0.9827 |

The winning configuration was retrained once on a clean tree as `r50-lr-half-final`
(run_id `d05c2f2b9ca7481bb657acb5cc2b29f6`, commit `e1844b7`). Its metrics match
`r50-lr-half` exactly, and it is the shipped model. Three earlier attempts named
`r50-lr-half-final` are recorded as FAILED: they hung before epoch 1. The cause was
not proven; tracking now fails fast when the server is unreachable. A
`backfill-resnet50-existing-run` run records the pre-MLflow v1 model from files on
disk.

MLflow UI comparison (taken by the owner; it lists the 6 selection runs, the final
run and the retrain run):

![MLflow run comparison](../reports/mlflow_comparison.png)

## Shipped model (v2)

| Item | Value |
|---|---|
| MLflow run | `r50-lr-half-final`, run_id `d05c2f2b9ca7481bb657acb5cc2b29f6` (experiment `pet-breed-classification`) |
| Registry | `PetBreedClassifier` version 1, alias `production`. Version 2 (the drift-triggered retrain candidate) holds `challenger`; see [Retraining loop](../README.md#retraining-loop) |
| Backbone | ResNet-50, lr 5e-5, batch 64, 10 epochs, AdamW, cosine, AMP |
| `model.onnx` sha256 | `58165e216e4a5e8945d6763801c31eef1bfc9a2a5ae348b607bb1098f0a88f72` |
| Val top-1 / macro-F1 | 0.9497 / 0.9499 (served path, n=736) |
| Temperature | 0.678530 (ECE 0.0568 -> 0.0321) |
| Abstain threshold | 0.918166 (target selective accuracy 0.98; coverage 0.8736, selective accuracy 0.9813 on val) |

**Calibration** (`reports/calibration.json`, reliability diagram
`reports/calibration.png`): the temperature was fitted on val by minimizing NLL
(0.2207 -> 0.1803). The threshold is the lowest calibrated confidence whose selective
accuracy on val reaches 0.98. A 2-fold swap on val checks that this choice is not
overfitted: fit T and the threshold on one half (368 images), evaluate on the other.

| Fold | T | Threshold | Held-out coverage | Held-out selective accuracy |
|---|---|---|---|---|
| 0 | 0.6654 | 0.9099 | 0.8913 | 0.9756 |
| 1 | 0.6898 | 0.9282 | 0.8587 | 0.9873 |
| pooled | | | 0.8750 | 0.9814 |

The in-sample threshold gives coverage 0.8736 and selective accuracy 0.9813, and the
pooled held-out numbers are within 0.002 of them.

MLflow 3 replaced the deprecated registry stages with aliases. In this project the
alias `production` on `PetBreedClassifier` is what "promoted to Production" means, and
`challenger` marks a candidate that passed the retrain gate and waits for a human.
Naming: `model_version` `v2` in `model_meta.json` counts shipped artifacts (this is the
second one), and that artifact is registry version 1.
The previous shipped model (v1, ONNX sha256 `2695971d...d14a5d`) is kept locally in
`artifacts/previous/`.

## Final test evaluation (one shot, model frozen)

`uv run python -m pet_breed_classification.evaluate_test` ran once on the official
test split through the served path: ONNX Runtime on CPU, the shipped temperature
0.678530 and the shipped threshold 0.918166. Nothing was changed after it ran. The
module refuses to overwrite `reports/final_test_evaluation.json`.

| Metric | Val (n=736) | Test (n=3,669) | Why they differ |
|---|---|---|---|
| top-1 | 0.9497 | 0.9182 | Val chose the winner among 6 runs, so its number is optimistically biased; test is unseen and 5x larger. |
| macro-F1 | 0.9499 | 0.9173 | Same selection bias as top-1. |
| ECE (calibrated) | 0.0321 | 0.0352 | The temperature was fitted on val, so it fits val slightly better. |
| coverage | 0.8736 | 0.8449 | The model is less accurate on test, so fewer predictions clear the val-chosen threshold. |
| selective accuracy | 0.9813 | 0.9739 | The threshold was set to reach 0.98 on val; on test it reaches 0.974, which misses the target by 0.6 points. |

The 10 most-confused pairs (true -> predicted, count) on test:
american_pit_bull_terrier -> staffordshire_bull_terrier 30;
Egyptian_Mau -> Bengal 14; Ragdoll -> Birman 14;
staffordshire_bull_terrier -> american_pit_bull_terrier 11; Abyssinian -> Bengal 10;
american_pit_bull_terrier -> american_bulldog 10; Russian_Blue -> Bombay 10;
Maine_Coon -> Bengal 9; staffordshire_bull_terrier -> american_bulldog 9;
Birman -> Ragdoll 8. These are look-alike breeds, and they are why the service
abstains rather than guessing. Confusion matrix: `reports/confusion_matrix.png`.

### Robustness (corrupted test copies)

For each corruption x severity, a seeded sample (seed 42) of n=1,000 of the 3,669
corrupted test images went through the same served path. Full table, including ECE
and macro-F1: `reports/robustness.csv`.

| Corruption | Sev 1 top-1 / cov / sel-acc | Sev 2 top-1 / cov / sel-acc | Sev 3 top-1 / cov / sel-acc |
|---|---|---|---|
| brightness_down | 0.910 / 0.825 / 0.973 | 0.901 / 0.812 / 0.973 | 0.903 / 0.766 / 0.980 |
| brightness_up | 0.911 / 0.824 / 0.967 | 0.906 / 0.799 / 0.972 | 0.879 / 0.751 / 0.968 |
| downscale_upscale | 0.811 / 0.687 / 0.913 | 0.698 / 0.553 / 0.890 | 0.560 / 0.365 / 0.827 |
| gaussian_blur | 0.880 / 0.769 / 0.969 | 0.788 / 0.606 / 0.932 | 0.516 / 0.203 / 0.897 |
| jpeg_compression | 0.856 / 0.765 / 0.948 | 0.821 / 0.727 / 0.930 | 0.680 / 0.536 / 0.860 |
| motion_blur | 0.866 / 0.752 / 0.967 | 0.794 / 0.588 / 0.947 | 0.625 / 0.300 / 0.920 |

Brightness shifts cost at most 4 points of top-1. Downscaling, blur and heavy JPEG
cost up to 40.2 points (Gaussian blur severity 3: 0.516 vs 0.918 clean). Abstention absorbs part of the damage, since coverage falls
faster than selective accuracy, but for downscale, blur and JPEG selective accuracy still drops to 0.83-0.92 at
severity 3, so the threshold does not hold its 0.98 promise under these shifts.
