"""Retraining loop closed by the batch drift check (no Airflow; a guarded function call).

When a drift check raises any alarm, ``on_drift_check`` asks the storm guard for
permission: a 24 h cooldown, at most one retrain per UTC day, and at least 1,000 images
scored since the last retrain. The guard's state lives in the git-ignored
data/monitoring/retrain_state.json.

A permitted retrain fine-tunes the Production checkpoint for 2 epochs on TRAIN images with
on-the-fly corruptions from the existing corruption functions (never the corrupted test
sets), with MLflow tracking on. It calibrates on val, evaluates on val through the served
ONNX path, registers the run, and calls ``registry.promote_if_better`` with the
``challenger`` alias: the candidate becomes the challenger only if its top-1 >= Production
top-1 - 0.01. The gate never moves ``production``; that needs a human approving the
challenger. The candidate is never copied into artifacts/ and no image is rebuilt.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from pet_breed_classification import calibrate, registry, tracking, train
from pet_breed_classification.artifacts import create_run_artifacts
from pet_breed_classification.config import cfg
from pet_breed_classification.data import PetBreedDataset, _load_classes, load_records
from pet_breed_classification.data.corruptions import CORRUPTIONS, SEVERITIES
from pet_breed_classification.export import export, validate_onnx_parity
from pet_breed_classification.model import load_checkpoint, save_checkpoint
from pet_breed_classification.transforms import (
    build_eval_transform,
    build_train_transform,
)

from .reference import CACHE_DIR

logger = logging.getLogger(__name__)

STATE_PATH = CACHE_DIR / "retrain_state.json"
COOLDOWN = timedelta(hours=24)
DAILY_CAP = 1
MIN_SCORED_IMAGES = 1000
FINE_TUNE_EPOCHS = 2
FINE_TUNE_LR = 1e-5
CORRUPTION_PROBABILITY = 0.5
RETRAIN_ARTIFACTS_DIR = cfg.ARTIFACTS_DIR / "retrain"
PROMOTION_MARGIN = 0.01


def load_state(path: Path = STATE_PATH) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"last_retrain_at": None, "retrains_by_day": {}, "scored_since_last_retrain": 0}


def save_state(state: dict[str, Any], path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def blocked_reason(state: dict[str, Any], now: datetime) -> str | None:
    """Return why the storm guard blocks a retrain now, or None when it is allowed."""
    if state["scored_since_last_retrain"] < MIN_SCORED_IMAGES:
        return (
            f"only {state['scored_since_last_retrain']} images scored since the last retrain "
            f"(< {MIN_SCORED_IMAGES})"
        )
    last = state["last_retrain_at"]
    if last is not None and now - datetime.fromisoformat(last) < COOLDOWN:
        return f"cooldown: last retrain at {last} is less than 24 h ago"
    if state["retrains_by_day"].get(now.date().isoformat(), 0) >= DAILY_CAP:
        return f"daily cap of {DAILY_CAP} retrain reached"
    return None


def on_drift_check(
    scored_images: int,
    any_alarm: bool,
    *,
    now: datetime | None = None,
    retrain: Callable[[], dict[str, Any]] | None = None,
    state_path: Path = STATE_PATH,
) -> dict[str, Any]:
    """Count scored images and, on an alarm the guard permits, run one retrain."""
    now = now or datetime.now(UTC)
    state = load_state(state_path)
    state["scored_since_last_retrain"] += scored_images
    decision: dict[str, Any] = {"triggered": False, "checked_at": now.isoformat()}
    if not any_alarm:
        decision["reason"] = "no drift alarm"
    elif (reason := blocked_reason(state, now)) is not None:
        decision["reason"] = f"storm guard: {reason}"
    else:
        # Record the attempt before training, so a crash cannot trigger a retrain storm.
        state["last_retrain_at"] = now.isoformat()
        day = now.date().isoformat()
        state["retrains_by_day"][day] = state["retrains_by_day"].get(day, 0) + 1
        state["scored_since_last_retrain"] = 0
        save_state(state, state_path)
        decision.update({"triggered": True, "reason": "drift alarm", **(retrain or fine_tune)()})
    save_state(state, state_path)
    logger.info("Retrain decision: %s", decision)
    return decision


class RandomCorruption:
    """Apply one random corruption at a random severity with a fixed probability."""

    def __init__(self, probability: float = CORRUPTION_PROBABILITY) -> None:
        self.probability = probability
        self.names = sorted(CORRUPTIONS)

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        corruption = CORRUPTIONS[random.choice(self.names)]
        return corruption(image.convert("RGB"), random.choice(SEVERITIES))


def _production_matches_shipped_checkpoint() -> str:
    """Return the Production version after checking it is the shipped model."""
    client = registry._client()
    production = client.get_model_version_by_alias(registry.MODEL_NAME, registry.PRODUCTION_ALIAS)
    shipped = json.loads(cfg.MODEL_META_PATH.read_text(encoding="utf-8"))["onnx_sha256"]
    if production.tags.get("onnx_sha256") != shipped:
        raise RuntimeError(
            "Production in the registry is not the shipped model; refusing to fine-tune "
            f"{cfg.MODEL_PATH} as if it were Production"
        )
    return str(production.version)


def fine_tune() -> dict[str, Any]:
    """Fine-tune Production on corrupted train images, track, register and gate it."""
    production_version = _production_matches_shipped_checkpoint()
    train.seed_everything(cfg.SEED)
    device = train.DEVICE
    model, _ = load_checkpoint(cfg.MODEL_PATH, device)
    train_transform = transforms.Compose([RandomCorruption(), build_train_transform()])
    loader_options = {
        "batch_size": cfg.BATCH_SIZE,
        "num_workers": min(cfg.NUM_WORKERS or 1, 8),
        "pin_memory": device.type == "cuda",
        "worker_init_fn": train.seed_worker,
    }
    generator = torch.Generator()
    generator.manual_seed(cfg.SEED)
    train_loader = DataLoader(
        PetBreedDataset(load_records("train"), train_transform),
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    val_records = load_records("val")
    val_loader = DataLoader(PetBreedDataset(val_records, build_eval_transform()), **loader_options)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FINE_TUNE_LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINE_TUNE_EPOCHS)
    train._SCALER = torch.amp.GradScaler("cuda", enabled=train.USE_AMP)

    run_name = f"retrain-drift-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    run_artifacts = create_run_artifacts(run_name, RETRAIN_ARTIFACTS_DIR)
    classes = _load_classes()
    with tracking.tracking_run(
        run_name,
        enabled=True,
        tags={"trigger": "drift_alarm", "parent_production_version": production_version},
    ) as tracker:
        tracker.log_params(
            {
                "backbone": "resnet50",
                "lr": FINE_TUNE_LR,
                "batch_size": cfg.BATCH_SIZE,
                "epochs": FINE_TUNE_EPOCHS,
                "seed": cfg.SEED,
                "fine_tune_from": f"{registry.MODEL_NAME}@{registry.PRODUCTION_ALIAS} (v{production_version})",
                "train_corruption_probability": CORRUPTION_PROBABILITY,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
            }
        )
        result = train.train_epochs(
            model, train_loader, val_loader, nn.CrossEntropyLoss(), optimizer, scheduler, FINE_TUNE_EPOCHS
        )
        model.load_state_dict(result.best_state_dict)
        save_checkpoint(model, run_artifacts.checkpoint, classes, cfg.SEED, result.best_epoch, result.best_metrics)
        run_artifacts.history.write_text(json.dumps(result.history, indent=2) + "\n", encoding="utf-8")
        export(run_artifacts.checkpoint, run_artifacts.onnx, run_artifacts.metadata)
        meta = json.loads(run_artifacts.metadata.read_text(encoding="utf-8"))
        meta["model_version"] = f"candidate-{run_name}"
        run_artifacts.metadata.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        parity = validate_onnx_parity(run_artifacts.checkpoint, run_artifacts.onnx, val_records)
        calibration = calibrate.fit_artifacts(
            run_artifacts.directory, cfg.TARGET_SELECTIVE_ACCURACY, seed=cfg.SEED
        )
        train._log_calibration_metrics(tracker, calibration)
        tracker.log_metrics({"max_logit_parity_diff": parity})
        tracking.log_history(tracker, result.history)
        for path in (
            run_artifacts.onnx,
            run_artifacts.metadata,
            calibration.figure_path,
            calibration.report_path,
            run_artifacts.history,
        ):
            tracker.log_artifact(path)
        run_id = tracker.run_id

    if run_id is None:
        raise RuntimeError("The retraining run did not start an MLflow run")
    version = registry.register_run(run_id)
    promoted = registry.promote_if_better(
        run_id, metric="top1", margin=PROMOTION_MARGIN, alias=registry.CHALLENGER_ALIAS
    )
    client = registry._client()
    production_top1 = float(
        client.get_run(client.get_model_version(registry.MODEL_NAME, production_version).run_id)
        .data.metrics["top1"]
    )
    return {
        "run_id": run_id,
        "run_name": run_name,
        "registered_version": str(version.version),
        "candidate_val_top1": calibration.after["top1"],
        "production_version": production_version,
        "production_val_top1": production_top1,
        "promoted_to_challenger": promoted,
        "rule": (
            f"alias {registry.CHALLENGER_ALIAS} only if candidate top1 >= production top1 "
            f"- {PROMOTION_MARGIN}; production needs human approval"
        ),
    }
