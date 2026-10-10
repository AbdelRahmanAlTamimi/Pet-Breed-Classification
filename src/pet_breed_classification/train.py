"""Reproducible transfer-learning training entry point."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from . import calibrate, tracking
from .artifacts import create_run_artifacts
from .config import cfg
from .data import PetBreedDataset, _load_classes, load_records
from .export import export, validate_onnx_parity
from .logging_conf import setup_logging
from .model import (
    DEFAULT_BACKBONE,
    SUPPORTED_BACKBONES,
    PetBreedClassifier,
    save_checkpoint,
)
from .transforms import build_eval_transform, build_train_transform

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
_SCALER: torch.amp.GradScaler | None = None


@dataclass(frozen=True)
class TrainingResult:
    """Metrics and best weights produced by the epoch loop."""

    history: list[dict[str, float | int]]
    best_epoch: int
    best_metrics: dict[str, float]
    best_train_metrics: dict[str, float]
    best_state_dict: dict[str, torch.Tensor]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch, matching the notebook."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy in a DataLoader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run one training or evaluation epoch and return its metrics."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    predictions: list[int] = []
    targets_seen: list[int] = []

    context = torch.enable_grad() if training else torch.inference_mode()
    scaler = _SCALER
    with context:
        for batch_index, (images, targets) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                logits = model(images)
                loss = criterion(logits, targets)

            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = targets.size(0)
            batch_predictions = logits.argmax(dim=1)
            total_loss += loss.item() * batch_size
            total_correct += (batch_predictions == targets).sum().item()
            total_examples += batch_size
            predictions.extend(batch_predictions.detach().cpu().tolist())
            targets_seen.extend(targets.detach().cpu().tolist())

    if total_examples == 0:
        raise ValueError("epoch did not process any examples")
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "macro_f1": f1_score(
            targets_seen, predictions, average="macro", zero_division=0
        ),
    }


def train_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epochs: int,
    *,
    max_train_batches: int | None = None,
) -> TrainingResult:
    """Run epochs and record the optimizer LR before each scheduler step."""
    history: list[dict[str, float | int]] = []
    best_val_accuracy = -1.0
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    best_train_metrics: dict[str, float] = {}
    best_state_dict: dict[str, torch.Tensor] = {}

    for epoch in range(1, epochs + 1):
        started_at = time.perf_counter()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            max_batches=max_train_batches,
        )
        val_metrics = run_epoch(model, val_loader, criterion)
        scheduler.step()
        elapsed_seconds = time.perf_counter() - started_at
        row: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "elapsed_seconds": elapsed_seconds,
        }
        history.append(row)
        logger.info(
            "Epoch %02d/%d | train loss=%.4f, acc=%.4f | val loss=%.4f, acc=%.4f, "
            "macro-F1=%.4f | %.1fs",
            epoch,
            epochs,
            train_metrics["loss"],
            train_metrics["accuracy"],
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
            elapsed_seconds,
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            best_train_metrics = dict(train_metrics)
            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    return TrainingResult(
        history=history,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        best_train_metrics=best_train_metrics,
        best_state_dict=best_state_dict,
    )


def _default_run_name(backbone: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{backbone}-{timestamp}-{uuid.uuid4().hex[:8]}"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pet breed classifier.")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--run-name")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Smoke-test limit for training batches per epoch.",
    )
    return parser.parse_args(argv)


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _log_calibration_metrics(tracker: tracking.Tracker, result: calibrate.FitResult) -> None:
    """Map calibration output to the stable MLflow metric names."""
    metrics: dict[str, float] = {
        "top1": result.after["top1"],
        "f1_macro": result.after["macro_f1"],
        "ece": result.after["ece"],
        "temperature": result.temperature,
        "ece_before": result.before["ece"],
        "nll_before": result.before["nll"],
        "nll_after": result.after["nll"],
    }
    if result.selection.threshold is not None:
        metrics.update(
            {
                "abstain_threshold": result.selection.threshold,
            }
        )
        if result.selection.coverage is not None:
            metrics["coverage"] = result.selection.coverage
        if result.selection.selective_accuracy is not None:
            metrics["selective_accuracy"] = result.selection.selective_accuracy
    tracker.log_metrics(metrics)


def main(argv: Sequence[str] | None = None) -> None:
    """Train, export, calibrate, and track one isolated run."""
    setup_logging()
    global _SCALER
    args = _parse_args(argv)
    if args.backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone {args.backbone!r}; choose from {SUPPORTED_BACKBONES}"
        )
    lr = cfg.LEARNING_RATE if args.lr is None else args.lr
    batch_size = cfg.BATCH_SIZE if args.batch_size is None else args.batch_size
    epochs = cfg.EPOCHS if args.epochs is None else args.epochs
    _validate_positive("lr", lr)
    _validate_positive("batch-size", batch_size)
    _validate_positive("epochs", epochs)
    if args.max_train_batches is not None:
        _validate_positive("max-train-batches", args.max_train_batches)

    seed_everything(cfg.SEED)
    train_records = load_records("train")
    val_records = load_records("val")
    train_dataset = PetBreedDataset(train_records, build_train_transform())
    val_dataset = PetBreedDataset(val_records, build_eval_transform())

    requested_workers = (
        cfg.NUM_WORKERS
        if cfg.NUM_WORKERS is not None
        else max(1, (os.cpu_count() or 2) - 1)
    )
    num_workers = min(requested_workers, 8)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(cfg.SEED)
    loader_options = {
        "num_workers": num_workers,
        "pin_memory": DEVICE.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": loader_generator,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_options,
    )

    classes = _load_classes()
    run_name = args.run_name or _default_run_name(args.backbone)
    run_artifacts = create_run_artifacts(run_name, cfg.ARTIFACTS_DIR)
    model = PetBreedClassifier(
        num_classes=len(classes), pretrained=True, backbone=args.backbone
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    _SCALER = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    with tracking.tracking_run(
        run_name=run_name,
    ) as tracker:
        tracker.log_params(
            {
                "backbone": args.backbone,
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "split_seed": cfg.SEED,
                "train_seed": cfg.SEED,
                "seed": cfg.SEED,
                "amp": USE_AMP,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "target_selective_accuracy": cfg.TARGET_SELECTIVE_ACCURACY,
            }
        )
        training_started_at = time.perf_counter()
        result = train_epochs(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            epochs,
            max_train_batches=args.max_train_batches,
        )
        total_training_seconds = time.perf_counter() - training_started_at
        model.load_state_dict(result.best_state_dict)
        save_checkpoint(
            model,
            run_artifacts.checkpoint,
            classes,
            seed=cfg.SEED,
            epoch=result.best_epoch,
            validation_metrics=result.best_metrics,
        )
        run_artifacts.history.write_text(
            json.dumps(result.history, indent=2) + "\n", encoding="utf-8"
        )
        export(run_artifacts.checkpoint, run_artifacts.onnx, run_artifacts.metadata)
        parity_difference = validate_onnx_parity(
            run_artifacts.checkpoint, run_artifacts.onnx, val_records
        )
        # These are the metrics measured by the training-time validation loop. The
        # ONNX-served validation metrics are logged from the calibration result below.
        train_eval_metrics = {
            "top1": result.best_metrics["accuracy"],
            "macro_f1": result.best_metrics["macro_f1"],
        }
        calibration = calibrate.fit_artifacts(
            run_artifacts.directory,
            cfg.TARGET_SELECTIVE_ACCURACY,
            seed=cfg.SEED,
        )
        save_checkpoint(
            model,
            run_artifacts.checkpoint,
            classes,
            seed=cfg.SEED,
            epoch=result.best_epoch,
            validation_metrics=result.best_metrics,
            temperature=calibration.temperature,
            abstain_threshold=calibration.selection.threshold,
        )

        tracker.log_metrics(
            {
                "train_eval_top1": train_eval_metrics["top1"],
                "train_eval_f1_macro": train_eval_metrics["macro_f1"],
                "train_duration_seconds": total_training_seconds,
                "model_size_mb": tracking.model_size_mb(run_artifacts.onnx),
                "param_count": float(sum(parameter.numel() for parameter in model.parameters())),
                "max_logit_parity_diff": parity_difference,
            }
        )
        _log_calibration_metrics(tracker, calibration)
        unlogged = tracking.log_history(tracker, result.history)
        if unlogged:
            logger.warning("Metrics not logged to MLflow: %s", ", ".join(unlogged))
        for path in (
            run_artifacts.onnx,
            run_artifacts.metadata,
            calibration.figure_path,
            calibration.report_path,
            run_artifacts.history,
            cfg.LABEL_MAP_PATH,
            cfg.SPLIT_INDEX_PATH,
        ):
            tracker.log_artifact(path)
        tracker.log_requirements()

        calibrate._print_fit_summary(calibration)
        logger.info("Run artifacts: %s", run_artifacts.directory)


if __name__ == "__main__":
    main()
