"""Reproducible ResNet-50 training entry point."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from .config import cfg
from .data import PetBreedDataset, load_records
from .logging_conf import setup_logging
from .model import PetBreedClassifier, save_checkpoint
from .transforms import build_eval_transform, build_train_transform

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
_SCALER: torch.amp.GradScaler | None = None


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
        for images, targets in loader:
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

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "macro_f1": f1_score(
            targets_seen, predictions, average="macro", zero_division=0
        ),
    }


def _load_classes() -> list[dict[str, Any]]:
    """Load the committed class metadata for checkpoint serialization."""
    label_map = json.loads(cfg.LABEL_MAP_PATH.read_text(encoding="utf-8"))
    classes = label_map.get("classes")
    if not isinstance(classes, list):
        raise TypeError("Label map must contain a classes list")
    return classes


def main() -> None:
    """Train ResNet-50 and save the best validation checkpoint."""
    setup_logging()
    global _SCALER

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
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        **loader_options,
    )

    classes = _load_classes()
    model = PetBreedClassifier(num_classes=len(classes), pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    _SCALER = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    history: list[dict[str, float | int]] = []
    best_val_accuracy = -1.0
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    training_started_at = time.perf_counter()

    for epoch in range(1, cfg.EPOCHS + 1):
        started_at = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, criterion, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion)
        scheduler.step()
        elapsed_seconds = time.perf_counter() - started_at
        row: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
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
            cfg.EPOCHS,
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
            best_metrics = val_metrics
            save_checkpoint(
                model=model,
                path=cfg.MODEL_PATH,
                classes=classes,
                seed=cfg.SEED,
                epoch=epoch,
                validation_metrics=val_metrics,
            )

    history_path = cfg.MODEL_PATH.with_name("resnet50_training_history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    total_training_seconds = time.perf_counter() - training_started_at
    logger.info(
        "Best epoch: %d; validation accuracy: %.4f; val macro-F1: %.4f",
        best_epoch,
        best_val_accuracy,
        best_metrics["macro_f1"],
    )
    logger.info("Total training time: %.1fs", total_training_seconds)
    logger.info("Checkpoint: %s", cfg.MODEL_PATH)
    logger.info("History: %s", history_path)


if __name__ == "__main__":
    main()
