"""ResNet-50 model and checkpoint serialization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torchvision import models

from .transforms import EVAL_TRANSFORM_METADATA

logger = logging.getLogger(__name__)

CHECKPOINT_KEYS = {
    "backbone",
    "num_classes",
    "classes",
    "eval_transform",
    "temperature",
    "abstain_threshold",
    "seed",
    "epoch",
    "validation_metrics",
    "model_state_dict",
}


class PetBreedClassifier(nn.Module):
    """ResNet-50 classifier with a configurable output head."""

    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        """Return class logits for a batch of images."""
        return self.backbone(images)


def save_checkpoint(
    model: PetBreedClassifier,
    path: Path,
    classes: list[dict[str, Any]],
    seed: int,
    epoch: int,
    validation_metrics: dict[str, float],
    eval_transform: dict[str, object] = EVAL_TRANSFORM_METADATA,
    temperature: float = 1.0,
    abstain_threshold: float | None = None,
) -> None:
    """Save model weights and the metadata required to reproduce inference."""
    num_classes = model.backbone.fc.out_features
    if num_classes != len(classes):
        raise ValueError(
            f"Model num_classes ({num_classes}) does not match classes length ({len(classes)})"
        )

    checkpoint: dict[str, Any] = {
        "backbone": "resnet50",
        "num_classes": num_classes,
        "classes": classes,
        "eval_transform": eval_transform,
        "temperature": temperature,
        "abstain_threshold": abstain_threshold,
        "seed": seed,
        "epoch": epoch,
        "validation_metrics": validation_metrics,
        "model_state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    logger.info("Saved checkpoint to %s", path)


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[PetBreedClassifier, dict[str, Any]]:
    """Load a checkpoint safely and return its eval-mode model and metadata."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a plain dictionary")
    missing_keys = CHECKPOINT_KEYS.difference(checkpoint)
    if missing_keys:
        raise ValueError(f"Checkpoint is missing keys: {sorted(missing_keys)}")

    num_classes = int(checkpoint["num_classes"])
    classes = checkpoint["classes"]
    if not isinstance(classes, list):
        raise TypeError("Checkpoint classes must be a list")
    if num_classes != len(classes):
        raise ValueError(
            f"Checkpoint num_classes ({num_classes}) does not match classes length ({len(classes)})"
        )
    if checkpoint["backbone"] != "resnet50":
        raise ValueError(f"Unsupported checkpoint backbone: {checkpoint['backbone']!r}")

    model = PetBreedClassifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
