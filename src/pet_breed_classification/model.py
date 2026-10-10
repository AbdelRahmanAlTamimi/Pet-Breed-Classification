"""Transfer-learning model factory and checkpoint serialization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torchvision import models

from .transforms import EVAL_TRANSFORM_METADATA

logger = logging.getLogger(__name__)

SUPPORTED_BACKBONES = ("resnet50", "resnet18", "mobilenet_v3_small")
DEFAULT_BACKBONE = "resnet50"
DEFAULT_NUM_CLASSES = 37

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


def build_backbone(
    backbone: str,
    num_classes: int,
    *,
    pretrained: bool = False,
) -> nn.Module:
    """Build one of the supported ImageNet backbones with a classification head."""
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone {backbone!r}; choose from {SUPPORTED_BACKBONES}"
        )

    if backbone == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        network = models.resnet50(weights=weights)
        network.fc = nn.Linear(network.fc.in_features, num_classes)
    elif backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        network = models.resnet18(weights=weights)
        network.fc = nn.Linear(network.fc.in_features, num_classes)
    else:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        network = models.mobilenet_v3_small(weights=weights)
        classifier = network.classifier
        classifier[-1] = nn.Linear(classifier[-1].in_features, num_classes)

    return network


class PetBreedClassifier(nn.Module):
    """ImageNet transfer-learning classifier with a configurable backbone."""

    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        pretrained: bool = False,
        backbone: str = DEFAULT_BACKBONE,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.backbone = build_backbone(
            backbone, num_classes, pretrained=pretrained
        )

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
    num_classes = _num_classes(model.backbone)
    if num_classes != len(classes):
        raise ValueError(
            f"Model num_classes ({num_classes}) does not match classes length ({len(classes)})"
        )

    checkpoint: dict[str, Any] = {
        "backbone": model.backbone_name,
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
    backbone = checkpoint["backbone"]
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported checkpoint backbone: {backbone!r}")

    model = PetBreedClassifier(
        num_classes=num_classes,
        pretrained=False,
        backbone=str(backbone),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _num_classes(backbone: nn.Module) -> int:
    """Return the output width of a supported torchvision classifier."""
    if hasattr(backbone, "fc"):
        head = backbone.fc
    else:
        head = backbone.classifier[-1]
    if not isinstance(head, nn.Linear):
        raise TypeError("Supported backbone must end in a linear classification head")
    return head.out_features
