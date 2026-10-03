"""Shared ResNet-50 model and image transforms."""

from __future__ import annotations

from torch import Tensor, nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

IMAGE_SIZE = 224
RESIZE_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform() -> transforms.Compose:
    """Build training-only augmentation and ImageNet preprocessing."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    """Build the deterministic transform shared by validation and serving."""
    return transforms.Compose(
        [
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class PetBreedClassifier(nn.Module):
    """ResNet-50 classifier with an Oxford-IIIT Pet output head."""

    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        return self.backbone(images)
