"""Image preprocessing shared by training, evaluation, and serving."""

from __future__ import annotations

import logging

from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

IMAGE_SIZE = 224
RESIZE_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EVAL_TRANSFORM_METADATA: dict[str, object] = {
    "color_mode": "RGB",
    "resize_size": RESIZE_SIZE,
    "crop_size": IMAGE_SIZE,
    "mean": list(IMAGENET_MEAN),
    "std": list(IMAGENET_STD),
}


def to_rgb(image: Image.Image) -> Image.Image:
    """Convert an image from any supported PIL mode to RGB."""
    return image.convert("RGB")


def build_train_transform() -> transforms.Compose:
    """Build the notebook's training augmentation and normalization pipeline."""
    return transforms.Compose(
        [
            transforms.Lambda(to_rgb),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(
    metadata: dict[str, object] = EVAL_TRANSFORM_METADATA,
) -> transforms.Compose:
    """Build a deterministic evaluation transform from checkpoint metadata."""
    resize_size = int(metadata["resize_size"])
    crop_size = int(metadata["crop_size"])
    mean_values = metadata["mean"]
    std_values = metadata["std"]
    if not isinstance(mean_values, (list, tuple)) or not isinstance(
        std_values, (list, tuple)
    ):
        raise TypeError("Transform metadata mean and std must be lists or tuples")
    mean = tuple(float(value) for value in mean_values)
    std = tuple(float(value) for value in std_values)
    return transforms.Compose(
        [
            transforms.Lambda(to_rgb),
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
