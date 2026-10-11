"""Per-image drift features: penultimate embedding, pixel statistics, calibrated confidence."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from pet_breed_classification.config import cfg
from pet_breed_classification.model import load_checkpoint
from pet_breed_classification.predict import PetBreedPredictor
from pet_breed_classification.transforms import build_eval_transform

logger = logging.getLogger(__name__)

PIXEL_STATS = ("brightness", "contrast", "sharpness")
# ITU-R BT.601 luma weights.
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float64)
EXTRACT_BATCH_SIZE = 64

Features = dict[str, NDArray[np.floating]]


def pixel_stats(image: Image.Image) -> tuple[float, float, float]:
    """Return (brightness, contrast, sharpness) of an image at its native resolution.

    brightness = mean luminance, contrast = luminance standard deviation, and
    sharpness = variance of the 4-neighbour Laplacian of the luminance.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    luminance = rgb @ LUMA_WEIGHTS
    laplacian = (
        luminance[:-2, 1:-1]
        + luminance[2:, 1:-1]
        + luminance[1:-1, :-2]
        + luminance[1:-1, 2:]
        - 4.0 * luminance[1:-1, 1:-1]
    )
    return float(luminance.mean()), float(luminance.std()), float(laplacian.var())


class FeatureExtractor:
    """Extract drift features with the shipped checkpoint and the shared transforms."""

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        predictor: PetBreedPredictor | None = None,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, checkpoint = load_checkpoint(checkpoint_path or cfg.MODEL_PATH, self.device)
        # Dropping the classification head leaves the 2048-d pooled ResNet-50 features.
        model.backbone.fc = torch.nn.Identity()
        self.model = model
        self.transform = build_eval_transform(checkpoint["eval_transform"])
        self.predictor = predictor or PetBreedPredictor.load()
        logger.info("Feature extractor on %s", self.device)

    @torch.inference_mode()
    def extract(self, paths: Sequence[Path], with_confidence: bool = True) -> Features:
        """Return features for every path, in order."""
        embeddings, stats, confidences = [], [], []
        for start in range(0, len(paths), EXTRACT_BATCH_SIZE):
            images = []
            for path in paths[start : start + EXTRACT_BATCH_SIZE]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            batch = torch.stack([self.transform(image) for image in images]).to(self.device)
            embeddings.append(self.model(batch).float().cpu().numpy())
            stats.extend(pixel_stats(image) for image in images)
            if with_confidence:
                confidences.append(self.predictor.probabilities(images).max(axis=1))
        stat_array = np.asarray(stats, dtype=np.float64)
        features: Features = {
            "embedding": np.concatenate(embeddings).astype(np.float32),
            **{name: stat_array[:, index] for index, name in enumerate(PIXEL_STATS)},
        }
        if with_confidence:
            features["max_confidence"] = np.concatenate(confidences)
        return features


def save_features(path: Path, features: Features) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **features)


def load_features(path: Path) -> Features:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def subset(features: Features, indices: NDArray[np.integer]) -> Features:
    """Return the rows ``indices`` of every feature array."""
    return {key: value[indices] for key, value in features.items()}
