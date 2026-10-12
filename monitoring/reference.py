"""Clean reference features, cached under data/monitoring/ (git-ignored, rebuilt on demand).

- Input-drift reference (embedding, pixel): every clean train image.
- Confidence reference: a seeded half of the clean val images. Train confidences are
  near 1.0 because the model saw those images, so they are not a fair reference. The
  other val half supplies the clean batches that set the confidence threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pet_breed_classification.config import cfg
from pet_breed_classification.data import load_records

from .features import FeatureExtractor, Features, load_features, save_features

CACHE_DIR = cfg.PROJECT_ROOT / "data" / "monitoring"
TRAIN_CACHE = CACHE_DIR / "reference_train.npz"
VAL_CACHE = CACHE_DIR / "reference_val.npz"


@dataclass(frozen=True)
class Reference:
    train: Features
    val: Features
    confidence_half: NDArray[np.integer]
    threshold_half: NDArray[np.integer]

    @property
    def confidence(self) -> NDArray[np.floating]:
        return self.val["max_confidence"][self.confidence_half]


def split_paths(split: str) -> list[Path]:
    return [cfg.PROJECT_ROOT / str(record["path"]) for record in load_records(split)]


def val_halves(count: int, seed: int) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
    """Return two disjoint seeded halves of range(count)."""
    order = np.random.default_rng(seed).permutation(count)
    return np.sort(order[: count // 2]), np.sort(order[count // 2 :])


def load_reference(extractor: FeatureExtractor | None = None) -> Reference:
    """Load cached reference features, extracting them first when the cache is missing."""
    if not TRAIN_CACHE.is_file() or not VAL_CACHE.is_file():
        extractor = extractor or FeatureExtractor()
        save_features(TRAIN_CACHE, extractor.extract(split_paths("train"), with_confidence=False))
        save_features(VAL_CACHE, extractor.extract(split_paths("val")))
    val = load_features(VAL_CACHE)
    confidence_half, threshold_half = val_halves(len(val["max_confidence"]), cfg.SEED)
    return Reference(load_features(TRAIN_CACHE), val, confidence_half, threshold_half)
