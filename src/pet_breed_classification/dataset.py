"""Dataset utilities shared by training entry points."""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class PetBreedDataset(Dataset[tuple[Tensor, int]]):
    """Load manifest records without silently substituting missing images."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        project_root: Path,
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        self.records = records
        self.project_root = project_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        record = self.records[index]
        image_path = self.project_root / str(record["path"])
        try:
            with Image.open(image_path) as image:
                tensor = self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Could not load image {image_path}") from exc
        return tensor, int(record["class_index"])


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy in a DataLoader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
