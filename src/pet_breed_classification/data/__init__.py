"""Manifest loading and the Oxford-IIIT Pet dataset."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from ..config import cfg

logger = logging.getLogger(__name__)

Record = dict[str, Any]


def _load_classes() -> list[Record]:
    """Load and validate the committed label map."""
    if not cfg.LABEL_MAP_PATH.is_file():
        raise FileNotFoundError(f"Missing label map: {cfg.LABEL_MAP_PATH}")

    label_map = json.loads(cfg.LABEL_MAP_PATH.read_text(encoding="utf-8"))
    classes = label_map.get("classes")
    if not isinstance(classes, list) or len(classes) != 37:
        raise ValueError("Label map must contain exactly 37 classes")
    if [entry.get("index") for entry in classes] != list(range(37)):
        raise ValueError("Label map class indices must be exactly 0 through 36")
    return classes


def load_records(split: str) -> list[dict[str, Any]]:
    """Load clean records for ``train`` or ``val`` and validate their labels."""
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    if not cfg.MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing manifest: {cfg.MANIFEST_PATH}")

    classes = _load_classes()
    class_by_breed = {str(entry["breed"]): entry for entry in classes}
    manifest = json.loads(cfg.MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise TypeError("Manifest must contain a list of records")

    records = [
        record
        for record in manifest
        if record.get("corruption") is None and record.get("split") == split
    ]
    if not records:
        raise ValueError(f"No clean {split} records were found")

    for record in records:
        breed = record.get("breed")
        if breed not in class_by_breed:
            raise ValueError(f"Breed {breed!r} is missing from the label map")
        label = class_by_breed[breed]
        if (
            record.get("class_index") != label["index"]
            or record.get("species") != label["species"]
        ):
            raise ValueError(
                f"Label metadata does not match the label map for {record.get('image_id')!r}"
            )
        image_path = cfg.PROJECT_ROOT / str(record["path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Manifest image does not exist: {image_path}")

    logger.info("Loaded %d clean %s records", len(records), split)
    return records


class PetBreedDataset(Dataset[tuple[Tensor, int]]):
    """Load manifest records using paths relative to the project root."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """Load and transform one image and return its class index."""
        record = self.records[index]
        image_path = cfg.PROJECT_ROOT / str(record["path"])
        try:
            with Image.open(image_path) as image:
                tensor = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Could not load image {image_path}") from exc
        return tensor, int(record["class_index"])
