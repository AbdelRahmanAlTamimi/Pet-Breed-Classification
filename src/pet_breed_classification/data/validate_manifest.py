"""Validate the combined clean and corrupted image manifest."""

import json

from PIL import Image

from ..config import cfg
from .corruptions import CORRUPTIONS

REQUIRED_FIELDS = {
    "image_id",
    "path",
    "breed",
    "species",
    "class_index",
    "split",
    "corruption",
    "severity",
    "width",
    "height",
}


def validate_manifest() -> None:
    records = json.loads(cfg.MANIFEST_PATH.read_text(encoding="utf-8"))
    label_map = json.loads(cfg.LABEL_MAP_PATH.read_text(encoding="utf-8"))
    classes = label_map["classes"]
    labels = {entry["breed"]: entry for entry in classes}

    if len(classes) != 37 or [entry["index"] for entry in classes] != list(range(37)):
        raise ValueError("Label map must contain exactly 37 ordered classes")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must be a non-empty list")

    seen_paths: set[str] = set()
    clean_ids_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    train_counts = {entry["breed"]: 0 for entry in classes}

    for record in records:
        if set(record) != REQUIRED_FIELDS:
            raise ValueError(f"Invalid manifest fields for {record.get('path')}")

        path_text = record["path"]
        if path_text in seen_paths:
            raise ValueError(f"Duplicate manifest path: {path_text}")
        seen_paths.add(path_text)

        path = cfg.PROJECT_ROOT / path_text
        if not path.is_file():
            raise FileNotFoundError(f"Manifest path does not exist: {path}")
        with Image.open(path) as image:
            if image.size != (record["width"], record["height"]):
                raise ValueError(f"Manifest dimensions do not match image: {path}")
            if image.width < 32 or image.height < 32:
                raise ValueError(f"Image is smaller than 32x32: {path}")

        breed = record["breed"]
        if breed not in labels:
            raise ValueError(f"Unknown breed in manifest: {breed}")
        label = labels[breed]
        if record["species"] != label["species"] or record["class_index"] != label["index"]:
            raise ValueError(f"Label metadata does not match map: {path}")

        split = record["split"]
        if split not in clean_ids_by_split:
            raise ValueError(f"Invalid split in manifest: {split}")

        corruption = record["corruption"]
        severity = record["severity"]
        if corruption is None:
            if severity != 0 or record["image_id"] in clean_ids_by_split[split]:
                raise ValueError(f"Invalid clean record: {path}")
            clean_ids_by_split[split].add(record["image_id"])
            if split == "train":
                train_counts[breed] += 1
        elif corruption not in CORRUPTIONS or severity not in (1, 2, 3):
            raise ValueError(f"Invalid corruption metadata: {path}")

    if not clean_ids_by_split["train"].isdisjoint(clean_ids_by_split["val"]):
        raise ValueError("An image appears in both train and val")
    if not clean_ids_by_split["train"].isdisjoint(clean_ids_by_split["test"]):
        raise ValueError("An image appears in both train and test")
    if not clean_ids_by_split["val"].isdisjoint(clean_ids_by_split["test"]):
        raise ValueError("An image appears in both val and test")

    underrepresented = [breed for breed, count in train_counts.items() if count < 50]
    if underrepresented:
        raise ValueError(f"Classes with fewer than 50 training images: {underrepresented}")


def main() -> None:
    validate_manifest()
    print(f"Manifest validation passed: {cfg.MANIFEST_PATH}")


if __name__ == "__main__":
    main()
