"""Validation tests for the committed manifest and label map."""

import json

from PIL import Image

from pet_breed_classification.config import cfg
from pet_breed_classification.data.corruptions import CORRUPTIONS

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


def test_manifest_paths_schema_and_labels() -> None:
    records = json.loads(cfg.MANIFEST_PATH.read_text(encoding="utf-8"))
    label_map = json.loads(cfg.LABEL_MAP_PATH.read_text(encoding="utf-8"))
    classes = label_map["classes"]
    labels = {entry["breed"]: entry for entry in classes}

    assert len(classes) == 37
    assert [entry["index"] for entry in classes] == list(range(37))
    assert records

    seen_paths: set[str] = set()
    clean_ids_by_split: dict[str, set[str]] = {
        "train": set(),
        "val": set(),
        "test": set(),
    }
    train_counts = {entry["breed"]: 0 for entry in classes}

    for record in records:
        assert set(record) == REQUIRED_FIELDS

        path_text = record["path"]
        assert path_text not in seen_paths
        seen_paths.add(path_text)

        path = cfg.PROJECT_ROOT / path_text
        assert path.is_file()
        with Image.open(path) as image:
            assert image.size == (record["width"], record["height"])
            assert image.width >= 32
            assert image.height >= 32

        breed = record["breed"]
        assert breed in labels
        label = labels[breed]
        assert record["species"] == label["species"]
        assert record["class_index"] == label["index"]

        split = record["split"]
        assert split in clean_ids_by_split
        corruption = record["corruption"]
        severity = record["severity"]
        if corruption is None:
            assert severity == 0
            assert record["image_id"] not in clean_ids_by_split[split]
            clean_ids_by_split[split].add(record["image_id"])
            if split == "train":
                train_counts[breed] += 1
        else:
            assert corruption in CORRUPTIONS
            assert severity in (1, 2, 3)

    assert clean_ids_by_split["train"].isdisjoint(clean_ids_by_split["val"])
    assert clean_ids_by_split["train"].isdisjoint(clean_ids_by_split["test"])
    assert clean_ids_by_split["val"].isdisjoint(clean_ids_by_split["test"])
    assert all(count >= 50 for count in train_counts.values())
