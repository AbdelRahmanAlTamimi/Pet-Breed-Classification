"""Validation tests for clean and corrupted manifests."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from pet_breed_classification.config import cfg
from pet_breed_classification.data import build_manifest, corruptions, split_data
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


def _assert_manifest_valid(
    manifest_path: Path,
    label_map_path: Path,
    project_root: Path,
    min_train_per_breed: int,
    image_splits: set[str] | None = None,
) -> list[dict[str, object]]:
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
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

        split = record["split"]
        assert split in clean_ids_by_split
        if image_splits is None or split in image_splits:
            path = project_root / path_text
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
    assert all(count >= min_train_per_breed for count in train_counts.values())
    return records


@pytest.mark.skipif(
    not cfg.MANIFEST_PATH.is_file() or not cfg.IMAGES_DIR.is_dir(),
    reason="real manifest or Oxford-IIIT Pet images unavailable",
)
def test_manifest_paths_schema_and_labels() -> None:
    _assert_manifest_valid(
        cfg.MANIFEST_PATH,
        cfg.LABEL_MAP_PATH,
        cfg.PROJECT_ROOT,
        50,
        image_splits={"train", "val"},
    )


def test_synthetic_manifest(tiny_pet_dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    split_data.main()
    monkeypatch.setattr(corruptions, "ProcessPoolExecutor", ThreadPoolExecutor)
    corruptions.main()
    build_manifest.main()

    records = _assert_manifest_valid(
        tiny_pet_dataset.MANIFEST_PATH,
        tiny_pet_dataset.LABEL_MAP_PATH,
        tiny_pet_dataset.PROJECT_ROOT,
        3,
    )
    assert sum(record["corruption"] is None for record in records) == 259
    assert sum(record["corruption"] is not None for record in records) == 37 * 6 * 3
