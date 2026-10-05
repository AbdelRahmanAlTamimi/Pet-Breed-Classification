import json

import pytest
import torch

import pet_breed_classification.data as data_module
from pet_breed_classification.config import Config
from pet_breed_classification.data import (
    PetBreedDataset,
    build_manifest,
    load_records,
    split_data,
)
from pet_breed_classification.transforms import build_eval_transform


def _prepare_manifest(tiny_pet_dataset: Config) -> None:
    split_data.main()
    build_manifest.main()


def test_load_records_on_tiny_manifest(tiny_pet_dataset: Config) -> None:
    _prepare_manifest(tiny_pet_dataset)

    train = load_records("train")
    val = load_records("val")
    split_index = json.loads(tiny_pet_dataset.SPLIT_INDEX_PATH.read_text(encoding="utf-8"))

    assert len(train) == 148
    assert len(val) == 37
    assert {record["image_id"] for record in train} == set(split_index["train"])
    assert {record["image_id"] for record in val} == set(split_index["val"])


def test_load_records_rejects_unknown_split() -> None:
    with pytest.raises(ValueError, match="train.*val"):
        load_records("test")


def test_load_records_rejects_missing_manifest(
    tiny_pet_dataset: Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_cfg = tiny_pet_dataset.model_copy(
        update={"MANIFEST_PATH": tiny_pet_dataset.PROJECT_ROOT / "missing.json"}
    )
    monkeypatch.setattr(data_module, "cfg", missing_cfg)

    with pytest.raises(FileNotFoundError, match="Missing manifest"):
        load_records("train")


def test_load_records_rejects_label_metadata_mismatch(tiny_pet_dataset: Config) -> None:
    _prepare_manifest(tiny_pet_dataset)
    records = json.loads(tiny_pet_dataset.MANIFEST_PATH.read_text(encoding="utf-8"))
    first_train = next(record for record in records if record["split"] == "train")
    first_train["class_index"] = (first_train["class_index"] + 1) % 37
    tiny_pet_dataset.MANIFEST_PATH.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(ValueError, match="Label metadata does not match"):
        load_records("train")


def test_load_records_rejects_missing_image(tiny_pet_dataset: Config) -> None:
    _prepare_manifest(tiny_pet_dataset)
    records = json.loads(tiny_pet_dataset.MANIFEST_PATH.read_text(encoding="utf-8"))
    first_train = next(record for record in records if record["split"] == "train")
    first_train["path"] = "data/raw/images/missing.jpg"
    tiny_pet_dataset.MANIFEST_PATH.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Manifest image does not exist"):
        load_records("train")


def test_dataset_returns_tensor_and_class_index(tiny_pet_dataset: Config) -> None:
    _prepare_manifest(tiny_pet_dataset)
    records = load_records("train")

    tensor, class_index = PetBreedDataset(records, build_eval_transform())[0]

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert isinstance(class_index, int)


def test_dataset_wraps_unreadable_image(tiny_pet_dataset: Config) -> None:
    _prepare_manifest(tiny_pet_dataset)
    records = json.loads(tiny_pet_dataset.MANIFEST_PATH.read_text(encoding="utf-8"))
    first_train = next(record for record in records if record["split"] == "train")
    bad_path = tiny_pet_dataset.PROJECT_ROOT / "data/raw/images/bad.jpg"
    bad_path.write_bytes(b"not an image")
    first_train["path"] = "data/raw/images/bad.jpg"
    tiny_pet_dataset.MANIFEST_PATH.write_text(json.dumps(records), encoding="utf-8")
    loaded_records = load_records("train")

    with pytest.raises(RuntimeError, match="Could not load image"):
        PetBreedDataset(loaded_records, build_eval_transform())[0]
