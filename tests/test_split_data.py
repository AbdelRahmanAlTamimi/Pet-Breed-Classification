import json
from collections import Counter
from pathlib import Path

import pytest

from pet_breed_classification.config import Config, cfg
from pet_breed_classification.data import split_data


def _annotation_ids(path: Path) -> set[str]:
    return {
        line.split()[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _rewrite_annotation(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_label_map() -> None:
    label_map = json.loads(cfg.LABEL_MAP_PATH.read_text(encoding="utf-8"))
    classes = label_map["classes"]

    assert len(classes) == 37
    assert [entry["index"] for entry in classes] == list(range(37))
    assert len({entry["breed"] for entry in classes}) == 37
    assert [entry["breed"] for entry in classes] == sorted(
        (entry["breed"] for entry in classes), key=str.casefold
    )


def test_synthetic_split_generation(tiny_pet_dataset: Config) -> None:
    split_data.main()
    split_index = json.loads(tiny_pet_dataset.SPLIT_INDEX_PATH.read_text(encoding="utf-8"))
    train = set(split_index["train"])
    val = set(split_index["val"])
    test = set(split_index["test"])
    all_ids = _annotation_ids(tiny_pet_dataset.ANNOTATIONS_DIR / "trainval.txt") | _annotation_ids(
        tiny_pet_dataset.ANNOTATIONS_DIR / "test.txt"
    )

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == all_ids
    assert test == _annotation_ids(tiny_pet_dataset.ANNOTATIONS_DIR / "test.txt")

    train_breeds = Counter(image_id.rsplit("_", 1)[0] for image_id in train)
    val_breeds = Counter(image_id.rsplit("_", 1)[0] for image_id in val)
    assert set(train_breeds) == set(val_breeds) == {
        f"breed_{index:02d}" for index in range(37)
    }

    classes = json.loads(tiny_pet_dataset.LABEL_MAP_PATH.read_text(encoding="utf-8"))["classes"]
    assert len(classes) == 37
    assert [entry["breed"] for entry in classes] == sorted(
        (entry["breed"] for entry in classes), key=str.casefold
    )
    assert all(entry["index"] == int(entry["breed"][-2:]) for entry in classes)


def test_synthetic_split_generation_is_byte_stable(tiny_pet_dataset: Config) -> None:
    split_data.main()
    label_map_before = tiny_pet_dataset.LABEL_MAP_PATH.read_bytes()
    split_index_before = tiny_pet_dataset.SPLIT_INDEX_PATH.read_bytes()

    split_data.main()

    assert tiny_pet_dataset.LABEL_MAP_PATH.read_bytes() == label_map_before
    assert tiny_pet_dataset.SPLIT_INDEX_PATH.read_bytes() == split_index_before


def test_duplicate_image_id_is_rejected(tiny_pet_dataset: Config) -> None:
    path = tiny_pet_dataset.ANNOTATIONS_DIR / "trainval.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    _rewrite_annotation(path, lines + [lines[0]])

    with pytest.raises(ValueError, match="duplicate image_id"):
        split_data.main()


def test_missing_image_is_rejected(tiny_pet_dataset: Config) -> None:
    path = tiny_pet_dataset.ANNOTATIONS_DIR / "trainval.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    fields = lines[0].split()
    fields[0] = "breed_00_missing"
    lines[0] = " ".join(fields)
    _rewrite_annotation(path, lines)

    with pytest.raises(FileNotFoundError, match="image file does not exist"):
        split_data.main()


def test_overlap_between_splits_is_rejected(tiny_pet_dataset: Config) -> None:
    trainval_path = tiny_pet_dataset.ANNOTATIONS_DIR / "trainval.txt"
    test_path = tiny_pet_dataset.ANNOTATIONS_DIR / "test.txt"
    trainval_lines = trainval_path.read_text(encoding="utf-8").splitlines()
    test_lines = test_path.read_text(encoding="utf-8").splitlines()
    _rewrite_annotation(trainval_path, trainval_lines + [test_lines[0]])

    with pytest.raises(ValueError, match="both trainval.txt and test.txt"):
        split_data.main()


def test_class_id_outside_range_is_rejected(tiny_pet_dataset: Config) -> None:
    path = tiny_pet_dataset.ANNOTATIONS_DIR / "trainval.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    fields = lines[0].split()
    fields[1] = "38"
    lines[0] = " ".join(fields)
    _rewrite_annotation(path, lines)

    with pytest.raises(ValueError, match="class_id must be in 1..37"):
        split_data.main()


def test_breed_with_two_class_ids_is_rejected(tiny_pet_dataset: Config) -> None:
    path = tiny_pet_dataset.ANNOTATIONS_DIR / "trainval.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    fields = lines[1].split()
    fields[1] = "2"
    lines[1] = " ".join(fields)
    _rewrite_annotation(path, lines)

    with pytest.raises(ValueError, match="maps to both"):
        split_data.main()


def test_invalid_species_is_rejected(tiny_pet_dataset: Config) -> None:
    path = tiny_pet_dataset.ANNOTATIONS_DIR / "trainval.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    fields = lines[0].split()
    fields[2] = "3"
    lines[0] = " ".join(fields)
    _rewrite_annotation(path, lines)

    with pytest.raises(ValueError, match="species must be 1 or 2"):
        split_data.main()


@pytest.mark.skipif(
    not (cfg.ANNOTATIONS_DIR / "test.txt").is_file(),
    reason="official Oxford-IIIT Pet annotations unavailable",
)
def test_split_index() -> None:
    split_index = json.loads(cfg.SPLIT_INDEX_PATH.read_text(encoding="utf-8"))
    train = set(split_index["train"])
    val = set(split_index["val"])
    test = set(split_index["test"])

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert len(train | val | test) == 7349

    test_ids = _annotation_ids(cfg.ANNOTATIONS_DIR / "test.txt")
    assert test == test_ids

    trainval_breeds = {
        image_id: image_id.rsplit("_", 1)[0]
        for image_id in train | val
    }
    train_breeds = Counter(trainval_breeds[image_id] for image_id in train)
    val_breeds = Counter(trainval_breeds[image_id] for image_id in val)
    assert set(train_breeds) == set(val_breeds)


@pytest.mark.skipif(
    not (cfg.ANNOTATIONS_DIR / "test.txt").is_file(),
    reason="official Oxford-IIIT Pet annotations unavailable",
)
def test_real_split_generation_matches_committed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_cfg = cfg.model_copy(
        update={
            "LABEL_MAP_PATH": tmp_path / "label_map.json",
            "SPLIT_INDEX_PATH": tmp_path / "split_index.json",
        }
    )
    monkeypatch.setattr(split_data, "cfg", output_cfg)
    monkeypatch.setattr(split_data, "setup_logging", lambda: None)

    split_data.main()

    assert output_cfg.LABEL_MAP_PATH.read_bytes() == cfg.LABEL_MAP_PATH.read_bytes()
    assert output_cfg.SPLIT_INDEX_PATH.read_bytes() == cfg.SPLIT_INDEX_PATH.read_bytes()
