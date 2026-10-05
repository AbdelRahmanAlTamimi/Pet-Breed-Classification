import json
from collections import Counter

from pet_breed_classification.config import cfg
from pet_breed_classification.data import split_data


def test_label_map() -> None:
    label_map = json.loads(cfg.LABEL_MAP_PATH.read_text(encoding="utf-8"))
    classes = label_map["classes"]

    assert len(classes) == 37
    assert [entry["index"] for entry in classes] == list(range(37))
    assert len({entry["breed"] for entry in classes}) == 37


def test_split_index() -> None:
    split_index = json.loads(cfg.SPLIT_INDEX_PATH.read_text(encoding="utf-8"))
    train = set(split_index["train"])
    val = set(split_index["val"])
    test = set(split_index["test"])

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert len(train | val | test) == 7349

    test_ids = {
        line.split()[0]
        for line in (cfg.ANNOTATIONS_DIR / "test.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert test == test_ids

    trainval_breeds = {
        image_id: image_id.rsplit("_", 1)[0]
        for image_id in train | val
    }
    train_breeds = Counter(trainval_breeds[image_id] for image_id in train)
    val_breeds = Counter(trainval_breeds[image_id] for image_id in val)
    assert set(train_breeds) == set(val_breeds)


def test_split_generation_is_byte_stable() -> None:
    label_map_before = cfg.LABEL_MAP_PATH.read_bytes()
    split_index_before = cfg.SPLIT_INDEX_PATH.read_bytes()

    split_data.main()

    assert cfg.LABEL_MAP_PATH.read_bytes() == label_map_before
    assert cfg.SPLIT_INDEX_PATH.read_bytes() == split_index_before
