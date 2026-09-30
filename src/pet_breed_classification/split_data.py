import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from .config import cfg

Record = tuple[str, str, int, str]


def _read_annotations(path: Path) -> list[Record]:
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {path}")

    records: list[Record] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        fields = line.split()
        if len(fields) != 4:
            raise ValueError(
                f"{path}:{line_number}: expected 4 fields, got {len(fields)}"
            )

        image_id, class_id_text, species_id_text, breed_id_text = fields
        if image_id in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate image_id {image_id!r}")
        seen_ids.add(image_id)

        try:
            class_id = int(class_id_text)
            species_id = int(species_id_text)
            int(breed_id_text)
        except ValueError as error:
            raise ValueError(
                f"{path}:{line_number}: class, species, and breed IDs must be integers"
            ) from error

        if not 1 <= class_id <= 37:
            raise ValueError(f"{path}:{line_number}: class_id must be in 1..37")
        if species_id not in (1, 2):
            raise ValueError(f"{path}:{line_number}: species must be 1 or 2")
        if "_" not in image_id:
            raise ValueError(
                f"{path}:{line_number}: image_id has no breed separator: {image_id!r}"
            )

        image_path = cfg.IMAGES_DIR / f"{image_id}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(
                f"{path}:{line_number}: image file does not exist: {image_path}"
            )

        breed = image_id.rsplit("_", 1)[0]
        species = "cat" if species_id == 1 else "dog"
        records.append((image_id, breed, class_id, species))

    return records


def _validate_breed_metadata(records: list[Record]) -> dict[str, tuple[int, str]]:
    breed_metadata: dict[str, tuple[int, str]] = {}
    for _, breed, class_id, species in records:
        previous = breed_metadata.get(breed)
        current = (class_id, species)
        if previous is not None and previous != current:
            raise ValueError(
                f"Breed {breed!r} maps to both {previous} and {current}"
            )
        breed_metadata[breed] = current
    return breed_metadata


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    trainval_records = _read_annotations(cfg.ANNOTATIONS_DIR / "trainval.txt")
    test_records = _read_annotations(cfg.ANNOTATIONS_DIR / "test.txt")

    trainval_ids = {record[0] for record in trainval_records}
    test_ids = {record[0] for record in test_records}
    overlap = sorted(trainval_ids & test_ids)
    if overlap:
        raise ValueError(
            "Image IDs appear in both trainval.txt and test.txt: "
            + ", ".join(overlap)
        )

    breed_metadata = _validate_breed_metadata(trainval_records + test_records)
    breeds = sorted(breed_metadata, key=lambda breed: (breed.casefold(), breed))
    if len(breeds) != 37:
        raise ValueError(f"Expected exactly 37 breeds, found {len(breeds)}")

    classes = []
    for index, breed in enumerate(breeds):
        class_id, species = breed_metadata[breed]
        assert index == class_id - 1, (
            f"Label-map order mismatch for {breed!r}: "
            f"index {index} != class_id - 1 ({class_id - 1})"
        )
        classes.append({"index": index, "breed": breed, "species": species})

    sorted_trainval = sorted(trainval_records, key=lambda record: record[0])
    sorted_trainval_ids = [record[0] for record in sorted_trainval]
    sorted_trainval_breeds = [record[1] for record in sorted_trainval]
    train_ids, val_ids = train_test_split(
        sorted_trainval_ids,
        test_size=cfg.VAL_RATIO,
        stratify=sorted_trainval_breeds,
        random_state=cfg.SEED,
    )
    train_ids = sorted(train_ids)
    val_ids = sorted(val_ids)
    test_ids_sorted = sorted(test_ids)

    _write_json(cfg.LABEL_MAP_PATH, {"classes": classes})
    _write_json(
        cfg.SPLIT_INDEX_PATH,
        {
            "seed": cfg.SEED,
            "val_ratio": cfg.VAL_RATIO,
            "train": train_ids,
            "val": val_ids,
            "test": test_ids_sorted,
        },
    )

    train_counts = Counter(record[1] for record in sorted_trainval if record[0] in train_ids)
    val_counts = Counter(record[1] for record in sorted_trainval if record[0] in val_ids)
    print(f"train: {len(train_ids)} images")
    print(f"val: {len(val_ids)} images")
    print(f"test: {len(test_ids_sorted)} images")
    print(
        "train images per class: "
        f"min={min(train_counts.values())}, max={max(train_counts.values())}"
    )
    print(
        "val images per class: "
        f"min={min(val_counts.values())}, max={max(val_counts.values())}"
    )


if __name__ == "__main__":
    main()
