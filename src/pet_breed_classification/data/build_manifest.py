"""Build the combined clean and corrupted image manifest."""

import json
import logging
from pathlib import Path

from PIL import Image

from ..config import cfg
from ..logging_conf import setup_logging
from .corruptions import CORRUPTIONS

logger = logging.getLogger(__name__)

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


def _relative_path(path: Path) -> str:
    return path.relative_to(cfg.PROJECT_ROOT).as_posix()


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _load_label_map() -> dict[str, dict[str, object]]:
    label_map = json.loads(cfg.LABEL_MAP_PATH.read_text(encoding="utf-8"))
    return {entry["breed"]: entry for entry in label_map["classes"]}


def _clean_records() -> list[dict[str, object]]:
    split_index = json.loads(cfg.SPLIT_INDEX_PATH.read_text(encoding="utf-8"))
    labels = _load_label_map()
    records = []

    for split in ("train", "val", "test"):
        for image_id in sorted(split_index[split]):
            breed = image_id.rsplit("_", 1)[0]
            label = labels[breed]
            path = cfg.IMAGES_DIR / f"{image_id}.jpg"
            width, height = _image_size(path)
            records.append(
                {
                    "image_id": image_id,
                    "path": _relative_path(path),
                    "breed": breed,
                    "species": label["species"],
                    "class_index": label["index"],
                    "split": split,
                    "corruption": None,
                    "severity": 0,
                    "width": width,
                    "height": height,
                }
            )

    return records


def _corrupted_records() -> list[dict[str, object]]:
    labels = _load_label_map()
    records = []

    for path in sorted(cfg.CORRUPTED_DIR.glob("*/severity_*/*/*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        relative_parts = path.relative_to(cfg.CORRUPTED_DIR).parts
        if len(relative_parts) != 4:
            raise ValueError(f"Unexpected corrupted image path: {path}")

        corruption, severity_name, breed, filename = relative_parts
        if corruption not in CORRUPTIONS or not severity_name.startswith("severity_"):
            raise ValueError(f"Unexpected corrupted image path: {path}")

        severity = int(severity_name.removeprefix("severity_"))
        image_id = Path(filename).stem
        label = labels[breed]
        width, height = _image_size(path)
        records.append(
            {
                "image_id": image_id,
                "path": _relative_path(path),
                "breed": breed,
                "species": label["species"],
                "class_index": label["index"],
                "split": "test",
                "corruption": corruption,
                "severity": severity,
                "width": width,
                "height": height,
            }
        )

    return records


def build_manifest() -> list[dict[str, object]]:
    records = _clean_records() + _corrupted_records()
    records.sort(key=lambda record: (
        record["split"],
        record["image_id"],
        record["corruption"] or "",
        record["severity"],
    ))
    return records


def main() -> None:
    setup_logging()
    records = build_manifest()
    if not records:
        raise ValueError("No images found while building the manifest")

    invalid_fields = REQUIRED_FIELDS - set(records[0])
    if invalid_fields:
        raise ValueError(f"Manifest is missing fields: {sorted(invalid_fields)}")

    cfg.MANIFEST_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote %d records to %s", len(records), cfg.MANIFEST_PATH)


if __name__ == "__main__":
    main()
