"""Batch-score trainval images through the shared calibrated predictor."""

from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from pathlib import Path
from typing import Any

from PIL import Image

from .config import cfg
from .predict import PetBreedPredictor, Prediction

IMAGE_SUFFIX = ".jpg"
BATCH_SIZE = 32


def _trainval_records() -> list[dict[str, Any]]:
    """Build records from train and val only; never inspect the test index."""
    split_index = json.loads(cfg.SPLIT_INDEX_PATH.read_text(encoding="utf-8"))
    image_ids = [
        image_id
        for split in ("train", "val")
        for image_id in split_index.get(split, [])
    ]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("train and val contain duplicate image IDs")
    label_map = json.loads(cfg.LABEL_MAP_PATH.read_text(encoding="utf-8"))
    classes = {str(item["breed"]): item for item in label_map["classes"]}
    records = []
    for image_id in image_ids:
        breed = str(image_id).rsplit("_", 1)[0]
        if breed not in classes:
            raise ValueError(f"Image {image_id!r} has no label-map breed")
        path = cfg.IMAGES_DIR / f"{image_id}{IMAGE_SUFFIX}"
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "image_id": str(image_id),
                "path": path,
                "breed": breed,
            }
        )
    return records


def _row(image_id: str, prediction: Prediction, model_version: str) -> dict[str, Any]:
    """Convert one shared predictor result to the scoring output schema."""
    return {
        "image_id": image_id,
        "breed": prediction.breed,
        "confidence": prediction.confidence,
        "decision": prediction.decision,
        "model_version": model_version,
    }


def score_records(
    records: list[dict[str, Any]],
    predictor: PetBreedPredictor,
    output_path: Path,
    *,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    """Score records in chunks and return measured run statistics."""
    started_at = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_version = str(predictor.metadata["model_version"])
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["image_id", "breed", "confidence", "decision", "model_version"],
        )
        writer.writeheader()
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            images = []
            for record in chunk:
                with Image.open(record["path"]) as image:
                    images.append(image.convert("RGB"))
            predictions = predictor.predict_batch(images)
            for record, prediction in zip(chunk, predictions, strict=True):
                writer.writerow(_row(record["image_id"], prediction, model_version))

    wall_clock_seconds = time.perf_counter() - started_at
    images_per_second = len(records) / wall_clock_seconds if wall_clock_seconds else 0.0
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "images_scored": len(records),
        "wall_clock_seconds": wall_clock_seconds,
        "images_per_second": images_per_second,
        "peak_rss_kib": peak_rss_kib,
        "output_path": output_path.as_posix(),
        "row_count": len(records),
    }


def main() -> None:
    """Score the requested number of trainval images."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=cfg.PROJECT_ROOT / "data" / "scoring" / "output" / "batch_predictions.csv",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    records = _trainval_records()
    if args.limit is not None:
        if args.limit > len(records):
            parser.error(f"--limit exceeds available trainval images ({len(records)})")
        records = records[: args.limit]
    result = score_records(records, PetBreedPredictor.load(), args.output, batch_size=args.batch_size)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
