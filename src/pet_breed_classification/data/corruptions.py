"""
Corruption suite for Oxford-IIIT Pet.

Generates a corrupted copy of the test split at 3 severity levels for each
corruption type. This is the project's labeled drift ground truth: every
generated image records which corruption and severity was applied, so drift
detectors built later can be scored against a known answer.

Each source image is processed in a separate worker process (multiprocessing),
so work runs in parallel across CPU cores without GIL contention. The manifest
order is deterministic regardless of the number of workers.

Reads:  cfg.SPLIT_INDEX_PATH and cfg.LABEL_MAP_PATH
Writes: cfg.CORRUPTED_DIR/<corruption>/severity_<n>/<breed>/<image_id>.jpg
        cfg.CORRUPTED_DIR (one generated image per corruption and severity)

Run:    uv run python -m pet_breed_classification.data.corruptions
"""

import io
import json
import os
import random
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from ..config import cfg

SEVERITIES = (1, 2, 3)

# Parameters per severity level: index 0 = severity 1 (mild), index 2 = severity 3 (severe).
# Keep data/README.md in sync with these values.
GAUSSIAN_BLUR_RADIUS = (1, 2, 4)
BRIGHTNESS_UP_FACTOR = (1.3, 1.6, 2.0)
BRIGHTNESS_DOWN_FACTOR = (0.7, 0.5, 0.3)
JPEG_QUALITY = (50, 30, 10)
# Target size for the shorter side before upscaling to the original dimensions.
DOWNSCALE_UPSCALE_SIZE = (128, 96, 64)
MOTION_BLUR_KERNEL_SIZE = (5, 9, 15)  # must be odd


# ---------- Corruption functions ----------

def apply_gaussian_blur(img: Image.Image, severity: int) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=GAUSSIAN_BLUR_RADIUS[severity - 1]))


def apply_brightness_up(img: Image.Image, severity: int) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(BRIGHTNESS_UP_FACTOR[severity - 1])


def apply_brightness_down(img: Image.Image, severity: int) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(BRIGHTNESS_DOWN_FACTOR[severity - 1])


def apply_jpeg_compression(img: Image.Image, severity: int) -> Image.Image:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=JPEG_QUALITY[severity - 1])
    buffer.seek(0)
    return Image.open(buffer).copy()  # copy() forces a full load before buffer is released


def apply_downscale_upscale(img: Image.Image, severity: int) -> Image.Image:
    size = DOWNSCALE_UPSCALE_SIZE[severity - 1]
    scale = size / min(img.size)
    small_size = tuple(round(dimension * scale) for dimension in img.size)
    small = img.resize(small_size, Image.Resampling.BILINEAR)
    return small.resize(img.size, Image.Resampling.BILINEAR)


def apply_motion_blur(img: Image.Image, severity: int) -> Image.Image:
    """Horizontal motion blur as a vectorized moving average along the width axis."""
    kernel_size = MOTION_BLUR_KERNEL_SIZE[severity - 1]
    arr = np.asarray(img, dtype=np.float32)
    width = arr.shape[1]
    pad = kernel_size // 2
    padded = np.pad(arr, ((0, 0), (pad, pad), (0, 0)), mode="edge")

    blurred = np.zeros_like(arr)
    for shift in range(kernel_size):
        blurred += padded[:, shift:shift + width, :]
    blurred /= kernel_size

    return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))


CORRUPTIONS: dict[str, Callable[[Image.Image, int], Image.Image]] = {
    "gaussian_blur": apply_gaussian_blur,
    "brightness_up": apply_brightness_up,
    "brightness_down": apply_brightness_down,
    "jpeg_compression": apply_jpeg_compression,
    "downscale_upscale": apply_downscale_upscale,
    "motion_blur": apply_motion_blur,
}


# ---------- Helpers ----------


def to_relative_posix(path: Path) -> str:
    return path.relative_to(cfg.PROJECT_ROOT).as_posix()


def load_test_records() -> list[dict]:
    split_index = json.loads(cfg.SPLIT_INDEX_PATH.read_text(encoding="utf-8"))
    label_map = json.loads(cfg.LABEL_MAP_PATH.read_text(encoding="utf-8"))
    labels = {entry["breed"]: entry for entry in label_map["classes"]}

    records = []
    for image_id in sorted(split_index["test"]):
        breed = image_id.rsplit("_", 1)[0]
        try:
            label = labels[breed]
        except KeyError as error:
            raise ValueError(
                f"Breed {breed!r} for test image {image_id!r} is missing from "
                f"the label map: {cfg.LABEL_MAP_PATH}"
            ) from error

        records.append(
            {
                "image_id": image_id,
                "path": cfg.IMAGES_DIR / f"{image_id}.jpg",
                "breed": breed,
                "species": label["species"],
                "class_index": label["index"],
                "split": "test",
            }
        )
    return records


def sample_per_class(records: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministic per-class sample, independent of file order on disk."""
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_class[r["breed"]].append(r)

    sampled = []
    for breed in sorted(by_class):
        items = sorted(by_class[breed], key=lambda r: r["image_id"])
        sampled.extend(items if len(items) <= n else rng.sample(items, n))
    return sampled


def resolve_num_workers() -> int:
    if cfg.NUM_WORKERS is not None:
        return max(1, cfg.NUM_WORKERS)
    return max(1, (os.cpu_count() or 2) - 1)


# ---------- Worker (must be a top-level function so it can be pickled) ----------

def process_record(record: dict) -> list[dict]:
    """Apply every corruption at every severity to one source image.
    Runs inside a worker process; returns the manifest records it produced."""
    with Image.open(record["path"]) as src:
        img = src.convert("RGB")

    produced = []
    for corruption_name, corruption_fn in CORRUPTIONS.items():
        for severity in SEVERITIES:
            corrupted = corruption_fn(img, severity)

            out_dir = (
                cfg.CORRUPTED_DIR / corruption_name
                / f"severity_{severity}" / record["breed"]
            )
            out_dir.mkdir(parents=True, exist_ok=True)  # safe under concurrent workers
            out_path = out_dir / f"{record['image_id']}.jpg"
            corrupted.save(out_path, format="JPEG", quality=95)

            produced.append({
                "image_id": record["image_id"],
                "path": to_relative_posix(out_path),
                "breed": record["breed"],
                "species": record["species"],
                "class_index": record["class_index"],
                "split": "test",
                "corruption": corruption_name,
                "severity": severity,
                "width": corrupted.width,
                "height": corrupted.height,
            })
    return produced


# ---------- Main generation ----------

def generate_corrupted_set(test_records: list[dict], num_workers: int) -> list[dict]:
    output_records = []
    total_sources = len(test_records)
    chunksize = max(1, total_sources // (num_workers * 20))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # executor.map preserves input order -> deterministic manifest
        for i, produced in enumerate(
            executor.map(process_record, test_records, chunksize=chunksize), start=1
        ):
            output_records.extend(produced)
            if i % 200 == 0 or i == total_sources:
                print(f"  {i}/{total_sources} source images processed")

    return output_records


def main() -> None:
    test_records = load_test_records()
    print(f"Loaded {len(test_records)} test images.")

    if cfg.CORRUPTION_SAMPLE_PER_CLASS is not None:
        test_records = sample_per_class(
            test_records, cfg.CORRUPTION_SAMPLE_PER_CLASS, cfg.SEED
        )
        print(f"Sampled {len(test_records)} images "
              f"({cfg.CORRUPTION_SAMPLE_PER_CLASS} per class).")

    num_workers = resolve_num_workers()
    total = len(test_records) * len(CORRUPTIONS) * len(SEVERITIES)
    print(f"Generating {total} corrupted images "
          f"({len(CORRUPTIONS)} corruptions x {len(SEVERITIES)} severities) "
          f"using {num_workers} worker processes.")

    start = time.perf_counter()
    output_records = generate_corrupted_set(test_records, num_workers)
    elapsed = time.perf_counter() - start

    print(f"\nGenerated {len(output_records)} corrupted images")
    print(f"Elapsed: {elapsed:.1f}s ({len(output_records) / elapsed:.1f} images/s)")

    print("Sample severity 3 paths:")
    sample_paths = {
        record["corruption"]: record["path"]
        for record in output_records
        if record["severity"] == 3
    }
    for corruption_name in CORRUPTIONS:
        print(f"  {corruption_name}: {sample_paths[corruption_name]}")


if __name__ == "__main__":
    main()
