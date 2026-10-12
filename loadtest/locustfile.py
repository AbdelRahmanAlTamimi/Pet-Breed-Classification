"""Locust baseline shared by the FastAPI and BentoML services."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from locust import HttpUser, between, task

from pet_breed_classification.config import cfg

MINIMUM_DISTINCT_IMAGES = 200
_START_OFFSETS = itertools.count()


def _validation_images() -> list[tuple[str, Path]]:
    """Load only validation IDs from the committed split index."""
    split_index = json.loads(cfg.SPLIT_INDEX_PATH.read_text(encoding="utf-8"))
    records = []
    for image_id in split_index.get("val", []):
        path = cfg.IMAGES_DIR / f"{image_id}.jpg"
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append((str(image_id), path))
    if len(records) < MINIMUM_DISTINCT_IMAGES:
        raise RuntimeError(
            f"Locust needs at least {MINIMUM_DISTINCT_IMAGES} validation images; found {len(records)}"
        )
    return records


class PetBreedUser(HttpUser):
    """One Locust user issuing 90% predictions and 10% health checks."""

    wait_time = between(1, 3)
    validation_images = _validation_images()

    def on_start(self) -> None:
        self.image_index = next(_START_OFFSETS) % len(self.validation_images)

    @task(9)
    def predict(self) -> None:
        image_id, path = self.validation_images[self.image_index]
        self.image_index = (self.image_index + 1) % len(self.validation_images)
        with path.open("rb") as image:
            self.client.post(
                "/predict",
                files={"file": (f"{image_id}.jpg", image, "image/jpeg")},
                name="/predict",
            )

    @task(1)
    def health(self) -> None:
        self.client.get("/health", name="/health")
