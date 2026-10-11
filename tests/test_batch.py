from __future__ import annotations

import csv
from pathlib import Path
from typing import ClassVar

from pet_breed_classification.batch import score_records
from pet_breed_classification.predict import Prediction


class FakePredictor:
    metadata: ClassVar[dict[str, str]] = {"model_version": "test"}

    def predict_batch(self, images):
        return [
            Prediction(
                breed="breed",
                species="dog",
                confidence=0.75,
                top_3=[("breed", 0.75), ("other", 0.2), ("third", 0.05)],
                decision="confident",
            )
            for _ in images
        ]


def test_score_records_is_hermetic(tmp_path: Path) -> None:
    from PIL import Image

    records = []
    for index in range(3):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (32, 32), (index, 20, 30)).save(path)
        records.append({"image_id": f"image-{index}", "path": path, "breed": "breed"})

    output_path = tmp_path / "output" / "scores.csv"
    result = score_records(records, FakePredictor(), output_path, batch_size=2)

    assert result["images_scored"] == 3
    assert result["row_count"] == 3
    with output_path.open(newline="", encoding="utf-8") as output:
        rows = list(csv.DictReader(output))
    assert len(rows) == 3
    assert rows[0]["model_version"] == "test"
