from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("bentoml")

# The serving entry point is a source file rather than an installed package.
sys.path.insert(0, str(Path(__file__).parents[1]))

from pet_breed_classification.config import cfg
from pet_breed_classification.predict import PetBreedPredictor
from serving.service import PetBreedRunnable, _decode


def test_bento_input_validation_and_rgba() -> None:
    image = Image.new("RGBA", (32, 32), (10, 20, 30, 128))
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    assert _decode(encoded.getvalue()).mode == "RGB"

    with pytest.raises(Exception, match="valid image"):
        _decode(b"not an image")
    with pytest.raises(Exception, match="exceeds"):
        _decode(b"x" * (cfg.MAX_UPLOAD_BYTES + 1))


def test_bento_predict_parity_on_validation_images() -> None:
    if not cfg.ONNX_PATH.is_file() or not cfg.MODEL_META_PATH.is_file():
        pytest.skip("real model artifacts are unavailable")

    split_index = json.loads(cfg.SPLIT_INDEX_PATH.read_text(encoding="utf-8"))
    validation_ids = [str(image_id) for image_id in split_index["val"][:50]]
    predictor = PetBreedPredictor.load()
    runnable = PetBreedRunnable()

    for image_id in validation_ids:
        image_path = cfg.IMAGES_DIR / f"{image_id}.jpg"
        with Image.open(image_path) as image:
            expected = predictor.predict_one(image.convert("RGB"))
        actual = runnable.predict([image_path])[0]
        assert actual["breed"] == expected.breed
        assert actual["decision"] == expected.decision
        assert actual["confidence"] == pytest.approx(expected.confidence, abs=1e-6)
        assert [item["breed"] for item in actual["top_3"]] == [
            item[0] for item in expected.top_3
        ]
        assert [item["probability"] for item in actual["top_3"]] == pytest.approx(
            [item[1] for item in expected.top_3], abs=1e-6
        )


def test_bento_batch_matches_single_predictions() -> None:
    if not cfg.ONNX_PATH.is_file() or not cfg.MODEL_META_PATH.is_file():
        pytest.skip("real model artifacts are unavailable")

    split_index = json.loads(cfg.SPLIT_INDEX_PATH.read_text(encoding="utf-8"))
    paths = [cfg.IMAGES_DIR / f"{image_id}.jpg" for image_id in split_index["val"][:8]]
    runnable = PetBreedRunnable()

    batched = runnable.predict(paths)
    singles = [runnable.predict([path])[0] for path in paths]
    assert len(batched) == len(paths)
    for batch_item, single_item in zip(batched, singles, strict=True):
        assert batch_item["breed"] == single_item["breed"]
        assert batch_item["decision"] == single_item["decision"]
        assert batch_item["confidence"] == pytest.approx(single_item["confidence"], abs=1e-5)
