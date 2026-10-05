import json
from pathlib import Path

import pytest
from PIL import Image

from pet_breed_classification.predict import PetBreedPredictor, Prediction


def test_predict_one_batch_and_timing(
    predictor: PetBreedPredictor,
    caplog: pytest.LogCaptureFixture,
) -> None:
    image = Image.new("RGBA", (301, 197), (20, 80, 140, 200))

    caplog.set_level("DEBUG")
    one = predictor.predict_one(image)
    batch = predictor.predict_batch([image])[0]

    assert isinstance(one, Prediction)
    assert len(one.top_3) == 3
    assert one.top_3 == sorted(one.top_3, key=lambda item: item[1], reverse=True)
    assert all(0 <= probability <= 1 for _, probability in one.top_3)
    assert abs(one.confidence - batch.confidence) < 1e-5
    assert one.breed == batch.breed
    assert one.top_3 == batch.top_3
    assert predictor.predict_one.__name__ == "predict_one"
    assert "predict_one" in caplog.text
    assert "ms" in caplog.text


def test_predict_is_deterministic(predictor: PetBreedPredictor) -> None:
    image = Image.new("RGB", (301, 197), (20, 80, 140))

    assert predictor.predict_one(image) == predictor.predict_one(image)


def test_predict_rejects_sha256_mismatch(
    onnx_artifacts: dict[str, Path],
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "bad_meta.json"
    metadata = json.loads(onnx_artifacts["meta"].read_text(encoding="utf-8"))
    metadata["onnx_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        PetBreedPredictor.load(onnx_artifacts["onnx"], metadata_path)
