import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from pet_breed_classification.calibration import softmax
from pet_breed_classification.model import CHECKPOINT_KEYS, load_checkpoint
from pet_breed_classification.predict import (
    PetBreedPredictor,
    Prediction,
    UncalibratedModelError,
)
from pet_breed_classification.transforms import (
    EVAL_TRANSFORM_METADATA,
    build_eval_transform,
)


def _parity_images() -> list[Image.Image]:
    palette = Image.new("P", (301, 197), 2)
    palette.putpalette([20, 80, 140, 200, 100, 40, 80, 160, 220] + [0] * 759)
    return [
        Image.new("RGB", (301, 197), (20, 80, 140)),
        Image.new("RGBA", (301, 197), (20, 80, 140, 200)),
        Image.new("L", (301, 197), 120),
        palette,
        Image.new("RGB", (40, 40), (80, 140, 210)),
    ]


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


def test_predict_rejects_malformed_sha256(
    onnx_artifacts: dict[str, Path],
    tmp_path: Path,
) -> None:
    metadata = json.loads(onnx_artifacts["meta"].read_text(encoding="utf-8"))
    metadata["onnx_sha256"] = "not-a-sha256-digest"
    metadata_path = tmp_path / "malformed_meta.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="64-character hexadecimal"):
        PetBreedPredictor.load(onnx_artifacts["onnx"], metadata_path)


def test_served_transform_and_probabilities_match_checkpoint(
    onnx_artifacts: dict[str, Path],
    predictor: PetBreedPredictor,
) -> None:
    model, checkpoint = load_checkpoint(onnx_artifacts["checkpoint"], torch.device("cpu"))
    metadata = json.loads(onnx_artifacts["meta"].read_text(encoding="utf-8"))
    assert CHECKPOINT_KEYS <= checkpoint.keys()
    assert metadata["eval_transform"] == checkpoint["eval_transform"]
    assert checkpoint["eval_transform"] == EVAL_TRANSFORM_METADATA

    checkpoint_transform = build_eval_transform(checkpoint["eval_transform"])
    default_transform = build_eval_transform()
    images = _parity_images()
    for image in images:
        assert torch.equal(predictor.transform(image), checkpoint_transform(image))
        assert torch.equal(predictor.transform(image), default_transform(image))

    inputs = torch.stack([checkpoint_transform(image) for image in images])
    with torch.inference_mode():
        probabilities = torch.softmax(
            model(inputs) / float(metadata["temperature"]), dim=1
        )
    reference_probabilities, reference_indices = torch.topk(probabilities, 3, dim=1)
    predictions = predictor.predict_batch(images)
    classes = {int(entry["index"]): entry for entry in checkpoint["classes"]}

    for prediction, row_probabilities, row_indices in zip(
        predictions, reference_probabilities, reference_indices, strict=True
    ):
        assert prediction.breed == classes[int(row_indices[0])]["breed"]
        for (breed, probability), reference_probability, reference_index in zip(
            prediction.top_3,
            row_probabilities,
            row_indices,
            strict=True,
        ):
            assert breed == classes[int(reference_index)]["breed"]
            assert abs(probability - float(reference_probability)) < 1e-4


def test_decision_uses_greater_or_equal_to_the_threshold(
    predictor: PetBreedPredictor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = Image.new("RGB", (120, 90), (40, 90, 160))
    top = predictor.predict_one(image).confidence

    monkeypatch.setattr(predictor, "abstain_threshold", top)
    assert predictor.predict_one(image).decision == "confident"

    monkeypatch.setattr(predictor, "abstain_threshold", math.nextafter(top, 2.0))
    assert predictor.predict_one(image).decision == "uncertain"


def test_probabilities_are_calibrated_softmax_rows(predictor: PetBreedPredictor) -> None:
    images = _parity_images()

    probabilities = predictor.probabilities(images)

    assert probabilities.shape == (len(images), 3)
    assert np.all(probabilities >= 0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    logits = predictor.raw_logits(images)
    assert np.allclose(probabilities, softmax(logits, predictor.temperature))


def test_temperature_is_applied_to_logits_outside_onnx(
    predictor: PetBreedPredictor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = Image.new("RGB", (120, 90), (40, 90, 160))
    logits = predictor.raw_logits([image])
    monkeypatch.setattr(predictor, "temperature", 2.0)

    probabilities = predictor.probabilities([image])

    assert np.allclose(probabilities, softmax(logits, 2.0))


def test_batch_and_single_predictions_agree_including_decision(
    predictor: PetBreedPredictor,
) -> None:
    images = _parity_images()

    batch = predictor.predict_batch(images)
    singles = [predictor.predict_one(image) for image in images]

    for single, grouped in zip(singles, batch, strict=True):
        assert single.breed == grouped.breed
        assert single.decision == grouped.decision
        assert abs(single.confidence - grouped.confidence) < 1e-6


def test_load_refuses_a_null_threshold(
    onnx_artifacts: dict[str, Path],
    tmp_path: Path,
) -> None:
    metadata = json.loads(onnx_artifacts["meta"].read_text(encoding="utf-8"))
    metadata["abstain_threshold"] = None
    null_path = tmp_path / "null_meta.json"
    null_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(UncalibratedModelError, match="calibrate fit"):
        PetBreedPredictor.load(onnx_artifacts["onnx"], null_path)

    loaded = PetBreedPredictor.load(
        onnx_artifacts["onnx"], null_path, require_calibration=False
    )
    assert loaded.abstain_threshold is None
    with pytest.raises(UncalibratedModelError):
        loaded.predict_one(Image.new("RGB", (40, 40)))


def test_load_rejects_non_positive_temperature(
    onnx_artifacts: dict[str, Path],
    tmp_path: Path,
) -> None:
    metadata = json.loads(onnx_artifacts["meta"].read_text(encoding="utf-8"))
    metadata["temperature"] = 0.0
    bad_path = tmp_path / "bad_temperature.json"
    bad_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="temperature"):
        PetBreedPredictor.load(onnx_artifacts["onnx"], bad_path)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), "0.5"])
def test_load_rejects_invalid_threshold(
    onnx_artifacts: dict[str, Path],
    tmp_path: Path,
    threshold: object,
) -> None:
    metadata = json.loads(onnx_artifacts["meta"].read_text(encoding="utf-8"))
    metadata["abstain_threshold"] = threshold
    bad_path = tmp_path / "bad_threshold.json"
    bad_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match="abstain_threshold"):
        PetBreedPredictor.load(onnx_artifacts["onnx"], bad_path)
