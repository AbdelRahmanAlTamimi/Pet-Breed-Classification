import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from pet_breed_classification.model import CHECKPOINT_KEYS, load_checkpoint
from pet_breed_classification.predict import PetBreedPredictor, Prediction
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
            model(inputs) / float(checkpoint["temperature"]), dim=1
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
