import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from PIL import Image

from pet_breed_classification.config import cfg
from pet_breed_classification.data import load_records
from pet_breed_classification.export import export
from pet_breed_classification.model import (
    PetBreedClassifier,
    load_checkpoint,
    save_checkpoint,
)
from pet_breed_classification.predict import PetBreedPredictor
from pet_breed_classification.transforms import build_eval_transform


def classes() -> list[dict[str, object]]:
    return [
        {"index": index, "breed": f"breed_{index}", "species": "dog"}
        for index in range(3)
    ]


def test_random_checkpoint_pytorch_onnx_parity(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    onnx_path = tmp_path / "model.onnx"
    meta_path = tmp_path / "model_meta.json"
    model = PetBreedClassifier(num_classes=3, pretrained=False)
    save_checkpoint(model, checkpoint_path, classes(), 42, 1, {})
    export(checkpoint_path, onnx_path, meta_path)
    loaded_model, _ = load_checkpoint(checkpoint_path, torch.device("cpu"))

    inputs = np.random.default_rng(42).standard_normal((2, 3, 224, 224)).astype(
        np.float32
    )
    with torch.inference_mode():
        torch_logits = loaded_model(torch.from_numpy(inputs)).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(["logits"], {"images": inputs})[0]

    assert np.allclose(torch_logits, onnx_logits, atol=1e-4)
    assert np.array_equal(torch_logits.argmax(axis=1), onnx_logits.argmax(axis=1))


@pytest.mark.skipif(
    not cfg.MODEL_PATH.is_file()
    or not cfg.ONNX_PATH.is_file()
    or not cfg.MODEL_META_PATH.is_file()
    or not cfg.MANIFEST_PATH.is_file()
    or not cfg.IMAGES_DIR.is_dir(),
    reason="real checkpoint, ONNX artifacts, or validation manifest unavailable",
)
def test_real_validation_parity() -> None:
    records = load_records("val")[:200]
    model, checkpoint = load_checkpoint(cfg.MODEL_PATH, torch.device("cpu"))
    checkpoint_transform = build_eval_transform(checkpoint["eval_transform"])
    predictor = PetBreedPredictor.load(cfg.ONNX_PATH, cfg.MODEL_META_PATH)
    images = []
    checkpoint_batches = []
    predictor_batches = []
    for record in records:
        with Image.open(cfg.PROJECT_ROOT / str(record["path"])) as image:
            image_copy = image.copy()
        images.append(image_copy)
        checkpoint_batches.append(checkpoint_transform(image_copy).numpy())
        predictor_batches.append(predictor.transform(image_copy).numpy())
    checkpoint_inputs = np.stack(checkpoint_batches).astype(np.float32, copy=False)
    predictor_inputs = np.stack(predictor_batches).astype(np.float32, copy=False)

    with torch.inference_mode():
        torch_probabilities = torch.softmax(
            model(torch.from_numpy(checkpoint_inputs)) / predictor.temperature,
            dim=1,
        ).numpy()
    onnx_logits = predictor.session.run(
        ["logits"], {predictor.input_name: predictor_inputs}
    )[0]
    scaled_logits = onnx_logits / float(predictor.temperature)
    scaled_logits -= np.max(scaled_logits, axis=1, keepdims=True)
    onnx_probabilities = np.exp(scaled_logits)
    onnx_probabilities /= np.sum(onnx_probabilities, axis=1, keepdims=True)

    metadata = json.loads(cfg.MODEL_META_PATH.read_text(encoding="utf-8"))
    assert metadata["framework"] == "onnxruntime"
    assert np.array_equal(
        torch_probabilities.argmax(axis=1), onnx_probabilities.argmax(axis=1)
    )
    assert float(np.max(np.abs(torch_probabilities - onnx_probabilities))) < 1e-4
    assert len(predictor.predict_batch(images)) == 200
