from __future__ import annotations

from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pet_breed_classification.export import export
from pet_breed_classification.model import PetBreedClassifier, save_checkpoint
from pet_breed_classification.predict import PetBreedPredictor


def _classes() -> list[dict[str, object]]:
    return [
        {"index": index, "breed": f"breed_{index}", "species": "dog"}
        for index in range(3)
    ]


@pytest.fixture
def sample_image() -> Image.Image:
    """Return a small non-square RGB image."""
    return Image.new("RGB", (100, 80), (20, 80, 140))


@pytest.fixture(scope="session")
def onnx_artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Create one session-scoped random checkpoint and ONNX export."""
    directory = tmp_path_factory.mktemp("onnx-artifacts")
    checkpoint_path = directory / "model.pt"
    onnx_path = directory / "model.onnx"
    meta_path = directory / "model_meta.json"
    model = PetBreedClassifier(num_classes=3, pretrained=False)
    save_checkpoint(model, checkpoint_path, _classes(), 42, 1, {})
    export(checkpoint_path, onnx_path, meta_path)
    return {"checkpoint": checkpoint_path, "onnx": onnx_path, "meta": meta_path}


@pytest.fixture
def predictor(onnx_artifacts: dict[str, Path]) -> PetBreedPredictor:
    """Load the session-scoped ONNX artifacts."""
    return PetBreedPredictor.load(onnx_artifacts["onnx"], onnx_artifacts["meta"])


@pytest.fixture
def client(predictor: PetBreedPredictor, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Create a TestClient whose lifespan loads the fixture predictor."""
    from pet_breed_classification.api import main as api_main

    monkeypatch.setattr(
        api_main.PetBreedPredictor,
        "load",
        classmethod(lambda cls, *args, **kwargs: predictor),
    )
    with TestClient(api_main.app) as test_client:
        yield test_client


def image_bytes(image: Image.Image, image_format: str = "PNG") -> bytes:
    """Encode a PIL image for multipart API tests."""
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()
