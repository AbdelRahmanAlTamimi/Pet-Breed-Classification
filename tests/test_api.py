from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pet_breed_classification.api import main as api_main
from pet_breed_classification.api.schemas import PredictionResponse
from pet_breed_classification.config import Config


def image_bytes(image: Image.Image, image_format: str = "PNG") -> bytes:
    """Encode a PIL image for multipart API tests."""
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def test_health_and_metadata(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    health = client.get("/health")
    metadata = client.get("/metadata")

    assert health.status_code == 200
    assert health.json()["status"] == "healthy"
    assert metadata.status_code == 200
    assert metadata.json()["framework"] == "onnxruntime"
    assert metadata.json()["num_classes"] == 3


def test_health_returns_503_when_predictor_is_missing(client: TestClient) -> None:
    client.app.state.predictor = None

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["correlation_id"] == response.headers["x-request-id"]


def test_predict_happy_path_and_correlation(
    client: TestClient,
    sample_image: Image.Image,
) -> None:
    response = client.post(
        "/predict",
        files={"file": ("pet.png", image_bytes(sample_image), "image/png")},
    )

    assert response.status_code == 200
    parsed = PredictionResponse.model_validate(response.json())
    assert parsed.model_version == "v1"
    assert parsed.correlation_id == response.headers["x-request-id"]


@pytest.mark.parametrize(
    ("filename", "contents"),
    [("text.txt", b"this is not an image"), ("empty.txt", b"")],
)
def test_predict_rejects_invalid_payloads(
    client: TestClient,
    filename: str,
    contents: bytes,
) -> None:
    response = client.post("/predict", files={"file": (filename, contents)})

    assert response.status_code == 422
    assert response.json()["detail"][0]["message"]


def test_predict_rejects_oversized_payload(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_main, "cfg", Config(MAX_UPLOAD_BYTES=10))

    response = client.post("/predict", files={"file": ("large.bin", b"x" * 11)})

    assert response.status_code == 422
    assert "exceeds" in response.json()["detail"][0]["message"]


def test_predict_rejects_missing_file(client: TestClient) -> None:
    response = client.post("/predict")

    assert response.status_code == 422
    assert response.json()["detail"][0]["field"] == "file"


def test_predict_accepts_rgba(client: TestClient) -> None:
    image = Image.new("RGBA", (80, 60), (20, 80, 140, 100))

    response = client.post(
        "/predict",
        files={"file": ("rgba.png", image_bytes(image), "image/png")},
    )

    assert response.status_code == 200


def test_predict_rejects_decompression_bomb(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_bomb(*args, **kwargs):
        raise Image.DecompressionBombError("bomb")

    monkeypatch.setattr(api_main.Image, "open", reject_bomb)
    response = client.post(
        "/predict",
        files={"file": ("bomb.png", b"not-used", "image/png")},
    )

    assert response.status_code == 422
    assert "decompression-bomb" in response.json()["detail"][0]["message"]


def test_batch_predict_and_limit(
    client: TestClient,
    sample_image: Image.Image,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contents = image_bytes(sample_image)
    response = client.post(
        "/predict/batch",
        files=[
            ("files", ("first.png", contents, "image/png")),
            ("files", ("second.png", contents, "image/png")),
        ],
    )
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2

    monkeypatch.setattr(api_main, "cfg", Config(MAX_BATCH_FILES=1))
    limited = client.post(
        "/predict/batch",
        files=[
            ("files", ("first.png", contents, "image/png")),
            ("files", ("second.png", contents, "image/png")),
        ],
    )
    assert limited.status_code == 422
    assert "files" in limited.json()["detail"][0]["field"]


def test_unexpected_error_is_safe(
    predictor,
    sample_image: Image.Image,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(image: Image.Image):
        raise RuntimeError("secret traceback")

    monkeypatch.setattr(predictor, "predict_one", fail)
    with TestClient(api_main.app, raise_server_exceptions=False) as error_client:
        error_client.app.state.predictor = predictor
        response = error_client.post(
            "/predict",
            files={"file": ("pet.png", image_bytes(sample_image), "image/png")},
        )

    assert response.status_code == 500
    assert "secret traceback" not in response.text
    assert response.headers["x-request-id"] == response.json()["correlation_id"]
