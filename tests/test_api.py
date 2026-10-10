from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pet_breed_classification import predict as predict_module
from pet_breed_classification.api import main as api_main
from pet_breed_classification.api.schemas import PredictionResponse
from pet_breed_classification.config import Config
from pet_breed_classification.predict import UncalibratedModelError


def test_health_and_metadata(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    health = client.get("/health")
    metadata = client.get("/metadata")

    assert health.status_code == 200
    assert health.json()["status"] == "healthy"
    assert metadata.status_code == 200
    assert metadata.json()["framework"] == "onnxruntime"
    assert metadata.json()["num_classes"] == 3


def test_health_returns_503_when_predictor_is_missing(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client.app.state.predictor = None
    caplog.set_level("WARNING")

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["correlation_id"] == response.headers["x-request-id"]
    assert any(
        record.message == "HTTP error"
        and record.status == 503
        and record.levelname == "ERROR"
        for record in caplog.records
    )


def test_not_found_uses_http_error_shape(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING")

    response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.json()["detail"] == "Not Found"
    assert response.json()["correlation_id"] == response.headers["x-request-id"]
    assert any(
        record.message == "HTTP error"
        and record.status == 404
        and record.levelname == "WARNING"
        for record in caplog.records
    )


def test_predict_happy_path_and_correlation(
    client: TestClient,
    sample_image: Image.Image,
    image_bytes,
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


def test_predict_accepts_rgba(client: TestClient, image_bytes) -> None:
    image = Image.new("RGBA", (80, 60), (20, 80, 140, 100))

    response = client.post(
        "/predict",
        files={"file": ("rgba.png", image_bytes(image), "image/png")},
    )

    assert response.status_code == 200


def test_predict_accepts_grayscale(client: TestClient, image_bytes) -> None:
    image = Image.new("L", (80, 60), 120)

    response = client.post(
        "/predict",
        files={"file": ("grayscale.png", image_bytes(image), "image/png")},
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
    image_bytes,
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


def test_batch_predict_rejects_invalid_upload(client: TestClient) -> None:
    response = client.post(
        "/predict/batch",
        files=[("files", ("not-an-image.txt", b"not an image", "text/plain"))],
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["field"] == "files[0]"


def test_unexpected_error_is_safe(
    predictor,
    sample_image: Image.Image,
    image_bytes,
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


def test_metadata_reports_temperature_and_abstain_threshold(client: TestClient) -> None:
    metadata = client.get("/metadata").json()

    assert metadata["temperature"] == pytest.approx(1.0)
    loaded = client.app.state.predictor.abstain_threshold
    assert metadata["abstain_threshold"] == pytest.approx(loaded)


def test_predict_returns_a_decision_and_calibrated_probabilities(
    client: TestClient,
    sample_image: Image.Image,
    image_bytes,
) -> None:
    response = client.post(
        "/predict",
        files={"file": ("pet.png", image_bytes(sample_image), "image/png")},
    )

    body = response.json()
    threshold = client.app.state.predictor.abstain_threshold
    assert body["decision"] in {"confident", "uncertain"}
    assert body["decision"] == (
        "confident" if body["confidence"] >= threshold else "uncertain"
    )
    assert len(body["top_3"]) == 3
    assert body["top_3"][0]["breed"] == body["breed"]
    assert body["top_3"][0]["probability"] == pytest.approx(body["confidence"])


def test_batch_predict_returns_a_decision_per_image(
    client: TestClient,
    sample_image: Image.Image,
    image_bytes,
) -> None:
    contents = image_bytes(sample_image)
    response = client.post(
        "/predict/batch",
        files=[("files", ("a.png", contents, "image/png"))],
    )

    assert response.status_code == 200
    decisions = [item["decision"] for item in response.json()["predictions"]]
    assert decisions and set(decisions) <= {"confident", "uncertain"}


def test_startup_refuses_to_serve_an_uncalibrated_model(
    onnx_artifacts: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = json.loads(onnx_artifacts["meta"].read_text(encoding="utf-8"))
    metadata["abstain_threshold"] = None
    null_path = tmp_path / "null_meta.json"
    null_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(
        predict_module,
        "cfg",
        predict_module.cfg.model_copy(
            update={"ONNX_PATH": onnx_artifacts["onnx"], "MODEL_META_PATH": null_path}
        ),
    )

    with (
        pytest.raises(UncalibratedModelError, match="calibrate fit"),
        TestClient(api_main.app),
    ):
        pass
