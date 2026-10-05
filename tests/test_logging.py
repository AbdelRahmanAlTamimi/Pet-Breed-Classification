import io
import json
import logging

from fastapi.testclient import TestClient
from PIL import Image

from pet_breed_classification.logging_conf import JsonFormatter, correlation_id_var


def test_json_formatter_includes_required_fields_and_extra() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("logging-test")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    token = correlation_id_var.set("logging-correlation")
    try:
        logger.info("prediction", extra={"latency_ms": 1.5})
    finally:
        correlation_id_var.reset(token)
        logger.removeHandler(handler)

    record = json.loads(stream.getvalue())
    assert {
        "timestamp",
        "level",
        "logger",
        "message",
        "correlation_id",
    }.issubset(record)
    assert record["correlation_id"] == "logging-correlation"
    assert record["latency_ms"] == 1.5


def test_request_prediction_and_response_share_correlation_id(
    client: TestClient,
    sample_image: Image.Image,
    image_bytes,
) -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        response = client.post(
            "/predict",
            files={"file": ("pet.png", image_bytes(sample_image), "image/png")},
        )
    finally:
        root.removeHandler(handler)

    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    correlation_id = response.json()["correlation_id"]
    prediction_log = next(
        record for record in records if record["message"] == "Prediction served"
    )
    request_log = next(
        record for record in records if record["message"] == "Request completed"
    )
    assert prediction_log["correlation_id"] == correlation_id
    assert request_log["correlation_id"] == correlation_id
    assert response.headers["x-request-id"] == correlation_id
