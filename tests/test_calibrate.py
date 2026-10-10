"""Hermetic tests for the calibrate command: report files, metadata, sanity inputs.

Real artifacts are used only by the skipped-when-absent hash check at the end.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from pet_breed_classification import calibrate
from pet_breed_classification.calibration import softmax
from pet_breed_classification.config import cfg
from pet_breed_classification.predict import PetBreedPredictor


def _synthetic_validation(
    samples: int = 300, classes: int = 4, seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(samples, classes)) * 2.0
    cumulative = softmax(logits).cumsum(axis=1)
    labels = (cumulative > rng.random((samples, 1))).argmax(axis=1).astype(np.int64)
    return logits, labels


def _meta_file(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "model_version": "v1",
                "onnx_sha256": "abc",
                "temperature": 1.0,
                "abstain_threshold": None,
                "backbone": "resnet50",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_fit_from_logits_writes_report_figure_and_metadata(tmp_path: Path) -> None:
    logits, labels = _synthetic_validation()
    meta_path = _meta_file(tmp_path / "model_meta.json")

    result = calibrate.fit_from_logits(
        logits,
        labels,
        model_version="v1",
        onnx_sha256="abc",
        target=0.9,
        reports_dir=tmp_path / "reports",
        meta_path=meta_path,
        seed=42,
    )

    assert result.before["top1"] == result.after["top1"]
    assert result.before["macro_f1"] == result.after["macro_f1"]
    assert result.figure_path.stat().st_size > 0
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["split"] == "val"
    assert report["temperature"] == pytest.approx(result.temperature)
    assert [row["target"] for row in report["threshold_table"]] == [0.9, 0.95, 0.98]
    chosen = [row for row in report["threshold_table"] if row["chosen"]]
    assert [row["target"] for row in chosen] == [0.9]
    assert "in_sample" in report["two_fold_check"]
    assert report["two_fold_check"]["seed"] == 42
    assert set(report["two_fold_check"]["gap_in_sample_minus_held_out"]) == {
        "coverage",
        "selective_accuracy",
    }
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["temperature"] == pytest.approx(result.temperature)
    assert meta["onnx_sha256"] == "abc"
    assert meta["backbone"] == "resnet50"
    if result.selection.threshold is None:
        assert meta["abstain_threshold"] is None
    else:
        assert meta["abstain_threshold"] == pytest.approx(result.selection.threshold)


def test_unattainable_target_records_null_and_never_a_made_up_threshold(
    tmp_path: Path,
) -> None:
    logits, _labels = _synthetic_validation()
    wrong_labels = (logits.argmax(axis=1) + 1) % logits.shape[1]
    meta_path = _meta_file(tmp_path / "model_meta.json")

    result = calibrate.fit_from_logits(
        logits,
        wrong_labels.astype(np.int64),
        model_version="v1",
        onnx_sha256="abc",
        target=0.95,
        reports_dir=tmp_path / "reports",
        meta_path=meta_path,
    )

    assert result.selection.threshold is None
    assert "unattainable" in result.selection.reason
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["abstain_threshold"] is None
    assert meta["temperature"] == pytest.approx(result.temperature)


def test_two_fold_swap_reports_each_half_and_pooled_held_out(tmp_path: Path) -> None:
    logits, labels = _synthetic_validation(samples=200)

    check = calibrate.two_fold_swap(logits, labels, target=0.5, seed=42)

    folds = check["folds"]
    assert check["seed"] == 42
    assert [fold["fold"] for fold in folds] == [0, 1]
    assert sum(fold["eval_size"] for fold in folds) == 200
    assert all(fold["fit_size"] + fold["eval_size"] == 200 for fold in folds)
    pooled = check["held_out_pooled"]
    assert 0.0 <= pooled["coverage"] <= 1.0
    assert (
        pooled["selective_accuracy"] is None
        or 0.0 <= pooled["selective_accuracy"] <= 1.0
    )


def test_two_fold_swap_pools_to_none_when_a_half_is_unattainable() -> None:
    logits, _labels = _synthetic_validation(samples=200)
    wrong = ((logits.argmax(axis=1) + 1) % logits.shape[1]).astype(np.int64)

    check = calibrate.two_fold_swap(logits, wrong, target=0.95, seed=42)

    assert check["held_out_pooled"] == {"coverage": None, "selective_accuracy": None}
    assert all(fold["threshold"] is None for fold in check["folds"])


def test_plot_reliability_writes_a_png(tmp_path: Path) -> None:
    logits, labels = _synthetic_validation()
    probabilities = softmax(logits)
    bins = calibrate.reliability_bins(probabilities, labels, 15)
    path = tmp_path / "reliability.png"

    calibrate.plot_reliability(bins, bins, 0.01, 0.01, 1.0, path)

    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_synthetic_inputs_cover_every_category_and_decode_correctly() -> None:
    categories = calibrate.synthetic_inputs()

    assert set(categories) == {
        "uniform noise",
        "solid colours",
        "gradients",
        "checkerboards",
        "RGBA PNG (4 channels)",
        "greyscale",
    }
    assert len(categories["uniform noise"]) == 20
    assert len(categories["RGBA PNG (4 channels)"]) == 5
    assert len(categories["greyscale"]) == 5
    rgba = Image.open(_bytes_io(categories["RGBA PNG (4 channels)"][0][1]))
    assert rgba.mode == "RGBA"
    grey = Image.open(_bytes_io(categories["greyscale"][0][1]))
    assert grey.mode == "L"
    assert categories == calibrate.synthetic_inputs()


def _bytes_io(contents: bytes) -> Any:
    import io

    return io.BytesIO(contents)


def test_owner_inputs_skip_a_missing_directory_and_keep_only_images(
    tmp_path: Path,
) -> None:
    assert calibrate.owner_inputs(tmp_path / "absent") == []

    (tmp_path / "photo.jpg").write_bytes(b"not decoded here")
    (tmp_path / "notes.txt").write_text("skip me", encoding="utf-8")

    owned = calibrate.owner_inputs(tmp_path)

    assert len(owned) == 1
    assert owned[0][1] == b"not decoded here"


def test_summarize_responses_counts_decisions_and_statuses() -> None:
    summary = calibrate.summarize_responses(
        [(200, "confident", 0.9), (200, "uncertain", 0.4), (500, None, None)]
    )

    assert summary["count"] == 3
    assert summary["confident"] == 1
    assert summary["uncertain"] == 1
    assert summary["confidence_min"] == pytest.approx(0.4)
    assert summary["confidence_max"] == pytest.approx(0.9)
    assert summary["statuses"] == {200: 2, 500: 1}


def test_sanity_runs_through_the_route_and_reports_no_server_errors(
    predictor: PetBreedPredictor,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        PetBreedPredictor, "load", classmethod(lambda cls, *args, **kwargs: predictor)
    )

    exit_code = calibrate.run_sanity(tmp_path / "absent")

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "uniform noise" in output
    assert "200x" in output
    assert "cannot guarantee out-of-distribution rejection" in output


def test_run_fit_writes_outputs_and_reports_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logits, labels = _synthetic_validation()
    meta_path = _meta_file(tmp_path / "model_meta.json")
    stub = SimpleNamespace(metadata={"model_version": "v1", "onnx_sha256": "abc"})
    monkeypatch.setattr(
        PetBreedPredictor, "load", classmethod(lambda cls, *args, **kwargs: stub)
    )
    monkeypatch.setattr(
        calibrate, "validation_logits", lambda predictor: (logits, labels)
    )
    monkeypatch.setattr(
        calibrate,
        "cfg",
        cfg.model_copy(
            update={
                "ARTIFACTS_DIR": tmp_path,
                "REPORTS_DIR": tmp_path / "reports",
                "MODEL_META_PATH": meta_path,
            }
        ),
    )

    exit_code = calibrate.run_fit(0.5)

    assert exit_code in {0, 1}
    assert (tmp_path / "reports" / "calibration.json").is_file()
    assert (tmp_path / "reports" / "calibration.png").is_file()
    assert "temperature T =" in capsys.readouterr().out


def test_validation_logits_uses_only_val_records_and_does_not_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_a = Image.new("RGBA", (32, 40), (10, 20, 30, 128))
    image_b = Image.new("L", (40, 32), 90)
    image_a.save(tmp_path / "a.png")
    image_b.save(tmp_path / "b.png")
    records = [
        {"path": "a.png", "class_index": 1},
        {"path": "b.png", "class_index": 0},
    ]
    requested_splits: list[str] = []
    batch_sizes: list[int] = []

    class StubPredictor:
        def raw_logits(self, images: list[Image.Image]) -> np.ndarray:
            batch_sizes.append(len(images))
            return np.tile(np.array([[1.0, 2.0, 0.0]]), (len(images), 1))

    monkeypatch.setattr(
        calibrate,
        "cfg",
        cfg.model_copy(update={"PROJECT_ROOT": tmp_path}),
    )
    monkeypatch.setattr(
        calibrate,
        "load_records",
        lambda split: requested_splits.append(split) or records,
    )

    first_logits, first_labels = calibrate.validation_logits(StubPredictor())
    second_logits, second_labels = calibrate.validation_logits(StubPredictor())

    assert requested_splits == ["val", "val"]
    assert batch_sizes == [2, 2]
    assert np.array_equal(first_logits, second_logits)
    assert np.array_equal(first_labels, np.array([1, 0], dtype=np.int64))
    assert np.array_equal(second_labels, first_labels)


def test_metadata_and_unrelated_artifact_bytes_are_preserved(
    tmp_path: Path,
) -> None:
    logits, labels = _synthetic_validation(samples=120)
    meta_path = _meta_file(tmp_path / "model_meta.json")
    onnx_path = tmp_path / "model.onnx"
    weights_path = tmp_path / "model.pt"
    onnx_bytes = b"synthetic-onnx-bytes"
    weights_bytes = b"synthetic-weight-bytes"
    onnx_path.write_bytes(onnx_bytes)
    weights_path.write_bytes(weights_bytes)
    metadata_before = json.loads(meta_path.read_text(encoding="utf-8"))

    result = calibrate.fit_from_logits(
        logits,
        labels,
        model_version="v1",
        onnx_sha256="abc",
        target=0.9,
        reports_dir=tmp_path / "reports",
        meta_path=meta_path,
    )

    metadata_after = json.loads(meta_path.read_text(encoding="utf-8"))
    assert metadata_after.keys() == metadata_before.keys()
    assert metadata_after["onnx_sha256"] == metadata_before["onnx_sha256"]
    assert metadata_after["temperature"] == pytest.approx(result.temperature)
    assert onnx_path.read_bytes() == onnx_bytes
    assert weights_path.read_bytes() == weights_bytes
    assert not list(tmp_path.glob(".*.tmp"))


def test_main_dispatches_fit_and_sanity(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(calibrate, "setup_logging", lambda: None)
    monkeypatch.setattr(
        calibrate, "run_fit", lambda target: calls.append(("fit", target)) or 0
    )
    monkeypatch.setattr(
        calibrate,
        "run_sanity",
        lambda directory: calls.append(("sanity", directory)) or 0,
    )

    assert calibrate.main(["fit", "--target-selective-accuracy", "0.9"]) == 0
    assert calibrate.main(["sanity"]) == 0

    assert calls[0] == ("fit", 0.9)
    assert calls[1][0] == "sanity"


@pytest.mark.skipif(
    not cfg.ONNX_PATH.is_file() or not cfg.MODEL_META_PATH.is_file(),
    reason="real ONNX artifact not present",
)
def test_real_model_meta_records_the_onnx_file_hash() -> None:
    metadata = json.loads(cfg.MODEL_META_PATH.read_text(encoding="utf-8"))

    actual = hashlib.sha256(cfg.ONNX_PATH.read_bytes()).hexdigest()

    assert metadata["onnx_sha256"] == actual


def test_exported_metadata_records_the_onnx_file_hash(
    onnx_artifacts: dict[str, Path],
) -> None:
    metadata = json.loads(onnx_artifacts["meta"].read_text(encoding="utf-8"))

    actual = hashlib.sha256(onnx_artifacts["onnx"].read_bytes()).hexdigest()

    assert metadata["onnx_sha256"] == actual
