from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pet_breed_classification import calibrate, tracking, train
from pet_breed_classification.artifacts import create_run_artifacts
from pet_breed_classification.config import cfg
from pet_breed_classification.model import (
    SUPPORTED_BACKBONES,
    PetBreedClassifier,
    build_backbone,
    load_checkpoint,
    save_checkpoint,
)


def _classes(count: int = 3) -> list[dict[str, object]]:
    return [
        {"index": index, "breed": f"breed_{index}", "species": "dog"}
        for index in range(count)
    ]


def test_train_epochs_records_learning_rate_before_scheduler_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_epoch(
        model: nn.Module,
        loader: object,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        del model, loader, criterion, max_batches
        return {
            "loss": 1.0,
            "accuracy": 0.5 if optimizer is not None else 0.6,
            "macro_f1": 0.4 if optimizer is not None else 0.55,
        }

    monkeypatch.setattr(train, "run_epoch", fake_run_epoch)
    model = nn.Linear(1, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    class ToyScheduler:
        def step(self) -> None:
            optimizer.param_groups[0]["lr"] *= 0.5

    scheduler = ToyScheduler()

    result = train.train_epochs(
        model,
        [],
        [],
        nn.CrossEntropyLoss(),
        optimizer,
        scheduler,
        epochs=3,
    )

    assert [row["learning_rate"] for row in result.history] == [0.1, 0.05, 0.025]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0125)


@pytest.mark.parametrize("backbone", SUPPORTED_BACKBONES)
def test_supported_backbone_has_a_37_class_head(backbone: str) -> None:
    model = PetBreedClassifier(backbone=backbone, pretrained=False)

    if backbone.startswith("resnet"):
        assert model.backbone.fc.out_features == 37
    else:
        assert model.backbone.classifier[-1].out_features == 37


def test_backbone_factory_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported backbone"):
        build_backbone("unknown", 37, pretrained=False)


def test_checkpoint_preserves_non_default_backbone(tmp_path: Path) -> None:
    model = PetBreedClassifier(num_classes=3, backbone="mobilenet_v3_small")
    checkpoint_path = tmp_path / "model.pt"

    save_checkpoint(model, checkpoint_path, _classes(), 42, 1, {})
    loaded, checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))

    assert checkpoint["backbone"] == "mobilenet_v3_small"
    assert loaded.backbone_name == "mobilenet_v3_small"
    assert loaded.backbone.classifier[-1].out_features == 3


def test_run_artifacts_are_isolated_and_not_reused(tmp_path: Path) -> None:
    artifacts = create_run_artifacts("smoke", tmp_path / "artifacts")

    assert artifacts.directory == tmp_path / "artifacts" / "runs" / "smoke"
    assert artifacts.reports.is_dir()
    assert artifacts.onnx.name == "model.onnx"
    assert artifacts.metadata.name == "model_meta.json"

    with pytest.raises(FileExistsError):
        create_run_artifacts("smoke", tmp_path / "artifacts")


def test_calibration_metric_mapping_preserves_names_and_skips_unattainable_values() -> None:
    logged: list[dict[str, float]] = []
    tracker = SimpleNamespace(log_metrics=lambda metrics: logged.append(dict(metrics)))
    result = SimpleNamespace(
        after={"top1": 0.8, "macro_f1": 0.7, "ece": 0.1, "nll": 0.9},
        before={"ece": 0.2, "nll": 1.1},
        temperature=1.5,
        selection=SimpleNamespace(
            threshold=None, coverage=None, selective_accuracy=None
        ),
    )

    train._log_calibration_metrics(tracker, result)

    assert logged == [
        {
            "top1": 0.8,
            "f1_macro": 0.7,
            "ece": 0.1,
            "temperature": 1.5,
            "ece_before": 0.2,
            "nll_before": 1.1,
            "nll_after": 0.9,
        }
    ]


def test_calibrate_cli_passes_explicit_artifacts_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[float, Path | None]] = []
    monkeypatch.setattr(calibrate, "setup_logging", lambda: None)
    monkeypatch.setattr(
        calibrate,
        "run_fit",
        lambda target, artifacts_dir=None: calls.append((target, artifacts_dir)) or 0,
    )

    artifacts_dir = tmp_path / "run"
    assert (
        calibrate.main(
            [
                "fit",
                "--target-selective-accuracy",
                "0.91",
                "--artifacts-dir",
                str(artifacts_dir),
            ]
        )
        == 0
    )

    assert calls == [(0.91, artifacts_dir)]


def test_calibrate_cli_default_target_is_098_and_uses_global_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[float, Path | None]] = []
    monkeypatch.setattr(calibrate, "setup_logging", lambda: None)
    monkeypatch.setattr(
        calibrate,
        "run_fit",
        lambda target, artifacts_dir=None: calls.append((target, artifacts_dir)) or 0,
    )

    assert calibrate.main(["fit"]) == 0

    assert calls == [(cfg.TARGET_SELECTIVE_ACCURACY, None)]
    assert cfg.TARGET_SELECTIVE_ACCURACY == 0.98


def test_git_dirty_is_true_for_any_tracked_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tracking,
        "_git",
        lambda *args: "M  .gitignore" if args[0] == "status" else "",
    )

    assert tracking._git_dirty() == "true"
