from pathlib import Path

import pytest
import torch

from pet_breed_classification.model import (
    PetBreedClassifier,
    load_checkpoint,
    save_checkpoint,
)


def classes() -> list[dict[str, object]]:
    return [
        {"index": index, "breed": f"breed_{index}", "species": "dog"}
        for index in range(3)
    ]


def test_checkpoint_round_trip_and_schema(tmp_path: Path) -> None:
    model = PetBreedClassifier(num_classes=3, pretrained=False)
    path = tmp_path / "model.pt"

    save_checkpoint(
        model,
        path,
        classes(),
        seed=42,
        epoch=2,
        validation_metrics={"accuracy": 0.5, "macro_f1": 0.4},
    )
    loaded_model, checkpoint = load_checkpoint(path, torch.device("cpu"))

    expected_keys = {
        "backbone",
        "num_classes",
        "classes",
        "eval_transform",
        "temperature",
        "abstain_threshold",
        "seed",
        "epoch",
        "validation_metrics",
        "model_state_dict",
    }
    assert expected_keys.issubset(checkpoint)
    assert checkpoint["num_classes"] == len(checkpoint["classes"])
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, loaded_model.state_dict()[name])


def test_checkpoint_rejects_num_classes_mismatch(tmp_path: Path) -> None:
    model = PetBreedClassifier(num_classes=3, pretrained=False)
    path = tmp_path / "model.pt"
    save_checkpoint(
        model,
        path,
        classes(),
        seed=42,
        epoch=1,
        validation_metrics={},
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["num_classes"] = 4
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="num_classes.*classes"):
        load_checkpoint(path, torch.device("cpu"))
