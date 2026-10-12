"""One-shot final evaluation on the official test split through the served path.

Run once, after the model, temperature and threshold are frozen:
``uv run python -m pet_breed_classification.evaluate_test``

It scores the clean test split and a seeded sample of every corruption x severity with
the shipped ONNX model, temperature and abstention threshold, and refuses to overwrite
an existing final report: nothing may be tuned after looking at these numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix

from .calibrate import logits_for_records, split_metrics
from .calibration import coverage_and_selective_accuracy, softmax
from .config import cfg
from .logging_conf import setup_logging
from .predict import PetBreedPredictor

logger = logging.getLogger(__name__)

ROBUSTNESS_SAMPLE_SIZE = 1000
TOP_CONFUSED_PAIRS = 10
REPORT_PATH = cfg.REPORTS_DIR / "final_test_evaluation.json"
CONFUSION_PATH = cfg.REPORTS_DIR / "confusion_matrix.png"
ROBUSTNESS_PATH = cfg.REPORTS_DIR / "robustness.csv"


def load_test_records(corruption: str | None = None, severity: int = 0) -> list[dict[str, Any]]:
    """Return manifest test records for one corruption and severity (None = clean)."""
    manifest = json.loads(cfg.MANIFEST_PATH.read_text(encoding="utf-8"))
    records = [
        record
        for record in manifest
        if record["split"] == "test"
        and record["corruption"] == corruption
        and int(record["severity"]) == severity
    ]
    if not records:
        raise ValueError(f"No test records for corruption={corruption!r} severity={severity}")
    return records


def corruption_levels() -> list[tuple[str, int]]:
    """Return every (corruption, severity) present in the manifest, sorted."""
    manifest = json.loads(cfg.MANIFEST_PATH.read_text(encoding="utf-8"))
    return sorted(
        {
            (str(record["corruption"]), int(record["severity"]))
            for record in manifest
            if record["split"] == "test" and record["corruption"] is not None
        }
    )


def seeded_sample(
    records: Sequence[dict[str, Any]], size: int, seed: int
) -> list[dict[str, Any]]:
    """Return a reproducible sample without replacement, in manifest order."""
    if size >= len(records):
        return list(records)
    chosen = np.random.default_rng(seed).choice(len(records), size=size, replace=False)
    return [records[index] for index in sorted(chosen)]


def selective_metrics(
    logits: NDArray[np.floating], labels: NDArray[np.integer], temperature: float, threshold: float
) -> dict[str, float | None]:
    """Return n, top-1, macro-F1, ECE, coverage and selective accuracy."""
    metrics = split_metrics(logits, labels, temperature)
    coverage, selective_accuracy = coverage_and_selective_accuracy(
        softmax(logits, temperature), labels, threshold
    )
    return {
        "n": len(labels),
        "top1": metrics["top1"],
        "macro_f1": metrics["macro_f1"],
        "ece": metrics["ece"],
        "coverage": coverage,
        "selective_accuracy": selective_accuracy,
    }


def most_confused_pairs(
    matrix: NDArray[np.integer], breeds: Sequence[str], limit: int = TOP_CONFUSED_PAIRS
) -> list[dict[str, Any]]:
    """Return the largest off-diagonal cells as true -> predicted breed counts."""
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    order = np.argsort(-off_diagonal, axis=None, kind="stable")[:limit]
    pairs = []
    for flat_index in order:
        true_index, predicted_index = np.unravel_index(flat_index, matrix.shape)
        count = int(off_diagonal[true_index, predicted_index])
        if count == 0:
            break
        pairs.append(
            {
                "true": breeds[true_index],
                "predicted": breeds[predicted_index],
                "count": count,
            }
        )
    return pairs


def plot_confusion(matrix: NDArray[np.integer], breeds: Sequence[str], path: Path) -> None:
    """Save a row-normalized confusion matrix heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, where=row_totals > 0, out=np.zeros(matrix.shape))
    figure, axis = plt.subplots(figsize=(14, 12))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(breeds)), breeds, rotation=90, fontsize=7)
    axis.set_yticks(range(len(breeds)), breeds, fontsize=7)
    axis.set_xlabel("Predicted breed")
    axis.set_ylabel("True breed")
    axis.set_title(f"Final test confusion matrix (row-normalized, n={int(matrix.sum())})")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(cfg.PROJECT_ROOT).as_posix()


def run(sample_size: int = ROBUSTNESS_SAMPLE_SIZE) -> dict[str, Any]:
    """Evaluate the clean test split and the corruption suite; write all reports."""
    predictor = PetBreedPredictor.load()
    temperature = predictor.temperature
    threshold = predictor.abstain_threshold
    if threshold is None:
        raise RuntimeError("The shipped model has no abstention threshold")
    breeds = [str(entry["breed"]) for entry in predictor.classes]

    logits, labels = logits_for_records(predictor, load_test_records())
    clean = selective_metrics(logits, labels, temperature, threshold)
    predictions = softmax(logits, temperature).argmax(axis=1)
    matrix = confusion_matrix(labels, predictions, labels=range(len(breeds)))
    plot_confusion(matrix, breeds, CONFUSION_PATH)

    rows = []
    for corruption, severity in corruption_levels():
        sample = seeded_sample(load_test_records(corruption, severity), sample_size, cfg.SEED)
        level_logits, level_labels = logits_for_records(predictor, sample)
        metrics = selective_metrics(level_logits, level_labels, temperature, threshold)
        rows.append({"corruption": corruption, "severity": severity, **metrics})
        logger.info("Robustness %s s%d: %s", corruption, severity, metrics)
    with ROBUSTNESS_PATH.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "corruption",
                "severity",
                "n",
                "top1",
                "macro_f1",
                "ece",
                "coverage",
                "selective_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "split": "test",
        "model_version": str(predictor.metadata["model_version"]),
        "onnx_sha256": str(predictor.metadata["onnx_sha256"]),
        "temperature": temperature,
        "abstain_threshold": threshold,
        "clean": clean,
        "most_confused_pairs": most_confused_pairs(matrix, breeds),
        "confusion_matrix": matrix.tolist(),
        "breeds": breeds,
        "robustness": {
            "sample_size": sample_size,
            "seed": cfg.SEED,
            "path": _relative(ROBUSTNESS_PATH),
        },
        "confusion_matrix_png": _relative(CONFUSION_PATH),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample-size", type=int, default=ROBUSTNESS_SAMPLE_SIZE)
    args = parser.parse_args(argv)
    if REPORT_PATH.exists():
        parser.error(f"{_relative(REPORT_PATH)} exists; the final test evaluation runs once")
    report = run(args.sample_size)
    print(json.dumps({key: report[key] for key in ("clean", "most_confused_pairs")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
