"""Calibrate the served model on the validation split, and sanity-check its abstention.

    uv run python -m pet_breed_classification.calibrate fit [--target-selective-accuracy 0.95]
    uv run python -m pet_breed_classification.calibrate sanity [--dir PATH]

``fit`` uses the validation split only. It fits the temperature on validation logits,
chooses the abstention threshold on the calibrated validation scores, checks the choice
with a stratified 2-fold swap, writes reports/calibration.png and reports/calibration.json,
and records temperature and abstain_threshold in model_meta.json.

``sanity`` sends synthetic and optional owner-supplied out-of-distribution images through
/predict and reports what is flagged. It is informational: its results never tune the
threshold.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from .calibration import (
    ReliabilityBins,
    ThresholdSelection,
    accepted_counts,
    coverage_and_selective_accuracy,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    reliability_bins,
    select_threshold,
    softmax,
)
from .config import cfg
from .data import load_records
from .logging_conf import setup_logging
from .predict import PetBreedPredictor

logger = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
LOGIT_BATCH_SIZE = 32
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"})
REPORT_JSON_NAME = "calibration.json"
REPORT_PNG_NAME = "calibration.png"


@dataclass(frozen=True)
class FitResult:
    """Outputs of one calibration fit."""

    temperature: float
    selection: ThresholdSelection
    before: dict[str, float]
    after: dict[str, float]
    table: list[dict[str, Any]]
    two_fold: dict[str, Any]
    report_path: Path
    figure_path: Path


def logits_for_records(
    predictor: PetBreedPredictor,
    records: Sequence[dict[str, Any]],
) -> tuple[FloatArray, IntArray]:
    """Return uncalibrated ONNX CPU logits and labels for clean records."""
    if not records:
        raise ValueError("records has no samples")
    labels = np.array(
        [int(record["class_index"]) for record in records], dtype=np.int64
    )
    blocks: list[FloatArray] = []
    for start in range(0, len(records), LOGIT_BATCH_SIZE):
        images: list[Image.Image] = []
        for record in records[start : start + LOGIT_BATCH_SIZE]:
            with Image.open(cfg.PROJECT_ROOT / str(record["path"])) as image:
                images.append(image.copy())
        logits = predictor.raw_logits(images)
        if logits.shape[0] != len(images):
            raise ValueError(
                "predictor.raw_logits returned a row count different from the input batch"
            )
        blocks.append(logits)
    return np.concatenate(blocks, axis=0), labels


def validation_logits(predictor: PetBreedPredictor) -> tuple[FloatArray, IntArray]:
    """Return uncalibrated validation logits through the served inference path."""
    records = load_records("val")
    return logits_for_records(predictor, records)


def split_metrics(
    logits: FloatArray, labels: IntArray, temperature: float
) -> dict[str, float]:
    """Return NLL, ECE, top-1 and macro-F1 for softmax(logits / temperature)."""
    probabilities = softmax(logits, temperature)
    predictions = probabilities.argmax(axis=1)
    return {
        "temperature": float(temperature),
        "nll": negative_log_likelihood(logits, labels, temperature),
        "ece": expected_calibration_error(probabilities, labels, cfg.CALIBRATION_BINS),
        "top1": float(np.mean(predictions == labels)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def _bins_to_json(bins: ReliabilityBins) -> dict[str, list[Any]]:
    def clean(values: FloatArray) -> list[float | None]:
        return [None if np.isnan(value) else float(value) for value in values]

    return {
        "edges": [float(edge) for edge in bins.edges],
        "counts": [int(count) for count in bins.counts],
        "confidence": clean(bins.confidence),
        "accuracy": clean(bins.accuracy),
    }


def two_fold_swap(
    logits: FloatArray, labels: IntArray, target: float, seed: int = cfg.SEED
) -> dict[str, Any]:
    """Fit on one stratified half and evaluate on the other, then swap.

    Each half gets its own temperature and threshold. The held-out figures pool the two
    evaluation halves. A half whose target is unattainable makes the pooled figures None.
    """
    negative_log_likelihood(logits, labels)
    if len(logits) < 2:
        raise ValueError(
            "two-fold calibration requires at least two validation samples"
        )
    class_counts = np.bincount(np.asarray(labels, dtype=np.int64))
    if len(class_counts) == 0 or int(class_counts[class_counts > 0].min()) < 2:
        raise ValueError(
            "two-fold stratified calibration requires at least two samples per class"
        )

    folds: list[dict[str, Any]] = []
    accepted_total = 0
    correct_total = 0
    evaluated_total = 0
    pooled_available = True
    splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    for fold_index, (fit_index, eval_index) in enumerate(
        splitter.split(logits, labels)
    ):
        temperature = fit_temperature(logits[fit_index], labels[fit_index])
        selection = select_threshold(
            softmax(logits[fit_index], temperature), labels[fit_index], target
        )
        entry: dict[str, Any] = {
            "fold": fold_index,
            "fit_size": len(fit_index),
            "eval_size": len(eval_index),
            "temperature": float(temperature),
            "threshold": selection.threshold,
            "threshold_reason": selection.reason,
            "held_out_coverage": None,
            "held_out_selective_accuracy": None,
            "accepted_count": None,
            "abstention_count": None,
        }
        if selection.threshold is None:
            pooled_available = False
            folds.append(entry)
            continue
        eval_probabilities = softmax(logits[eval_index], temperature)
        coverage, selective = coverage_and_selective_accuracy(
            eval_probabilities, labels[eval_index], selection.threshold
        )
        accepted, correct = accepted_counts(
            eval_probabilities, labels[eval_index], selection.threshold
        )
        accepted_total += accepted
        correct_total += correct
        evaluated_total += len(eval_index)
        entry["held_out_coverage"] = coverage
        entry["held_out_selective_accuracy"] = selective
        entry["accepted_count"] = accepted
        entry["abstention_count"] = len(eval_index) - accepted
        folds.append(entry)

    pooled: dict[str, float | None] = {"coverage": None, "selective_accuracy": None}
    if pooled_available and evaluated_total > 0:
        pooled["coverage"] = accepted_total / evaluated_total
        pooled["selective_accuracy"] = (
            None if accepted_total == 0 else correct_total / accepted_total
        )
    return {"seed": seed, "folds": folds, "held_out_pooled": pooled}


def _gap(in_sample: float | None, held_out: float | None) -> float | None:
    if in_sample is None or held_out is None:
        return None
    return in_sample - held_out


def _write_model_meta(
    meta_path: Path, temperature: float, threshold: float | None
) -> None:
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError("model metadata must be a JSON object")
    metadata["temperature"] = float(temperature)
    metadata["abstain_threshold"] = None if threshold is None else float(threshold)
    _atomic_write_text(meta_path, json.dumps(metadata, indent=2) + "\n")


def _atomic_write_text(path: Path, contents: str) -> None:
    """Replace one JSON/text output without leaving a partially written target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(contents, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def plot_reliability(
    before: ReliabilityBins,
    after: ReliabilityBins,
    ece_before: float,
    ece_after: float,
    temperature: float,
    path: Path,
) -> None:
    """Save a two-panel reliability diagram, before and after temperature scaling."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    panels = (
        ("Before (T = 1)", before, ece_before),
        (f"After (T = {temperature:.3f})", after, ece_after),
    )
    for axis, (title, bins, ece) in zip(axes, panels, strict=True):
        centers = (bins.edges[:-1] + bins.edges[1:]) / 2
        populated = bins.counts > 0
        width = 0.9 / len(bins.counts)
        axis.bar(
            centers[populated],
            bins.accuracy[populated],
            width=width,
            color="tab:blue",
            edgecolor="black",
            alpha=0.7,
            label="accuracy in bin",
        )
        axis.plot(
            bins.confidence[populated],
            bins.accuracy[populated],
            "o",
            color="tab:red",
            label="mean confidence vs accuracy",
        )
        axis.plot([0, 1], [0, 1], "k--", label="perfect calibration")
        axis.set_title(f"{title}\nECE = {ece:.4f}")
        axis.set_xlabel("top-1 confidence")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(loc="lower right", fontsize=8)
    figure.suptitle("Reliability diagram, validation split")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def fit_from_logits(
    logits: FloatArray,
    labels: IntArray,
    *,
    model_version: str,
    onnx_sha256: str,
    target: float,
    reports_dir: Path,
    meta_path: Path | None,
    seed: int = cfg.SEED,
) -> FitResult:
    """Fit temperature and threshold on validation logits and write the report files.

    ``meta_path`` receives temperature and abstain_threshold when given. A threshold of
    None is written as null, never replaced with a made-up value.
    """
    before = split_metrics(logits, labels, 1.0)
    temperature = fit_temperature(logits, labels)
    after = split_metrics(logits, labels, temperature)
    calibrated = softmax(logits, temperature)
    raw = softmax(logits, 1.0)
    raw_predictions = np.argmax(logits, axis=1)
    calibrated_predictions = np.argmax(calibrated, axis=1)
    if not np.array_equal(raw_predictions, calibrated_predictions):
        raise RuntimeError(
            "temperature scaling changed the argmax; this must not happen"
        )
    if after["top1"] != before["top1"] or after["macro_f1"] != before["macro_f1"]:
        raise RuntimeError(
            "temperature scaling changed top-1 or macro-F1; this must not happen"
        )

    targets = sorted(set(cfg.CALIBRATION_REPORT_TARGETS) | {target})
    table: list[dict[str, Any]] = []
    selections: dict[float, ThresholdSelection] = {}
    for value in targets:
        selection = select_threshold(calibrated, labels, value)
        selections[value] = selection
        table.append(
            {
                "target": value,
                "chosen": value == target,
                "threshold": selection.threshold,
                "coverage": selection.coverage,
                "selective_accuracy": selection.selective_accuracy,
                "reason": selection.reason,
            }
        )
    selection = selections[target]

    two_fold = two_fold_swap(logits, labels, target, seed)
    pooled = two_fold["held_out_pooled"]
    in_sample_coverage = selection.coverage
    in_sample_selective = selection.selective_accuracy
    two_fold["in_sample"] = {
        "threshold": selection.threshold,
        "coverage": in_sample_coverage,
        "selective_accuracy": in_sample_selective,
    }
    two_fold["gap_in_sample_minus_held_out"] = {
        "coverage": _gap(in_sample_coverage, pooled["coverage"]),
        "selective_accuracy": _gap(in_sample_selective, pooled["selective_accuracy"]),
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    figure_path = reports_dir / REPORT_PNG_NAME
    report_path = reports_dir / REPORT_JSON_NAME
    plot_reliability(
        reliability_bins(raw, labels, cfg.CALIBRATION_BINS),
        reliability_bins(calibrated, labels, cfg.CALIBRATION_BINS),
        before["ece"],
        after["ece"],
        temperature,
        figure_path,
    )
    report = {
        "split": "val",
        "n_val": len(labels),
        "model_version": model_version,
        "onnx_sha256": onnx_sha256,
        "calibration_bins": cfg.CALIBRATION_BINS,
        "temperature": float(temperature),
        "target_selective_accuracy": target,
        "abstain_threshold": selection.threshold,
        "abstain_threshold_reason": selection.reason,
        "before": before,
        "after": after,
        "reliability_before": _bins_to_json(
            reliability_bins(raw, labels, cfg.CALIBRATION_BINS)
        ),
        "reliability_after": _bins_to_json(
            reliability_bins(calibrated, labels, cfg.CALIBRATION_BINS)
        ),
        "threshold_table": table,
        "two_fold_check": two_fold,
    }
    _atomic_write_text(report_path, json.dumps(report, indent=2) + "\n")
    if meta_path is not None:
        _write_model_meta(meta_path, temperature, selection.threshold)
    return FitResult(
        temperature=float(temperature),
        selection=selection,
        before=before,
        after=after,
        table=table,
        two_fold=two_fold,
        report_path=report_path,
        figure_path=figure_path,
    )


def _print_fit_summary(result: FitResult) -> None:
    print(f"temperature T = {result.temperature:.6f}")
    print("validation metrics (val split only)")
    print(f"  {'':8}{'NLL':>10}{'ECE':>10}{'top-1':>10}{'macro-F1':>11}")
    for name, values in (("before", result.before), ("after", result.after)):
        print(
            f"  {name:8}{values['nll']:>10.4f}{values['ece']:>10.4f}"
            f"{values['top1']:>10.4f}{values['macro_f1']:>11.4f}"
        )
    print("abstention on calibrated val scores")
    print(f"  {'target':>8}{'threshold':>12}{'coverage':>11}{'sel. acc.':>11}  chosen")
    for row in result.table:
        threshold = "none" if row["threshold"] is None else f"{row['threshold']:.6f}"
        coverage = "-" if row["coverage"] is None else f"{row['coverage']:.4f}"
        selective = (
            "-"
            if row["selective_accuracy"] is None
            else f"{row['selective_accuracy']:.4f}"
        )
        marker = "<-" if row["chosen"] else ""
        print(
            f"  {row['target']:>8.2f}{threshold:>12}{coverage:>11}{selective:>11}  {marker}"
        )
        if row["threshold"] is None:
            print(f"           reason: {row['reason']}")
    two = result.two_fold
    print(f"2-fold swap (stratified, seed={two['seed']})")
    for fold in two["folds"]:
        held_cov = fold["held_out_coverage"]
        held_sel = fold["held_out_selective_accuracy"]
        print(
            f"  fold {fold['fold']}: T={fold['temperature']:.4f} "
            f"threshold={fold['threshold']} "
            f"abstentions={fold['abstention_count']} "
            f"held-out coverage={held_cov} held-out sel. acc.={held_sel}"
        )
    pooled = two["held_out_pooled"]
    gap = two["gap_in_sample_minus_held_out"]
    print(
        f"  pooled held-out: coverage={pooled['coverage']} "
        f"selective accuracy={pooled['selective_accuracy']}"
    )
    print(
        f"  in-sample minus held-out: coverage gap={gap['coverage']} "
        f"selective-accuracy gap={gap['selective_accuracy']}"
    )
    print(f"wrote {result.figure_path}")
    print(f"wrote {result.report_path}")


def fit_artifacts(
    artifacts_dir: Path,
    target: float,
    *,
    seed: int = cfg.SEED,
) -> FitResult:
    """Fit calibration for one artifact directory using validation records only."""
    return _fit_paths(
        artifacts_dir / "model.onnx",
        artifacts_dir / "model_meta.json",
        artifacts_dir / "reports",
        target,
        seed=seed,
    )


def _fit_paths(
    onnx_path: Path,
    meta_path: Path,
    reports_dir: Path,
    target: float,
    *,
    seed: int,
) -> FitResult:
    """Fit calibration for explicit artifact files."""
    predictor = PetBreedPredictor.load(
        onnx_path, meta_path, require_calibration=False
    )
    logits, labels = validation_logits(predictor)
    return fit_from_logits(
        logits,
        labels,
        model_version=str(predictor.metadata["model_version"]),
        onnx_sha256=str(predictor.metadata["onnx_sha256"]),
        target=target,
        reports_dir=reports_dir,
        meta_path=meta_path,
        seed=seed,
    )


def run_fit(target: float, artifacts_dir: Path | None = None) -> int:
    """Fit calibration for an artifact directory. Returns a process exit code."""
    if artifacts_dir is None:
        directory = cfg.ARTIFACTS_DIR
        result = fit_artifacts(directory, target, seed=cfg.SEED)
    else:
        directory = artifacts_dir
        result = fit_artifacts(directory, target, seed=cfg.SEED)
    _print_fit_summary(result)
    print(f"wrote temperature and abstain_threshold to {directory / 'model_meta.json'}")
    if result.selection.threshold is None:
        print(
            "abstain_threshold is null: no threshold reaches the target. The API will refuse "
            "to start until a threshold is set."
        )
        return 1
    return 0


def synthetic_inputs() -> dict[str, list[tuple[str, bytes]]]:
    """Return deterministic synthetic out-of-distribution PNG inputs by category."""
    rng = np.random.default_rng(cfg.SEED)

    def encode(image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    categories: dict[str, list[tuple[str, bytes]]] = {}
    categories["uniform noise"] = [
        (
            f"uniform-noise-{index:02d}",
            encode(
                Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))
            ),
        )
        for index in range(20)
    ]
    solid = [
        ("black", (0, 0, 0)),
        ("white", (255, 255, 255)),
        ("red", (255, 0, 0)),
        ("green", (0, 255, 0)),
        ("blue", (0, 0, 255)),
        ("grey", (128, 128, 128)),
        ("yellow", (255, 255, 0)),
        ("cyan", (0, 255, 255)),
        ("magenta", (255, 0, 255)),
        ("brown", (139, 69, 19)),
    ]
    categories["solid colours"] = [
        (f"solid-{name}", encode(Image.new("RGB", (224, 224), color)))
        for name, color in solid
    ]
    gradients: list[tuple[str, bytes]] = []
    for size in (224, 320):
        ramp = np.linspace(0, 255, size)
        grid_y, grid_x = np.meshgrid(ramp, ramp, indexing="ij")
        radial = np.sqrt((grid_x - size / 2) ** 2 + (grid_y - size / 2) ** 2)
        patterns = {
            "horizontal": grid_x,
            "vertical": grid_y,
            "diagonal": (grid_x + grid_y) / 2,
            "radial": 255 * radial / radial.max(),
        }
        for name, values in patterns.items():
            array = np.stack([values, 255 - values, values[::-1]], axis=-1).astype(
                np.uint8
            )
            gradients.append(
                (f"gradient-{name}-{size}", encode(Image.fromarray(array)))
            )
    categories["gradients"] = gradients
    checkerboards: list[tuple[str, bytes]] = []
    for cell in (4, 8, 16, 32):
        rows, cols = np.indices((224, 224))
        pattern = ((rows // cell + cols // cell) % 2).astype(np.uint8) * 255
        for invert in (False, True):
            values = 255 - pattern if invert else pattern
            array = np.stack([values] * 3, axis=-1)
            name = f"checkerboard-{cell}-{'inv' if invert else 'std'}"
            checkerboards.append((name, encode(Image.fromarray(array))))
    categories["checkerboards"] = checkerboards
    categories["RGBA PNG (4 channels)"] = [
        (
            f"rgba-{index:02d}",
            encode(
                Image.fromarray(rng.integers(0, 256, (256, 256, 4), dtype=np.uint8))
            ),
        )
        for index in range(5)
    ]
    categories["greyscale"] = [
        (
            f"greyscale-{index:02d}",
            encode(Image.fromarray(rng.integers(0, 256, (256, 256), dtype=np.uint8))),
        )
        for index in range(5)
    ]
    return categories


def owner_inputs(directory: Path) -> list[tuple[str, bytes]]:
    """Return raw bytes of image files in ``directory``. Names are never printed."""
    if not directory.is_dir():
        return []
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    return [
        (f"owner-{index:03d}", path.read_bytes()) for index, path in enumerate(files)
    ]


def summarize_responses(
    responses: Sequence[tuple[int, str | None, float | None]],
) -> dict[str, Any]:
    """Summarize (status, decision, confidence) triples for one category."""
    confidences = [value for _, _, value in responses if value is not None]
    decisions = Counter(
        decision for _, decision, _ in responses if decision is not None
    )
    return {
        "count": len(responses),
        "uncertain": decisions.get("uncertain", 0),
        "confident": decisions.get("confident", 0),
        "confidence_min": min(confidences) if confidences else None,
        "confidence_max": max(confidences) if confidences else None,
        "statuses": dict(sorted(Counter(status for status, _, _ in responses).items())),
    }


def _send_inputs(
    categories: dict[str, list[tuple[str, bytes]]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """POST every input to /predict through the app; return summaries and confident names."""
    from fastapi.testclient import TestClient

    from .api import main as api_main

    summaries: dict[str, dict[str, Any]] = {}
    confident_synthetic: list[str] = []
    with TestClient(api_main.app) as client:
        for category, items in categories.items():
            responses: list[tuple[int, str | None, float | None]] = []
            for name, contents in items:
                response = client.post(
                    "/predict",
                    files={"file": (f"{name}.png", contents, "image/png")},
                )
                body = response.json() if response.status_code == 200 else {}
                responses.append(
                    (response.status_code, body.get("decision"), body.get("confidence"))
                )
                if (
                    response.status_code == 200
                    and body.get("decision") == "confident"
                    and not category.startswith("owner-supplied")
                ):
                    confident_synthetic.append(name)
            summaries[category] = summarize_responses(responses)
    return summaries, confident_synthetic


def run_sanity(directory: Path) -> int:
    """Send out-of-distribution inputs through /predict and print what is flagged."""
    categories: dict[str, list[tuple[str, bytes]]] = dict(synthetic_inputs())
    owned = owner_inputs(directory)
    if owned:
        categories["owner-supplied photos (not printed)"] = owned

    # Per-request logs would bury the table; restore the levels afterwards.
    quiet_names = ("", "pet_breed_classification", "httpx")
    previous_levels = {name: logging.getLogger(name).level for name in quiet_names}
    for name in quiet_names:
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        summaries, confident_synthetic = _send_inputs(categories)
    finally:
        for name, level in previous_levels.items():
            logging.getLogger(name).setLevel(level)

    print(
        f"{'category':34}{'n':>4}{'uncertain':>11}{'confident':>11}{'conf. range':>21}  statuses"
    )
    for category, summary in summaries.items():
        low, high = summary["confidence_min"], summary["confidence_max"]
        span = "-" if low is None else f"{low:.4f} .. {high:.4f}"
        statuses = ", ".join(
            f"{code}x{count}" for code, count in summary["statuses"].items()
        )
        print(
            f"{category:34}{summary['count']:>4}{summary['uncertain']:>11}"
            f"{summary['confident']:>11}{span:>21}  {statuses}"
        )
    print("synthetic inputs served as confident:", confident_synthetic or "none")
    print(
        "Softmax abstention cannot guarantee out-of-distribution rejection. These results are "
        "informational and were not used to choose the threshold."
    )
    server_errors = [
        category
        for category, summary in summaries.items()
        if any(code >= 500 for code in summary["statuses"])
    ]
    if server_errors:
        print("server errors (5xx) in:", ", ".join(server_errors))
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the calibrate command-line interface."""
    parser = argparse.ArgumentParser(
        description="Calibrate and sanity-check the model."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    fit_parser = subcommands.add_parser(
        "fit", help="Fit temperature and threshold on val."
    )
    fit_parser.add_argument(
        "--target-selective-accuracy",
        type=float,
        default=cfg.TARGET_SELECTIVE_ACCURACY,
        help="Required selective accuracy on the calibrated validation scores.",
    )
    fit_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Directory containing model.onnx and model_meta.json (default: artifacts).",
    )
    sanity_parser = subcommands.add_parser(
        "sanity",
        help="Run out-of-distribution inputs through /predict (informational).",
    )
    sanity_parser.add_argument(
        "--dir",
        type=Path,
        default=cfg.OOD_SANITY_DIR,
        help="Optional directory of owner-supplied photos; skipped when absent.",
    )
    args = parser.parse_args(argv)
    setup_logging()
    if args.command == "fit":
        target = float(args.target_selective_accuracy)
        if args.artifacts_dir is None:
            return run_fit(target)
        return run_fit(target, Path(args.artifacts_dir))
    return run_sanity(Path(args.dir))


if __name__ == "__main__":
    raise SystemExit(main())
