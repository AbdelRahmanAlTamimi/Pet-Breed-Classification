from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

from PIL import Image

from monitoring import retrain_on_drift
from monitoring.retrain_on_drift import RandomCorruption, load_state, on_drift_check

NOW = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)


def _fake_retrain() -> dict[str, object]:
    return {"run_id": "fake", "promoted": False}


def test_alarm_below_the_scored_image_minimum_is_blocked(tmp_path: Path) -> None:
    state = tmp_path / "state.json"

    decision = on_drift_check(999, True, now=NOW, retrain=_fake_retrain, state_path=state)

    assert decision["triggered"] is False
    assert "999 images" in decision["reason"]
    assert load_state(state)["scored_since_last_retrain"] == 999


def test_alarm_triggers_once_then_cooldown_and_daily_cap_block(tmp_path: Path) -> None:
    state = tmp_path / "state.json"

    first = on_drift_check(1000, True, now=NOW, retrain=_fake_retrain, state_path=state)
    second = on_drift_check(
        1000, True, now=NOW + timedelta(hours=1), retrain=_fake_retrain, state_path=state
    )
    later = on_drift_check(
        1000, True, now=NOW + timedelta(hours=25), retrain=_fake_retrain, state_path=state
    )

    assert first["triggered"] is True and first["run_id"] == "fake"
    assert second["triggered"] is False and "cooldown" in second["reason"]
    assert later["triggered"] is True
    saved = load_state(state)
    assert saved["retrains_by_day"] == {"2026-10-11": 1, "2026-10-12": 1}
    assert saved["scored_since_last_retrain"] == 0


def test_daily_cap_blocks_after_cooldown_expires_on_the_same_day(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    retrain_on_drift.save_state(
        {
            "last_retrain_at": (NOW - timedelta(hours=30)).isoformat(),
            "retrains_by_day": {NOW.date().isoformat(): 1},
            "scored_since_last_retrain": 5000,
        },
        state,
    )

    decision = on_drift_check(10, True, now=NOW, retrain=_fake_retrain, state_path=state)

    assert decision["triggered"] is False
    assert "daily cap" in decision["reason"]


def test_no_alarm_never_retrains_but_counts_images(tmp_path: Path) -> None:
    state = tmp_path / "state.json"

    decision = on_drift_check(5000, False, now=NOW, retrain=_fake_retrain, state_path=state)

    assert decision == {"triggered": False, "checked_at": NOW.isoformat(), "reason": "no drift alarm"}
    assert load_state(state)["scored_since_last_retrain"] == 5000


def test_random_corruption_keeps_size_and_respects_probability() -> None:
    image = Image.new("RGB", (64, 48), (120, 80, 40))
    random.seed(0)

    assert RandomCorruption(probability=0.0)(image) is image
    corrupted = RandomCorruption(probability=1.0)(image)
    assert corrupted is not image
    assert corrupted.size == image.size
