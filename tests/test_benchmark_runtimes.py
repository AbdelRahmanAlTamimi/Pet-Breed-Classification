from __future__ import annotations

import builtins
import platform
from pathlib import Path

import pytest

from pet_breed_classification import benchmark_runtimes


def test_cpu_model_falls_back_when_proc_cpuinfo_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("/proc/cpuinfo")

    monkeypatch.setattr(builtins, "open", missing)
    monkeypatch.setattr(platform, "processor", lambda: "arm")

    assert benchmark_runtimes.cpu_model() == "arm"


def test_cpu_model_uses_machine_when_processor_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args: object, **kwargs: object) -> object:
        raise OSError("no cpuinfo")

    monkeypatch.setattr(builtins, "open", missing)
    monkeypatch.setattr(platform, "processor", lambda: "")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    assert benchmark_runtimes.cpu_model() == "arm64"


@pytest.mark.skipif(not Path("/proc/cpuinfo").is_file(), reason="/proc/cpuinfo unavailable")
def test_cpu_model_reads_linux_model_name() -> None:
    with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
        names = [line.split(":", 1)[1].strip() for line in cpuinfo if line.startswith("model name")]
    if not names:
        pytest.skip("/proc/cpuinfo has no model name line on this CPU")

    assert benchmark_runtimes.cpu_model() == names[0]
