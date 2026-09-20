"""
mlcommons/storage#830 — the shared training and checkpointing ``summary.json``
fixtures must use the metric keys real DLIO emits.

Before this test the fixtures carried keys DLIO never writes —
``train_io_throughput_MB_per_second`` (a list; DLIO's I/O figure is the scalar
``train_io_mean_MB_per_second``) and ``checkpoint_{read,write}_throughput_GB_per_second``
(lists; DLIO emits ``save_``/``load_`` scalar means and durations, no per-op
lists). Tests therefore exercised reportgen against a metric layout production
never produces, and the aggregation grew ``*_mean_of_*`` columns that could
never populate from a real run.

The key sets below are transcribed from ``dlio_benchmark/utils/statscounter.py``
(DLIO_local_changes) and cross-checked against the v3.0 submissions tree
(every training run there carries exactly the nine training keys; every
checkpointing run carries a subset of the nine checkpointing keys depending
on which phases it ran).
"""

from __future__ import annotations

import json
import pathlib

import pytest

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "sample_results"

# statscounter.py:201-205 — training. Lists are per-epoch; the rest scalars.
REAL_TRAINING_LISTS = {
    "train_au_percentage",
    "train_throughput_samples_per_second",
}
REAL_TRAINING_SCALARS = {
    "train_au_mean_percentage",
    "train_au_stdev_percentage",
    "train_au_meet_expectation",
    "train_throughput_mean_samples_per_second",
    "train_throughput_stdev_samples_per_second",
    "train_io_mean_MB_per_second",
    "train_io_stdev_MB_per_second",
}

# statscounter.py:178-185 — checkpointing. Scalars only; no per-op lists.
REAL_CHECKPOINTING_SCALARS = {
    "save_checkpoint_io_mean_GB_per_second",
    "save_checkpoint_io_stdev_GB_per_second",
    "save_checkpoint_duration_mean_seconds",
    "save_checkpoint_duration_stdev_seconds",
    "load_checkpoint_io_mean_GB_per_second",
    "load_checkpoint_io_stdev_GB_per_second",
    "load_checkpoint_duration_mean_seconds",
    "load_checkpoint_duration_stdev_seconds",
    "checkpoint_size_GB",
}


def _summaries(kind: str) -> list[pathlib.Path]:
    return sorted(
        p for p in FIXTURES.rglob("summary.json")
        if f"/{kind}/" in p.as_posix() and "/run/" in p.as_posix() or (
            kind == "checkpointing" and f"/{kind}/" in p.as_posix()
        )
    )


def _metric(path: pathlib.Path) -> dict:
    with open(path) as fh:
        return json.load(fh).get("metric") or {}


@pytest.mark.parametrize("path", _summaries("training"), ids=lambda p: p.relative_to(FIXTURES).as_posix())
def test_training_fixture_uses_only_real_dlio_keys(path):
    metric = _metric(path)
    unknown = set(metric) - REAL_TRAINING_LISTS - REAL_TRAINING_SCALARS
    assert not unknown, f"{path.relative_to(FIXTURES)}: keys real DLIO never emits: {sorted(unknown)}"
    for key in REAL_TRAINING_LISTS & set(metric):
        assert isinstance(metric[key], list), f"{key} is a per-epoch list in real DLIO"
    for key in REAL_TRAINING_SCALARS & set(metric):
        assert not isinstance(metric[key], list), f"{key} is a scalar in real DLIO"
    # Every training run DLIO writes carries its I/O figure as this scalar;
    # it is what the Read B/W column is built from.
    assert "train_io_mean_MB_per_second" in metric, (
        f"{path.relative_to(FIXTURES)}: missing train_io_mean_MB_per_second"
    )


@pytest.mark.parametrize("path", _summaries("checkpointing"), ids=lambda p: p.relative_to(FIXTURES).as_posix())
def test_checkpointing_fixture_uses_only_real_dlio_keys(path):
    metric = _metric(path)
    unknown = set(metric) - REAL_CHECKPOINTING_SCALARS
    assert not unknown, f"{path.relative_to(FIXTURES)}: keys real DLIO never emits: {sorted(unknown)}"
    for key, value in metric.items():
        assert not isinstance(value, list), f"{key}: real DLIO checkpointing emits no per-op lists"
    assert metric, "a checkpointing run's summary carries at least the save_ scalars"


def test_fixture_inventory_is_non_trivial():
    """Guard the parametrization itself: if the fixture tree moves, the two
    tests above must not silently collect zero cases."""
    assert len(_summaries("training")) >= 15
    assert len(_summaries("checkpointing")) >= 3
