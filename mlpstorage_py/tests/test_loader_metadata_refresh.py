"""Tests for BUG-01: Loader.load() metadata_path refresh inside training run loop.

The training run loop (loader.py lines 117-122) reuses the metadata_path
from the final datagen iteration rather than refreshing it for each run
timestamp. This means every run tuple carries the DATAGEN metadata, not
the per-run metadata.

Fix: add ``metadata_path = self.find_metadata_path(timestamp_path)``
at the top of the run loop body (mirroring line 112 in the datagen loop).

References:
  - D-E1 in Phase 2 CONTEXT.md
  - RESEARCH.md §Codebase Investigation: Loader (BUG-01 site)
  - Requirements: BUG-01
"""

import json
import os

import pytest

from mlpstorage_py.submission_checker.loader import Loader, SubmissionLogs
from mlpstorage_py.submission_checker.configuration.configuration import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path, data):
    """Write *data* as JSON to *path*, creating parent directories."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bug01_run_metadata_not_datagen_carryover(tmp_path):
    """Each run timestamp must carry its OWN metadata, not the datagen metadata.

    BEFORE the BUG-01 fix all six run tuples carry num_processes=4 (the
    datagen carryover).  AFTER the fix all six carry num_processes=8.
    """
    # Build: closed/Acme/results/sys-v1/training/unet3d/
    base = tmp_path / "closed" / "Acme" / "results" / "sys-v1" / "training" / "unet3d"
    datagen_dir = base / "datagen"
    run_dir = base / "run"

    # One datagen timestamp with num_processes=4
    datagen_ts = "20250101_120000"
    datagen_ts_dir = datagen_dir / datagen_ts
    _write_json(str(datagen_ts_dir / "metadata.json"), {"args": {"num_processes": 4}})
    _write_json(str(datagen_ts_dir / "summary.json"), {"num_hosts": 1})

    # Six run timestamps each with num_processes=8
    run_timestamps = [
        "20250101_130001",
        "20250101_140001",
        "20250101_150001",
        "20250101_160001",
        "20250101_170001",
        "20250101_180001",
    ]
    for ts in run_timestamps:
        ts_dir = run_dir / ts
        _write_json(str(ts_dir / "metadata.json"), {"args": {"num_processes": 8}})
        _write_json(str(ts_dir / "summary.json"), {"num_hosts": 1})

    # Minimal system YAML (Loader does not call schema_validator)
    systems_dir = tmp_path / "closed" / "Acme" / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "sys-v1.yaml").write_text("system_under_test: {}", encoding="utf-8")

    config = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
    loader = Loader(root=str(tmp_path), version="v2.0", config=config)

    training_logs = None
    for logs in loader.load():
        if (
            logs.loader_metadata is not None
            and logs.loader_metadata.mode == "training"
        ):
            training_logs = logs
            break

    assert training_logs is not None, "Expected at least one training SubmissionLogs"
    assert training_logs.run_files is not None, "run_files must not be None"
    assert len(training_logs.run_files) == 6, (
        f"Expected 6 run tuples, got {len(training_logs.run_files)}"
    )

    # BEFORE BUG-01 fix this assertion fails because all tuples carry num_processes=4
    for _summary, metadata, ts in training_logs.run_files:
        num_processes = (metadata or {}).get("args", {}).get("num_processes")
        assert num_processes == 8, (
            f"Run timestamp {ts}: expected num_processes=8 (per-run metadata), "
            f"got {num_processes} — BUG-01 not yet fixed (datagen carryover still active)"
        )


def test_bug01_checkpointing_branch_unaffected(tmp_path):
    """The checkpointing branch already refreshes metadata_path per timestamp.

    This test pins that behaviour so the BUG-01 fix (confined to the training
    branch) does NOT accidentally regress the checkpointing branch.

    Each checkpointing timestamp gets a DISTINCT num_processes value (8, 16, 32).
    After loading, each SubmissionLogs.checkpoint_files entry must carry its
    own metadata.
    """
    base = (
        tmp_path / "closed" / "Acme" / "results" / "sys-v1"
        / "checkpointing" / "llama3-8b"
    )

    ts_data = [
        ("20250101_130001", 8),
        ("20250101_140001", 16),
        ("20250101_150001", 32),
    ]
    for ts, num_p in ts_data:
        ts_dir = base / ts
        _write_json(str(ts_dir / "metadata.json"), {"args": {"num_processes": num_p}})
        _write_json(str(ts_dir / "summary.json"), {"num_hosts": 1})

    systems_dir = tmp_path / "closed" / "Acme" / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "sys-v1.yaml").write_text("system_under_test: {}", encoding="utf-8")

    config = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
    loader = Loader(root=str(tmp_path), version="v2.0", config=config)

    checkpoint_logs = None
    for logs in loader.load():
        if (
            logs.loader_metadata is not None
            and logs.loader_metadata.mode == "checkpointing"
        ):
            checkpoint_logs = logs
            break

    assert checkpoint_logs is not None, "Expected at least one checkpointing SubmissionLogs"
    assert checkpoint_logs.checkpoint_files is not None

    # Map timestamp → num_processes from what the loader yielded
    loaded = {
        ts: (metadata or {}).get("args", {}).get("num_processes")
        for _summary, metadata, ts in checkpoint_logs.checkpoint_files
    }
    expected = {ts: np for ts, np in ts_data}
    assert loaded == expected, (
        f"Checkpointing branch returned wrong per-timestamp num_processes.\n"
        f"Expected: {expected}\nGot: {loaded}"
    )
