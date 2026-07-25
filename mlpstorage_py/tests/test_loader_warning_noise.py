"""Worklist A1 + A4 (2026-07-24): loader warning-noise fixes.

A1 — datagen phase dirs can never contain a summary.json: DLIO only calls
``stats.save_data()`` under ``if not self.generate_only:`` (dlio_benchmark
main.py), so every training datagen dir in every submission triggers a
spurious "Could not load Summary log" warning (77 of the 79 in the v3.0
baseline). The loader must load the datagen summary slot silently (None
when absent) while run-phase summaries keep warning — a missing run
summary is a real submission defect.

A4 — a missing metadata file produced TWO warnings per occurrence:
``find_metadata_path`` warned "Could not find metadata file at <dir>" and
then returned the nonexistent default path, which ``load_single_log``
re-warned as "Could not load Metadata log from <path>". 46 conditions →
92 baseline lines. The find-warning (with directory context) survives;
the second is suppressed by returning None and having ``load_single_log``
treat a None path as an already-reported miss.
"""

import json
import logging
import os

from mlpstorage_py.submission_checker.loader import Loader
from mlpstorage_py.submission_checker.configuration.configuration import Config


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _build_training_tree(tmp_path, with_datagen_summary=False,
                         with_run_summary=True, with_run_metadata=True):
    """Minimal closed/Acme/results/sys-v1/training/unet3d tree with one
    datagen timestamp and one run timestamp."""
    base = tmp_path / "closed" / "Acme" / "results" / "sys-v1" / "training" / "unet3d"
    datagen_ts = base / "datagen" / "20250101_120000"
    run_ts = base / "run" / "20250101_130000"

    _write_json(str(datagen_ts / "metadata.json"), {"args": {}})
    if with_datagen_summary:
        _write_json(str(datagen_ts / "summary.json"), {"num_hosts": 1})

    if with_run_metadata:
        _write_json(str(run_ts / "metadata.json"), {"args": {}})
    else:
        os.makedirs(str(run_ts), exist_ok=True)
    if with_run_summary:
        _write_json(str(run_ts / "summary.json"), {"num_hosts": 1})
    return base


def _load_all(tmp_path, caplog):
    config = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
    loader = Loader(root=str(tmp_path), version="v2.0", config=config)
    with caplog.at_level(logging.DEBUG, logger="Loader"):
        return list(loader.load())


def _warnings(caplog, needle):
    return [
        r for r in caplog.records
        if r.levelno == logging.WARNING and needle in r.getMessage()
    ]


def test_a1_datagen_missing_summary_is_silent(tmp_path, caplog):
    """No summary.json in a training datagen dir → no warning; the tuple's
    summary slot is None. Datagen runs cannot produce one (DLIO
    generate_only), so warning is pure noise."""
    _build_training_tree(tmp_path, with_datagen_summary=False)
    logs = _load_all(tmp_path, caplog)
    assert len(logs) == 1
    summary_warnings = _warnings(caplog, "Summary")
    assert not summary_warnings, (
        f"datagen summary.json absence must not warn (A1); got: "
        f"{[r.getMessage() for r in summary_warnings]}"
    )
    datagen_files = logs[0].datagen_files
    assert len(datagen_files) == 1
    assert datagen_files[0][0] is None  # summary slot


def test_a1_run_missing_summary_still_warns(tmp_path, caplog):
    """A missing summary.json in a RUN dir is a genuine defect — the
    warning must survive the A1 silencing."""
    _build_training_tree(tmp_path, with_run_summary=False)
    _load_all(tmp_path, caplog)
    run_summary_warnings = [
        r for r in _warnings(caplog, "Summary")
        if "/run/" in r.getMessage() or os.sep + "run" + os.sep in r.getMessage()
    ]
    assert run_summary_warnings, (
        "missing RUN summary.json must still warn after A1"
    )


def test_a1_datagen_present_summary_still_loads(tmp_path, caplog):
    """When a datagen summary.json DOES exist (future tool versions), it
    must still be loaded into the tuple."""
    _build_training_tree(tmp_path, with_datagen_summary=True)
    logs = _load_all(tmp_path, caplog)
    assert logs[0].datagen_files[0][0] == {"num_hosts": 1}


def test_a4_missing_metadata_warns_exactly_once(tmp_path, caplog):
    """A run dir without any metadata file must produce ONE warning (the
    find-time one with directory context), not the find+load pair."""
    _build_training_tree(tmp_path, with_run_metadata=False)
    logs = _load_all(tmp_path, caplog)
    metadata_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and ("metadata file" in r.getMessage() or "Metadata log" in r.getMessage())
    ]
    assert len(metadata_warnings) == 1, (
        f"missing metadata must warn exactly once (A4); got "
        f"{[r.getMessage() for r in metadata_warnings]}"
    )
    # The tuple still carries None in the metadata slot.
    assert logs[0].run_files[0][1] is None


def test_a4_present_metadata_loads_without_warning(tmp_path, caplog):
    """Regression guard: an existing metadata.json loads fine and warns
    about nothing."""
    _build_training_tree(tmp_path)
    logs = _load_all(tmp_path, caplog)
    metadata_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and ("metadata file" in r.getMessage() or "Metadata log" in r.getMessage())
    ]
    assert not metadata_warnings
    assert logs[0].run_files[0][1] == {"args": {}}
