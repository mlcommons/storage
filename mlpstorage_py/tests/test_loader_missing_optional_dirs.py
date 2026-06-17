"""Tests for BUG-T1: Loader.load() must not crash on missing datagen/, run/, or checkpoint timestamp dirs.

When a submission omits the datagen/ subdir under training/<workload>/ (a
structural violation that SubmissionStructureCheck reports as STRUCT-12 /
rule 2.1.12), the loader's unguarded `list_dir(datagen_path)` raised
FileNotFoundError, which escaped run_checks() and terminated the entire
validate run mid-corpus.

Fix: guard datagen_path, run_path, and checkpoint_path with os.path.isdir
before list_dir; yield empty file lists when missing so traversal continues.
The structural check still flags the missing dir.
"""

import json
import os

from mlpstorage_py.submission_checker.loader import Loader
from mlpstorage_py.submission_checker.configuration.configuration import Config


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _write_systems_yaml(tmp_path, submitter, system):
    systems_dir = tmp_path / "closed" / submitter / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / f"{system}.yaml").write_text(
        "system_under_test: {}", encoding="utf-8"
    )


def test_bug_t1_training_missing_datagen_does_not_crash(tmp_path):
    """Missing datagen/ under training/<workload>/ must not raise."""
    base = tmp_path / "closed" / "Acme" / "results" / "sys-v1" / "training" / "unet3d"
    run_dir = base / "run" / "20250101_130001"
    _write_json(str(run_dir / "metadata.json"), {"args": {"num_processes": 8}})
    _write_json(str(run_dir / "summary.json"), {"num_hosts": 1})
    _write_systems_yaml(tmp_path, "Acme", "sys-v1")

    config = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
    loader = Loader(root=str(tmp_path), version="v2.0", config=config)

    # Must not raise. Must yield a training SubmissionLogs with empty datagen_files.
    training = [l for l in loader.load() if l.loader_metadata.mode == "training"]
    assert len(training) == 1
    assert training[0].datagen_files == []
    assert len(training[0].run_files) == 1


def test_bug_t1_training_missing_run_does_not_crash(tmp_path):
    """Missing run/ under training/<workload>/ must not raise."""
    base = tmp_path / "closed" / "Acme" / "results" / "sys-v1" / "training" / "unet3d"
    dg_dir = base / "datagen" / "20250101_120000"
    _write_json(str(dg_dir / "metadata.json"), {"args": {"num_processes": 4}})
    _write_json(str(dg_dir / "summary.json"), {"num_hosts": 1})
    _write_systems_yaml(tmp_path, "Acme", "sys-v1")

    config = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
    loader = Loader(root=str(tmp_path), version="v2.0", config=config)

    training = [l for l in loader.load() if l.loader_metadata.mode == "training"]
    assert len(training) == 1
    assert len(training[0].datagen_files) == 1
    assert training[0].run_files == []


def test_bug_t1_checkpointing_missing_timestamp_dir_does_not_crash(tmp_path):
    """Empty checkpointing/<workload>/ (no timestamp dirs) must not raise.

    Yields a SubmissionLogs with checkpoint_files == [].
    """
    base = tmp_path / "closed" / "Acme" / "results" / "sys-v1" / "checkpointing" / "llama3-8b"
    base.mkdir(parents=True, exist_ok=True)
    _write_systems_yaml(tmp_path, "Acme", "sys-v1")

    config = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
    loader = Loader(root=str(tmp_path), version="v2.0", config=config)

    checkpointing = [
        l for l in loader.load() if l.loader_metadata.mode == "checkpointing"
    ]
    assert len(checkpointing) == 1
    assert checkpointing[0].checkpoint_files == []


def test_bug_t1_traversal_continues_to_later_submitters(tmp_path):
    """A crash in submitter A must not block submitter B from being processed.

    Pins the Rules.md intro guarantee that the validator continues across
    failures rather than aborting on the first.
    """
    # Submitter A: training/unet3d/ with NO datagen/, NO run/
    base_a = tmp_path / "closed" / "AcmeA" / "results" / "sys-A" / "training" / "unet3d"
    base_a.mkdir(parents=True, exist_ok=True)
    _write_systems_yaml(tmp_path, "AcmeA", "sys-A")

    # Submitter B: well-formed training + run timestamp
    base_b = tmp_path / "closed" / "AcmeB" / "results" / "sys-B" / "training" / "unet3d"
    dg_b = base_b / "datagen" / "20250101_120000"
    run_b = base_b / "run" / "20250101_130000"
    _write_json(str(dg_b / "metadata.json"), {"args": {"num_processes": 4}})
    _write_json(str(dg_b / "summary.json"), {"num_hosts": 1})
    _write_json(str(run_b / "metadata.json"), {"args": {"num_processes": 8}})
    _write_json(str(run_b / "summary.json"), {"num_hosts": 1})
    _write_systems_yaml(tmp_path, "AcmeB", "sys-B")

    config = Config(
        version="v2.0", submitters=["AcmeA", "AcmeB"], skip_output_file=True
    )
    loader = Loader(root=str(tmp_path), version="v2.0", config=config)

    by_submitter = {}
    for logs in loader.load():
        by_submitter.setdefault(logs.loader_metadata.submitter, []).append(logs)

    assert "AcmeA" in by_submitter, "Submitter A must yield even when datagen/run missing"
    assert "AcmeB" in by_submitter, "Submitter B must be reached after A's missing dirs"
    assert by_submitter["AcmeA"][0].datagen_files == []
    assert by_submitter["AcmeA"][0].run_files == []
    assert len(by_submitter["AcmeB"][0].datagen_files) == 1
    assert len(by_submitter["AcmeB"][0].run_files) == 1
