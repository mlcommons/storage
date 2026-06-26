"""End-to-end tests for TrainingCheck.closed_submission_checksum (§3.6.1).

Plan 04-02 rewrote §3.6.1 from a TODO-stub to a real delegation to
``helpers._check_code_image_layered`` — the same helper VdbCheck §5.6.1
calls. This file is the CD-04 dedup cross-check: the SAME helper is
exercised through a DIFFERENT rule ID, and the violation messages must
carry the §3.6.1 / trainingClosedSubmissionChecksum tag, not the §5.6.1
or §2.1.6 tag. The 5.6.1 side is locked down by test_vdb_checks.py.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlpstorage_py.submission_checker.checks.training_checks import TrainingCheck
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.loader import LoaderMetadata, SubmissionLogs
from mlpstorage_py.submission_checker.tools.code_image import capture_code_image


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _build_training_leaf(
    tmp_path: Path,
    division: str,
    orgname: str,
    system: str,
    *,
    model: str = "unet3d",
    with_code_image: bool = False,
) -> Path:
    """Synthesize a training submission tree under tmp_path.

    Shape:
        <tmp_path>/<division>/<orgname>/results/<system>/training/<model>/
            [<orgname>/code/.code-hash.json   when with_code_image]

    Returns the per-leaf training path.
    """
    leaf = (
        tmp_path
        / division
        / orgname
        / "results"
        / system
        / "training"
        / model
    )
    leaf.mkdir(parents=True, exist_ok=True)

    if with_code_image:
        submitter_dir = tmp_path / division / orgname
        submitter_dir.mkdir(parents=True, exist_ok=True)
        _capture_code_image_at(submitter_dir)

    return leaf


def _capture_code_image_at(target_dir: Path):
    """Capture a synthetic code image at target_dir/code/ (deterministic)."""
    log = MagicMock()
    src = target_dir / "_src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "pyproject.toml").write_text("# stub\n", encoding="utf-8")
    (src / "mod.py").write_text("# mod\n", encoding="utf-8")
    capture_code_image(src, target_dir, log)


def _make_training_check(
    leaf_path: Path,
    division: str,
    log,
    *,
    reference_checksum_override=None,
):
    """Instantiate TrainingCheck against fake SubmissionLogs / LoaderMetadata."""
    config = Config(
        version="v3.0",
        submitters=None,
        skip_output_file=True,
        reference_checksum_override=reference_checksum_override,
    )
    loader_metadata = LoaderMetadata(
        division=division,
        submitter="acme",
        system="sys-1",
        mode="training",
        benchmark="unet3d",
        folder=str(leaf_path),
    )
    submissions_logs = SubmissionLogs(
        datagen_files=[],
        run_files=[],
        system_file=None,
        loader_metadata=loader_metadata,
    )
    return TrainingCheck(log=log, config=config, submissions_logs=submissions_logs)


def _violations(mock_logger, rule_id: str, rule_name: str):
    prefix = "[%s %s]" % (rule_id, rule_name)
    return [m for m in mock_logger.errors if prefix in m]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_closed_training_self_consistent_passes(tmp_path, mock_logger):
    """Self-consistent .code-hash.json under CLOSED → True, no violations."""
    leaf = _build_training_leaf(
        tmp_path, "closed", "acme", "sys-1", with_code_image=True,
    )
    check = _make_training_check(leaf, "closed", mock_logger)
    assert check.closed_submission_checksum() is True
    assert _violations(mock_logger, "3.6.1", "trainingClosedSubmissionChecksum") == []
    assert mock_logger.errors == []
    assert mock_logger.warnings == []


def test_closed_training_self_consistency_violation(tmp_path, mock_logger):
    """Tamper with .code-hash.json → violation tagged 3.6.1 (NOT 5.6.1 / 2.1.6).

    Load-bearing CD-04 wiring proof: the shared helper attributes the
    violation to the caller's rule ID. If a regression hardcoded a rule
    ID inside the helper, this test would fail because the violation
    would carry the wrong tag.
    """
    leaf = _build_training_leaf(
        tmp_path, "closed", "acme", "sys-1", with_code_image=True,
    )
    hash_file = tmp_path / "closed" / "acme" / "code" / ".code-hash.json"
    payload = json.loads(hash_file.read_text(encoding="utf-8"))
    payload["hash"] = "0" * 32
    hash_file.write_text(json.dumps(payload), encoding="utf-8")

    check = _make_training_check(leaf, "closed", mock_logger)
    assert check.closed_submission_checksum() is False
    viol = _violations(mock_logger, "3.6.1", "trainingClosedSubmissionChecksum")
    assert len(viol) == 1, (
        "expected exactly one [3.6.1 trainingClosedSubmissionChecksum] violation; "
        "found %s" % mock_logger.errors
    )
    # Cross-rule guard: must NOT leak into 5.6.1 / 2.1.6 tags.
    assert not _violations(mock_logger, "5.6.1", "vdbClosedSubmissionChecksum"), (
        "3.6.1 violation leaked into 5.6.1 vdbClosedSubmissionChecksum tag"
    )
    assert not _violations(mock_logger, "2.1.6", "codeDirectoryContents"), (
        "3.6.1 violation leaked into 2.1.6 codeDirectoryContents tag"
    )
    assert "code tree hash does not match" in viol[0]


def test_closed_training_upstream_identity_violation_when_reference_set(
    tmp_path, mock_logger,
):
    """REFERENCE_CHECKSUMS override that mismatches → violation tagged 3.6.1.

    Proves the upstream-identity branch of the layered helper is wired
    through the caller's rule ID/name.
    """
    leaf = _build_training_leaf(
        tmp_path, "closed", "acme", "sys-1", with_code_image=True,
    )
    bogus_ref = "ff" * 16
    check = _make_training_check(
        leaf, "closed", mock_logger,
        reference_checksum_override=bogus_ref,
    )
    assert check.closed_submission_checksum() is False
    viol = _violations(mock_logger, "3.6.1", "trainingClosedSubmissionChecksum")
    assert len(viol) == 1, mock_logger.errors
    assert "code tree MD5 mismatch" in viol[0]
    assert bogus_ref in viol[0]
    assert not _violations(mock_logger, "5.6.1", "vdbClosedSubmissionChecksum")
    assert not _violations(mock_logger, "2.1.6", "codeDirectoryContents")


def test_open_training_is_noop(tmp_path, mock_logger):
    """OPEN division → §3.6.1 short-circuits to True; STRUCT-06 self-consistency loop owns OPEN."""
    leaf = _build_training_leaf(
        tmp_path, "open", "acme", "sys-1", with_code_image=True,
    )
    check = _make_training_check(leaf, "open", mock_logger)
    assert check.closed_submission_checksum() is True
    assert mock_logger.errors == []
    assert mock_logger.warnings == []


def test_missing_code_dir_does_not_double_violate(tmp_path, mock_logger):
    """CLOSED with no code/ subdir → §3.6.1 no-ops; §2.1.6 owns VALS-01.

    Guards the design choice from Plan 04-02 Task 1 Step B item 4:
    the missing-code/ structural violation is owned by STRUCT-06 (2.1.6),
    and §3.6.1 must NOT double-count by re-firing.
    """
    leaf = _build_training_leaf(tmp_path, "closed", "acme", "sys-1")
    # No with_code_image=True → submitter dir has no code/.
    check = _make_training_check(leaf, "closed", mock_logger)
    assert check.closed_submission_checksum() is True
    assert _violations(mock_logger, "3.6.1", "trainingClosedSubmissionChecksum") == []
    assert mock_logger.errors == []
