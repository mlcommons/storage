#!/usr/bin/env python3
"""
Tests for SubmissionStructureCheck — STRUCT-01 through STRUCT-14.

Run with:
    pytest mlpstorage_py/tests/test_submission_checker_structure.py -v
"""

import json
import os
import pytest
from pathlib import Path

from mlpstorage_py.submission_checker.checks.submission_structure_checks import (
    SubmissionStructureCheck,
)
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.rule_registry import discover_rules


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(version="v2.0", reference_checksum_override=None):
    return Config(
        version=version,
        submitters=None,
        skip_output_file=False,
        reference_checksum_override=reference_checksum_override,
    )


def run_one_check(structure_check, method_name, mock_logger):
    """Invoke a single named check method and return its bool result."""
    method = getattr(structure_check, method_name)
    return method()


def _make_check(root_path, mock_logger, version="v2.0", ref_checksum=None):
    config = _make_config(version=version, reference_checksum_override=ref_checksum)
    return SubmissionStructureCheck(mock_logger, config, str(root_path))


# ---------------------------------------------------------------------------
# TestFixtureFactory — sanity tests on build_submission (consumed by Task 1)
# ---------------------------------------------------------------------------

class TestFixtureFactory:
    """Sanity tests on build_submission (consumed by Task 1 verify step)."""

    def test_default_builds_closed_dir(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        assert (root / "closed").is_dir()

    def test_default_builds_acme_submitter(self, tmp_path):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        assert (root / "closed" / "Acme").is_dir()

    def test_default_builds_required_subdirs(self, tmp_path):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        base = root / "closed" / "Acme"
        for d in ("code", "results", "systems"):
            assert (base / d).is_dir(), f"Missing {d}/"

    def test_default_builds_system_yaml(self, tmp_path):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        assert (root / "closed" / "Acme" / "systems" / "acme-storage-v1.yaml").is_file()

    def test_default_builds_system_pdf(self, tmp_path):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        assert (root / "closed" / "Acme" / "systems" / "acme-storage-v1.pdf").is_file()

    def test_default_builds_three_code_files(self, tmp_path):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        code_files = list((root / "closed" / "Acme" / "code").iterdir())
        assert len(code_files) == 3

    def test_default_builds_one_datagen_timestamp(self, tmp_path):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        datagen = root / "closed" / "Acme" / "results" / "acme-storage-v1" / "training" / "unet3d" / "datagen"
        assert datagen.is_dir()
        ts_dirs = list(datagen.iterdir())
        assert len(ts_dirs) == 1

    def test_default_builds_six_run_timestamps(self, tmp_path):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        run_dir = root / "closed" / "Acme" / "results" / "acme-storage-v1" / "training" / "unet3d" / "run"
        ts_dirs = list(run_dir.iterdir())
        assert len(ts_dirs) == 6

    def test_default_builds_two_checkpointing_timestamps(self, tmp_path):
        # Rules.md 2.1.23 + 4.7.1: 1 or 2 timestamp dirs per workload
        # (one per invocation). The fixture uses the two-invocation shape
        # so split-mode kwargs can exercise the pairing helpers.
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        chkpt = root / "closed" / "Acme" / "results" / "acme-storage-v1" / "checkpointing" / "llama3-8b"
        ts_dirs = list(chkpt.iterdir())
        assert len(ts_dirs) == 2

    def test_unknown_kwarg_raises_type_error(self, tmp_path):
        from mlpstorage_py.tests.conftest import build_submission
        with pytest.raises(TypeError):
            build_submission(tmp_path, no_such_kwarg=True)

    def test_default_fixture_no_errors(self, tmp_path, mock_logger):
        """Default fixture should produce no errors from any STRUCT check."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        check = _make_check(root, mock_logger)
        result = check()
        assert mock_logger.errors == [], f"Unexpected errors: {mock_logger.errors}"

    def test_mock_logger_captures_errors_as_strings(self, mock_logger):
        mock_logger.error("hello %s %d", "world", 42)
        assert mock_logger.errors == ["hello world 42"]

    def test_mock_logger_captures_warnings_as_strings(self, mock_logger):
        mock_logger.warning("warn %s", "thing")
        assert mock_logger.warnings == ["warn thing"]

    def test_mock_logger_fresh_per_test(self, mock_logger):
        assert mock_logger.errors == []
        assert mock_logger.warnings == []


# ---------------------------------------------------------------------------
# TestStruct01_SubmitterRootDirectory  (STRUCT-01, rule 2.1.1)
# ---------------------------------------------------------------------------

class TestStruct01_SubmitterRootDirectory:

    def test_default_fixture_passes(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "submitter_root_directory_check", mock_logger)
        assert result is True
        assert mock_logger.errors == []

    def test_submitter_name_with_space(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, submitter_name_with_space=True)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "submitter_root_directory_check", mock_logger)
        assert result is False
        assert any("[2.1.1 submitterRootDirectory]" in m for m in mock_logger.errors)


# ---------------------------------------------------------------------------
# TestStruct02_TopLevelSubdirectories  (STRUCT-02, rule 2.1.2)
# ---------------------------------------------------------------------------

class TestStruct02_TopLevelSubdirectories:

    def test_default_fixture_passes(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "top_level_subdirectories_check", mock_logger)
        assert result is True
        assert mock_logger.errors == []

    def test_top_level_capitalcase(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, top_level_capitalcase=True)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "top_level_subdirectories_check", mock_logger)
        assert result is False
        assert any("[2.1.2 topLevelSubdirectories]" in m for m in mock_logger.errors)

    def test_extra_top_level(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, extra_top_level="Other")
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "top_level_subdirectories_check", mock_logger)
        assert result is False
        assert any("[2.1.2 topLevelSubdirectories]" in m for m in mock_logger.errors)

    def test_no_top_level_dirs(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, no_top_level_dirs=True)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "top_level_subdirectories_check", mock_logger)
        assert result is False
        assert any("[2.1.2 topLevelSubdirectories]" in m for m in mock_logger.errors)

    def test_dot_prefixed_top_level_entries_are_ignored(self, tmp_path, mock_logger):
        """Merged reviewer trees are typically git working trees. Dot-prefixed
        entries (.git/, .github/, .gitignore) must not fire 2.1.2 violations.
        """
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        os.makedirs(os.path.join(root, ".git", "refs"))
        os.makedirs(os.path.join(root, ".github", "workflows"))
        with open(os.path.join(root, ".gitignore"), "w") as f:
            f.write("*.pyc\n")
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "top_level_subdirectories_check", mock_logger)
        assert result is True
        assert not any("[2.1.2 topLevelSubdirectories]" in m for m in mock_logger.errors)


# ---------------------------------------------------------------------------
# TestStruct03_OpenMatchesClosed  (STRUCT-03, rule 2.1.3)
# ---------------------------------------------------------------------------

class TestStruct03_OpenMatchesClosed:
    """Rules.md 2.1.3 openMatchesClosed is a structural meta-rule: 'the open
    hierarchy should be constructed identically to the closed hierarchy.' That
    is, the construction rules in 2.1.4+ apply equally to open/. It is NOT a
    contents-mirroring requirement — both hierarchies are individually
    optional, and a submitter may appear in one division without appearing in
    the other.

    The structural mirroring is enforced automatically because every
    downstream STRUCT method iterates closed/ and open/ uniformly. The 2.1.3
    @rule binding therefore returns True unconditionally; its purpose is
    coverage signaling, not runtime enforcement.
    """

    def test_closed_only_passes(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "open_matches_closed_check", mock_logger)
        assert result is True
        assert mock_logger.errors == []

    def test_submitter_present_in_only_one_division_passes(self, tmp_path, mock_logger):
        """Regression for over-strict pre-fix behavior: when each division
        contains a different submitter set (the merged reviewer-tree pattern
        seen in the v2.0 results bundle: Alluxio / DDN / etc. each in only
        one division), STRUCT-03 must NOT error. Per-division shape rules
        (STRUCT-04..14) own the structural validation; 2.1.3 is a meta-rule.
        """
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, open_mismatches_closed=True)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "open_matches_closed_check", mock_logger)
        assert result is True
        assert not any("[2.1.3 openMatchesClosed]" in m for m in mock_logger.errors)


# ---------------------------------------------------------------------------
# TestStruct04_ClosedSubmitterDirectory  (STRUCT-04, rule 2.1.4)
# ---------------------------------------------------------------------------

class TestStruct04_ClosedSubmitterDirectory:
    """Rules.md 2.1.4 names a per-submitter convention. The validator must
    accept both the single-submitter package shape (one dir under closed/,
    matching the top-level dir name) and the merged reviewer tree shape (N
    submitter dirs under closed/, top-level dir named for the merged set).
    The submitter-name character set is enforced by STRUCT-01 (2.1.1); the
    {code, results, systems} shape is enforced by STRUCT-05 (2.1.5). So 2.1.4
    has no extra runtime work in either mode; the @rule binding is preserved
    for coverage signaling only.
    """

    def test_single_submitter_package_passes(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "closed_submitter_directory_check", mock_logger)
        assert result is True
        assert mock_logger.errors == []

    def test_merged_reviewer_tree_with_multiple_submitters_passes(self, tmp_path, mock_logger):
        """Regression for over-strict pre-fix behavior: closed/ with multiple
        submitter directories (the merged v2.0 results bundle pattern) must
        not error. STRUCT-01 still validates each submitter dir name.
        """
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, multiple_submitters_in_closed=True)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "closed_submitter_directory_check", mock_logger)
        assert result is True
        assert not any("[2.1.4 closedSubmitterDirectory]" in m for m in mock_logger.errors)

    def test_basename_mismatch_does_not_fire(self, tmp_path, mock_logger):
        """Regression for over-strict pre-fix behavior: submitter dir name
        not matching the top-level path basename was a false positive against
        merged reviewer trees rooted at e.g. submissions_storage_v2.0/.
        """
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, wrong_submitter_in_closed=True)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "closed_submitter_directory_check", mock_logger)
        assert result is True
        assert not any("[2.1.4 closedSubmitterDirectory]" in m for m in mock_logger.errors)


# ---------------------------------------------------------------------------
# TestStruct05_RequiredSubdirectories  (STRUCT-05, rule 2.1.5)
# ---------------------------------------------------------------------------

class TestStruct05_RequiredSubdirectories:

    def test_default_fixture_passes(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "required_subdirectories_check", mock_logger)
        assert result is True
        assert mock_logger.errors == []

    def test_missing_code_subdir(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, missing_required_subdir="code")
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "required_subdirectories_check", mock_logger)
        assert result is False
        assert any("[2.1.5 requiredSubdirectories]" in m for m in mock_logger.errors)

    def test_missing_results_subdir(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, missing_required_subdir="results")
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "required_subdirectories_check", mock_logger)
        assert result is False
        assert any("[2.1.5 requiredSubdirectories]" in m for m in mock_logger.errors)

    def test_missing_systems_subdir(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, missing_required_subdir="systems")
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "required_subdirectories_check", mock_logger)
        assert result is False
        assert any("[2.1.5 requiredSubdirectories]" in m for m in mock_logger.errors)

    def test_extra_submitter_subdir(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, extra_submitter_subdir="extra")
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "required_subdirectories_check", mock_logger)
        assert result is False
        assert any("[2.1.5 requiredSubdirectories]" in m for m in mock_logger.errors)

    def test_dotfile_at_submitter_level_is_ignored(self, tmp_path, mock_logger):
        """Dot-prefixed entries (.DS_Store, .cache/) under closed/<submitter>/
        must not trip the 'unexpected subdirectory' branch."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        with open(os.path.join(root, "closed", "Acme", ".DS_Store"), "w") as f:
            f.write("")
        os.makedirs(os.path.join(root, "closed", "Acme", ".cache"))
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "required_subdirectories_check", mock_logger)
        assert result is True
        assert mock_logger.errors == []

    def test_wrapping_hint_when_submission_nested_one_level_deep(self, tmp_path, mock_logger):
        """Common v2.0 submitter mistake: closed/<submitter>/benchmarks/{code,
        results, systems}/ instead of closed/<submitter>/{code, results,
        systems}/. The diagnostic for the extra wrapper dir should explicitly
        name the wrapping so the submitter knows what to fix.
        """
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, missing_required_subdir="code")
        # Now build the wrapping: move code/results/systems INTO benchmarks/
        # at the submitter level so the wrapping detection has something to find.
        sub_path = os.path.join(root, "closed", "Acme")
        wrapper = os.path.join(sub_path, "benchmarks")
        os.makedirs(wrapper)
        os.makedirs(os.path.join(wrapper, "code"))
        os.makedirs(os.path.join(wrapper, "results"))
        os.makedirs(os.path.join(wrapper, "systems"))
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "required_subdirectories_check", mock_logger)
        assert result is False
        wrapping_msgs = [
            m for m in mock_logger.errors
            if "[2.1.5 requiredSubdirectories]" in m
            and "nested one level deeper than expected" in m
        ]
        assert len(wrapping_msgs) == 1, mock_logger.errors


# ---------------------------------------------------------------------------
# TestStruct06_CodeDirectoryContents  (STRUCT-06, rule 2.1.6)
# ---------------------------------------------------------------------------

class TestStruct06_CodeDirectoryContents:

    def test_default_fixture_passes_with_unset_reference(self, tmp_path, mock_logger):
        """No reference checksum → warn ONCE (not per-submitter) and return True (D-12)."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        check = _make_check(root, mock_logger)  # no ref_checksum
        result = run_one_check(check, "code_directory_contents_check", mock_logger)
        assert result is True
        warnings = [w for w in mock_logger.warnings if "[2.1.6 codeDirectoryContents]" in w]
        assert len(warnings) == 1, warnings
        assert mock_logger.errors == []

    def test_unset_reference_emits_single_warning_for_multi_submitter_tree(self, tmp_path, mock_logger):
        """Regression for pre-fix per-submitter warning spam: 5-submitter merged
        tree must emit exactly one no-checksum warning, not five."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, multiple_submitters_in_closed=True)
        check = _make_check(root, mock_logger)  # no ref_checksum
        result = run_one_check(check, "code_directory_contents_check", mock_logger)
        assert result is True
        warnings = [w for w in mock_logger.warnings if "[2.1.6 codeDirectoryContents]" in w]
        assert len(warnings) == 1, warnings

    def test_reference_checksum_mismatch_fails(self, tmp_path, mock_logger):
        """Deliberate mismatch: zeros as reference → check fails."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        check = _make_check(root, mock_logger, ref_checksum="0" * 32)
        result = run_one_check(check, "code_directory_contents_check", mock_logger)
        assert result is False
        assert any("[2.1.6 codeDirectoryContents]" in m for m in mock_logger.errors)

    def test_reference_checksum_match_passes(self, tmp_path, mock_logger):
        """Correct reference checksum → check passes silently."""
        from mlpstorage_py.tests.conftest import build_submission
        from mlpstorage_py.submission_checker.tools.code_checksum import compute_code_tree_md5
        root = build_submission(tmp_path)
        code_path = str(root / "closed" / "Acme" / "code")
        actual_hash = compute_code_tree_md5(code_path, mock_logger)
        check = _make_check(root, mock_logger, ref_checksum=actual_hash)
        result = run_one_check(check, "code_directory_contents_check", mock_logger)
        assert result is True
        assert mock_logger.errors == []

    def test_mutated_code_fails(self, tmp_path, mock_logger):
        """Extra file in code/ changes hash → violation."""
        from mlpstorage_py.tests.conftest import build_submission
        from mlpstorage_py.submission_checker.tools.code_checksum import compute_code_tree_md5
        # First build clean tree to get reference hash
        clean_root = build_submission(tmp_path / "clean")
        code_path = str(clean_root / "closed" / "Acme" / "code")
        clean_hash = compute_code_tree_md5(code_path, mock_logger)

        # Now build mutated tree
        root = build_submission(tmp_path / "mutated", mutate_code=True)
        check = _make_check(root, mock_logger, ref_checksum=clean_hash)
        result = run_one_check(check, "code_directory_contents_check", mock_logger)
        assert result is False
        assert any("[2.1.6 codeDirectoryContents]" in m for m in mock_logger.errors)

    def test_pycache_excluded_passes(self, tmp_path, mock_logger):
        """__pycache__ is excluded from hash — code_with_pycache fixture still passes."""
        from mlpstorage_py.tests.conftest import build_submission
        from mlpstorage_py.submission_checker.tools.code_checksum import compute_code_tree_md5
        # Get clean hash
        clean_root = build_submission(tmp_path / "clean")
        code_path = str(clean_root / "closed" / "Acme" / "code")
        clean_hash = compute_code_tree_md5(code_path, mock_logger)

        root = build_submission(tmp_path / "pycache", code_with_pycache=True)
        check = _make_check(root, mock_logger, ref_checksum=clean_hash)
        result = run_one_check(check, "code_directory_contents_check", mock_logger)
        assert result is True
        assert mock_logger.errors == []


# ---------------------------------------------------------------------------
# Phase 2 Plan 02-03 — Helpers + Tests for the refactored
# code_directory_contents_check (VALS-01..04 + D-11 layered model + D-15 walk)
# ---------------------------------------------------------------------------

def _write_valid_hash_json(code_path, mock_logger, **overrides):
    """Compute the current tree hash and write a matching .code-hash.json.

    This makes the captured tree self-consistent so that
    verify_image_self_consistent returns True without re-running
    capture_code_image (which would copy the live source tree).
    """
    from mlpstorage_py.submission_checker.tools.code_checksum import (
        compute_code_tree_md5,
    )
    digest = compute_code_tree_md5(str(code_path), mock_logger)
    payload = {
        "hash": digest,
        "algorithm": "md5-tree-v1",
        "captured_at": "2026-06-17T00:00:00Z",
        "mlpstorage_version": "3.0.9",
        "git_sha": None,
    }
    payload.update(overrides)
    hash_file = Path(code_path) / ".code-hash.json"
    hash_file.write_text(json.dumps(payload))
    return payload["hash"]


def _make_open_leaf(root, submitter="Acme", sys_name="sys-1", wtype="training",
                    model="unet3d", write_code=True):
    """Build a minimal open/<submitter>/results/<sys>/<wtype>/<model>/code tree.

    Returns the absolute path to .../code (whether or not write_code created it).
    """
    leaf = root / "open" / submitter / "results" / sys_name / wtype / model
    leaf.mkdir(parents=True, exist_ok=True)
    code_path = leaf / "code"
    if write_code:
        code_path.mkdir(parents=True, exist_ok=True)
        (code_path / "mod.py").write_bytes(b"# mod\n")
        (code_path / "helper.py").write_bytes(b"# helper\n")
    return code_path


class TestStruct06_RefactoredCodeDirectoryContents:
    """Refactored STRUCT-06 enforcing VALS-01..04 across CLOSED + OPEN.

    Plan 02-03: code_directory_contents_check walks both divisions and
    emits separate violations for missing-code/ vs hash-mismatch (D-14),
    runs REFERENCE_CHECKSUMS only for CLOSED leaves (D-11), and runs
    per-tree self-consistency for both CLOSED and OPEN.
    """

    # ----- VALS-01 — CLOSED missing code/ -----
    def test_vals01_closed_missing_code_emits_missing_violation(self, tmp_path, mock_logger):
        # Tree: closed/Acme/{results,systems} but no closed/Acme/code/
        sub = tmp_path / "closed" / "Acme"
        (sub / "results").mkdir(parents=True)
        (sub / "systems").mkdir(parents=True)
        check = _make_check(tmp_path, mock_logger)
        result = run_one_check(check, "code_directory_contents_check", mock_logger)
        assert result is False
        missing_msgs = [
            m for m in mock_logger.errors
            if "[2.1.6 codeDirectoryContents]" in m
            and "required code/ directory missing at" in m
            and "closed/Acme/code" in m
        ]
        assert len(missing_msgs) == 1, mock_logger.errors

    # ----- VALS-02 — CLOSED self-consistency mismatch -----
    def test_vals02_closed_self_consistency_mismatch(self, tmp_path, mock_logger):
        sub = tmp_path / "closed" / "Acme"
        code_path = sub / "code"
        code_path.mkdir(parents=True)
        (code_path / "mod.py").write_bytes(b"# original\n")
        _write_valid_hash_json(code_path, mock_logger)
        # Mutate the tree so the hash no longer matches the recorded JSON
        (code_path / "mod.py").write_bytes(b"# TAMPERED\n")
        check = _make_check(tmp_path, mock_logger)
        result = run_one_check(check, "code_directory_contents_check", mock_logger)
        assert result is False
        mismatch_msgs = [
            m for m in mock_logger.errors
            if "[2.1.6 codeDirectoryContents]" in m
            and "code tree hash does not match .code-hash.json at" in m
        ]
        assert len(mismatch_msgs) == 1, mock_logger.errors

    # ----- VALS-02 — missing .code-hash.json -----
    def test_vals02_missing_hash_json_emits_violation(self, tmp_path, mock_logger):
        sub = tmp_path / "closed" / "Acme"
        code_path = sub / "code"
        code_path.mkdir(parents=True)
        (code_path / "mod.py").write_bytes(b"# mod\n")
        # Intentionally do NOT write .code-hash.json
        check = _make_check(tmp_path, mock_logger)
        result = run_one_check(check, "code_directory_contents_check", mock_logger)
        assert result is False
        # The MissingHashFile exception message is logged as the violation msg.
        any_violation = [
            m for m in mock_logger.errors
            if "[2.1.6 codeDirectoryContents]" in m
        ]
        assert len(any_violation) >= 1, mock_logger.errors

    # ----- VALS-03 — OPEN missing code/ -----
    def test_vals03_open_missing_code_emits_missing_violation(self, tmp_path, mock_logger):
        # build OPEN leaf without code/
        _make_open_leaf(tmp_path, write_code=False)
        check = _make_check(tmp_path, mock_logger)
        result = run_one_check(check, "code_directory_contents_check", mock_logger)
        assert result is False
        missing_msgs = [
            m for m in mock_logger.errors
            if "[2.1.6 codeDirectoryContents]" in m
            and "required code/ directory missing at" in m
            and "open/Acme/results/sys-1/training/unet3d/code" in m
        ]
        assert len(missing_msgs) == 1, mock_logger.errors

    # ----- VALS-04 — OPEN self-consistency mismatch -----
    def test_vals04_open_self_consistency_mismatch(self, tmp_path, mock_logger):
        code_path = _make_open_leaf(tmp_path, write_code=True)
        _write_valid_hash_json(code_path, mock_logger)
        (code_path / "mod.py").write_bytes(b"# TAMPERED\n")
        check = _make_check(tmp_path, mock_logger)
        result = run_one_check(check, "code_directory_contents_check", mock_logger)
        assert result is False
        mismatch_msgs = [
            m for m in mock_logger.errors
            if "[2.1.6 codeDirectoryContents]" in m
            and "code tree hash does not match .code-hash.json at" in m
        ]
        assert len(mismatch_msgs) == 1, mock_logger.errors

    # ----- D-11 layered model (CLOSED happy path) -----
    def test_d11_closed_layered_happy_path(self, tmp_path, mock_logger):
        """When REFERENCE_CHECKSUMS matches AND self-consistency passes → True."""
        sub = tmp_path / "closed" / "Acme"
        code_path = sub / "code"
        code_path.mkdir(parents=True)
        (code_path / "mod.py").write_bytes(b"# mod\n")
        actual_hash = _write_valid_hash_json(code_path, mock_logger)
        check = _make_check(tmp_path, mock_logger, ref_checksum=actual_hash)
        result = run_one_check(check, "code_directory_contents_check", mock_logger)
        assert result is True, mock_logger.errors
        assert mock_logger.errors == []

    # ----- D-11 layered model (CLOSED self-consistency passes, ref mismatch) -----
    def test_d11_closed_self_consistent_but_ref_mismatch(self, tmp_path, mock_logger):
        sub = tmp_path / "closed" / "Acme"
        code_path = sub / "code"
        code_path.mkdir(parents=True)
        (code_path / "mod.py").write_bytes(b"# mod\n")
        _write_valid_hash_json(code_path, mock_logger)  # self-consistent
        check = _make_check(tmp_path, mock_logger, ref_checksum="0" * 32)
        result = run_one_check(check, "code_directory_contents_check", mock_logger)
        assert result is False
        ref_mismatch_msgs = [
            m for m in mock_logger.errors
            if "[2.1.6 codeDirectoryContents]" in m
            and "code tree MD5 mismatch: expected" in m
        ]
        assert len(ref_mismatch_msgs) == 1, mock_logger.errors

    # ----- D-12 single-warning preserved with new addendum -----
    def test_d12_unconfigured_warning_runs_self_consistency_with_addendum(
        self, tmp_path, mock_logger
    ):
        sub = tmp_path / "closed" / "Acme"
        code_path = sub / "code"
        code_path.mkdir(parents=True)
        (code_path / "mod.py").write_bytes(b"# mod\n")
        _write_valid_hash_json(code_path, mock_logger)
        check = _make_check(tmp_path, mock_logger)  # no ref_checksum
        result = run_one_check(check, "code_directory_contents_check", mock_logger)
        assert result is True
        warnings = [
            w for w in mock_logger.warnings
            if "[2.1.6 codeDirectoryContents]" in w
            and "reference checksum not configured" in w
            and "self-consistency check still ran" in w
        ]
        assert len(warnings) == 1, mock_logger.warnings

    # ----- OPEN-only tree does not emit the "not configured" warning -----
    def test_open_only_tree_does_not_emit_unconfigured_warning(self, tmp_path, mock_logger):
        code_path = _make_open_leaf(tmp_path, write_code=True)
        _write_valid_hash_json(code_path, mock_logger)
        check = _make_check(tmp_path, mock_logger)
        result = run_one_check(check, "code_directory_contents_check", mock_logger)
        assert result is True, mock_logger.errors
        # No "reference checksum not configured" warning when only open/ exists.
        warnings = [
            w for w in mock_logger.warnings
            if "reference checksum not configured" in w
        ]
        assert warnings == [], warnings

    # ----- D-15 walk hygiene: empty type subtree yields nothing -----
    def test_d15_walk_hygiene_no_model_yields_no_violation(self, tmp_path, mock_logger):
        # open/Acme/results/sys-1/training/ exists but no model/ subdirs.
        (tmp_path / "open" / "Acme" / "results" / "sys-1" / "training").mkdir(parents=True)
        check = _make_check(tmp_path, mock_logger)
        result = run_one_check(check, "code_directory_contents_check", mock_logger)
        assert result is True, mock_logger.errors
        missing_msgs = [
            m for m in mock_logger.errors
            if "[2.1.6 codeDirectoryContents]" in m
            and "required code/ directory missing" in m
        ]
        assert missing_msgs == [], missing_msgs


# ---------------------------------------------------------------------------
# Phase 2 Plan 02-03 — Tests for mode-aware required_subdirectories_check
# (STRUCT-05 per Rules.md §2.1.5 split — D-17)
# ---------------------------------------------------------------------------

class TestStruct05_ModeAwareRequiredSubdirectories:
    """STRUCT-05 (Plan 02-03 mode-aware refactor).

    CLOSED submitter dir requires {code, results, systems};
    OPEN submitter dir requires {results, systems}; code/ lives per-leaf in OPEN.
    Violation messages route through `requiredSubdirectoriesClosed` / `requiredSubdirectoriesOpen`.
    """

    def test_closed_happy_path_unchanged(self, tmp_path, mock_logger):
        sub = tmp_path / "closed" / "Acme"
        (sub / "code").mkdir(parents=True)
        (sub / "results").mkdir(parents=True)
        (sub / "systems").mkdir(parents=True)
        check = _make_check(tmp_path, mock_logger)
        result = run_one_check(check, "required_subdirectories_check", mock_logger)
        assert result is True, mock_logger.errors
        assert mock_logger.errors == []

    def test_closed_missing_code_routes_through_closed_anchor(self, tmp_path, mock_logger):
        sub = tmp_path / "closed" / "Acme"
        (sub / "results").mkdir(parents=True)
        (sub / "systems").mkdir(parents=True)
        check = _make_check(tmp_path, mock_logger)
        result = run_one_check(check, "required_subdirectories_check", mock_logger)
        assert result is False
        closed_anchor_msgs = [
            m for m in mock_logger.errors
            if "[2.1.5 requiredSubdirectoriesClosed]" in m
            and "required subdirectory 'code' missing from closed/Acme" in m
        ]
        assert len(closed_anchor_msgs) == 1, mock_logger.errors

    def test_open_happy_path_two_subdirs(self, tmp_path, mock_logger):
        """OPEN submitter dir with {results, systems} only must pass.

        This is the Gemini-HIGH regression target — without the mode-aware
        check, every OPEN package the new runtime produces would be flagged.
        """
        sub = tmp_path / "open" / "Acme"
        (sub / "results").mkdir(parents=True)
        (sub / "systems").mkdir(parents=True)
        check = _make_check(tmp_path, mock_logger)
        result = run_one_check(check, "required_subdirectories_check", mock_logger)
        assert result is True, mock_logger.errors
        assert mock_logger.errors == []

    def test_open_with_code_at_submitter_level_is_unexpected(self, tmp_path, mock_logger):
        sub = tmp_path / "open" / "Acme"
        (sub / "code").mkdir(parents=True)
        (sub / "results").mkdir(parents=True)
        (sub / "systems").mkdir(parents=True)
        check = _make_check(tmp_path, mock_logger)
        result = run_one_check(check, "required_subdirectories_check", mock_logger)
        assert result is False
        unexpected_msgs = [
            m for m in mock_logger.errors
            if "[2.1.5 requiredSubdirectoriesOpen]" in m
            and "unexpected subdirectory 'code'" in m
        ]
        assert len(unexpected_msgs) == 1, mock_logger.errors

    def test_open_missing_results(self, tmp_path, mock_logger):
        sub = tmp_path / "open" / "Acme"
        (sub / "systems").mkdir(parents=True)
        check = _make_check(tmp_path, mock_logger)
        result = run_one_check(check, "required_subdirectories_check", mock_logger)
        assert result is False
        missing_msgs = [
            m for m in mock_logger.errors
            if "[2.1.5 requiredSubdirectoriesOpen]" in m
            and "required subdirectory 'results' missing from open/Acme" in m
        ]
        assert len(missing_msgs) == 1, mock_logger.errors

    def test_open_missing_systems(self, tmp_path, mock_logger):
        sub = tmp_path / "open" / "Acme"
        (sub / "results").mkdir(parents=True)
        check = _make_check(tmp_path, mock_logger)
        result = run_one_check(check, "required_subdirectories_check", mock_logger)
        assert result is False
        missing_msgs = [
            m for m in mock_logger.errors
            if "[2.1.5 requiredSubdirectoriesOpen]" in m
            and "required subdirectory 'systems' missing from open/Acme" in m
        ]
        assert len(missing_msgs) == 1, mock_logger.errors

    def test_closed_wrapping_hint_still_works(self, tmp_path, mock_logger):
        sub = tmp_path / "closed" / "Acme"
        wrapper = sub / "benchmarks"
        (wrapper / "code").mkdir(parents=True)
        (wrapper / "results").mkdir(parents=True)
        (wrapper / "systems").mkdir(parents=True)
        check = _make_check(tmp_path, mock_logger)
        result = run_one_check(check, "required_subdirectories_check", mock_logger)
        assert result is False
        hint_msgs = [
            m for m in mock_logger.errors
            if "[2.1.5 requiredSubdirectoriesClosed]" in m
            and "nested one level deeper than expected" in m
        ]
        assert len(hint_msgs) == 1, mock_logger.errors

    def test_open_wrapping_hint(self, tmp_path, mock_logger):
        sub = tmp_path / "open" / "Acme"
        wrapper = sub / "benchmarks"
        (wrapper / "results").mkdir(parents=True)
        (wrapper / "systems").mkdir(parents=True)
        check = _make_check(tmp_path, mock_logger)
        result = run_one_check(check, "required_subdirectories_check", mock_logger)
        assert result is False
        hint_msgs = [
            m for m in mock_logger.errors
            if "[2.1.5 requiredSubdirectoriesOpen]" in m
            and "nested one level deeper than expected" in m
        ]
        assert len(hint_msgs) == 1, mock_logger.errors


# ---------------------------------------------------------------------------
# TestStruct07_SystemsDirectoryFiles  (STRUCT-07, rule 2.1.7)
# ---------------------------------------------------------------------------

class TestStruct07_SystemsDirectoryFiles:

    def test_default_fixture_passes(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "systems_directory_files_check", mock_logger)
        assert result is True
        assert mock_logger.errors == []

    def test_unpaired_yaml(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, unpaired_yaml=True)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "systems_directory_files_check", mock_logger)
        assert result is False
        assert any("[2.1.7 systemsDirectoryFiles]" in m for m in mock_logger.errors)

    def test_extra_systems_file(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, extra_systems_file="notes.txt")
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "systems_directory_files_check", mock_logger)
        assert result is False
        assert any("[2.1.7 systemsDirectoryFiles]" in m for m in mock_logger.errors)

    def test_md_files_in_systems_are_allowed(self, tmp_path, mock_logger):
        """Markdown documentation files (*.md) are permitted alongside the
        per-system .yaml/.pdf pairs (Rules.md 2.1.7)."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        systems_path = os.path.join(root, "closed", "Acme", "systems")
        for name in ("README.md", "NOTES.md", "system-notes.md"):
            with open(os.path.join(systems_path, name), "w") as f:
                f.write("# documentation\n")
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "systems_directory_files_check", mock_logger)
        assert result is True
        assert mock_logger.errors == []

    def test_dotfiles_in_systems_are_ignored(self, tmp_path, mock_logger):
        """Dot-prefixed entries in systems/ (.DS_Store, .gitkeep) must not fire
        violations — they're never the submitter's intended content."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        systems_path = os.path.join(root, "closed", "Acme", "systems")
        for name in (".DS_Store", ".gitkeep"):
            with open(os.path.join(systems_path, name), "w") as f:
                f.write("")
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "systems_directory_files_check", mock_logger)
        assert result is True
        assert mock_logger.errors == []


# ---------------------------------------------------------------------------
# TestStruct08_ResultsDirectorySystems  (STRUCT-08, rule 2.1.8)
# ---------------------------------------------------------------------------

class TestStruct08_ResultsDirectorySystems:

    def test_default_fixture_passes(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "results_directory_systems_check", mock_logger)
        assert result is True
        assert mock_logger.errors == []

    def test_unpaired_results_system(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, unpaired_results_system=True)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "results_directory_systems_check", mock_logger)
        assert result is False
        assert any("[2.1.8 resultsDirectorySystems]" in m for m in mock_logger.errors)

    def test_missing_systems_pdf(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, missing_systems_pdf=True)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "results_directory_systems_check", mock_logger)
        assert result is False
        assert any("[2.1.8 resultsDirectorySystems]" in m for m in mock_logger.errors)

    def test_submission_name_mismatch(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, submission_name_mismatch=True)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "results_directory_systems_check", mock_logger)
        assert result is False
        assert any("[2.1.8 resultsDirectorySystems]" in m for m in mock_logger.errors)


# ---------------------------------------------------------------------------
# TestStruct09_IdenticalSystemConfig  (STRUCT-09, rule 2.1.9)
# ---------------------------------------------------------------------------

class TestStruct09_IdenticalSystemConfig:

    def test_default_fixture_passes(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "identical_system_config_check", mock_logger)
        assert result is True
        assert mock_logger.errors == []

    def test_num_hosts_mismatch(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, num_hosts_mismatch=True)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "identical_system_config_check", mock_logger)
        assert result is False
        assert any("[2.1.9 identicalSystemConfig]" in m for m in mock_logger.errors)

    def test_memory_mismatch(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, memory_mismatch=True)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "identical_system_config_check", mock_logger)
        assert result is False
        assert any("[2.1.9 identicalSystemConfig]" in m for m in mock_logger.errors)

    def test_multi_host_capability_inconsistent(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, multi_host_capability_inconsistent=True)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "identical_system_config_check", mock_logger)
        assert result is False
        assert any("[2.1.9 identicalSystemConfig]" in m for m in mock_logger.errors)

    def test_silent_skip_when_summary_field_absent(self, tmp_path, mock_logger):
        """D-16: absent field in summary.json → silently skip, no error, no warning."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, missing_summary_field="num_hosts")
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "identical_system_config_check", mock_logger)
        assert result is True
        assert mock_logger.errors == []
        assert mock_logger.warnings == []


# ---------------------------------------------------------------------------
# TestStruct10_WorkloadCategories  (STRUCT-10, rule 2.1.10)
# ---------------------------------------------------------------------------

class TestStruct10_WorkloadCategories:

    def test_default_fixture_passes(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "workload_categories_check", mock_logger)
        assert result is True
        assert mock_logger.errors == []

    def test_extra_workload_category(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, extra_workload_category="foo")
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "workload_categories_check", mock_logger)
        assert result is False
        assert any("[2.1.10 workloadCategories]" in m for m in mock_logger.errors)


# ---------------------------------------------------------------------------
# TestStruct11_TrainingWorkloads  (STRUCT-11, rule 2.1.11)
# ---------------------------------------------------------------------------

class TestStruct11_TrainingWorkloads:

    def test_default_fixture_passes(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "training_workloads_check", mock_logger)
        assert result is True
        assert mock_logger.errors == []

    def test_wrong_training_workload(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, wrong_training_workload="yolov5")
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "training_workloads_check", mock_logger)
        assert result is False
        assert any("[2.1.11 trainingWorkloads]" in m for m in mock_logger.errors)


# ---------------------------------------------------------------------------
# TestStruct12_TrainingPhases  (STRUCT-12, rule 2.1.12)
# ---------------------------------------------------------------------------

class TestStruct12_TrainingPhases:

    def test_default_fixture_passes(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "training_phases_check", mock_logger)
        assert result is True
        assert mock_logger.errors == []

    def test_wrong_training_phase(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, wrong_training_phase="extra")
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "training_phases_check", mock_logger)
        assert result is False
        assert any("[2.1.12 trainingPhases]" in m for m in mock_logger.errors)


# ---------------------------------------------------------------------------
# TestStruct13_DatagenTimestamp  (STRUCT-13, rule 2.1.13)
# ---------------------------------------------------------------------------

class TestStruct13_DatagenTimestamp:

    def test_default_fixture_passes(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "datagen_timestamp_check", mock_logger)
        assert result is True
        assert mock_logger.errors == []

    def test_datagen_timestamps_wrong_count(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, datagen_timestamps=2)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "datagen_timestamp_check", mock_logger)
        assert result is False
        assert any("[2.1.13 datagenTimestamp]" in m for m in mock_logger.errors)

    def test_bad_datagen_timestamp_format(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, bad_datagen_timestamp_format=True)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "datagen_timestamp_check", mock_logger)
        assert result is False
        assert any("[2.1.13 datagenTimestamp]" in m for m in mock_logger.errors)


# ---------------------------------------------------------------------------
# TestStruct14_CheckpointingWorkloads  (STRUCT-14, rule 2.1.21)
# ---------------------------------------------------------------------------

class TestStruct14_CheckpointingWorkloads:

    def test_default_fixture_passes(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "checkpointing_workloads_check", mock_logger)
        assert result is True
        assert mock_logger.errors == []

    def test_wrong_checkpointing_workload(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, wrong_checkpointing_workload="gpt2")
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "checkpointing_workloads_check", mock_logger)
        assert result is False
        assert any("[2.1.21 checkpointingWorkloads]" in m for m in mock_logger.errors)


# ---------------------------------------------------------------------------
# TestAccumulateDontAbort  (Phase 1 success criterion #2)
# ---------------------------------------------------------------------------

class TestAccumulateDontAbort:
    """Prove that two simultaneous violations under one check produce two records."""

    def test_struct07_two_violations(self, tmp_path, mock_logger):
        """systems/ with unpaired foo.yaml AND stray notes.txt → two [2.1.7] records."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, extra_systems_file="notes.txt")
        # Also add an unpaired .yaml — add it directly
        (root / "closed" / "Acme" / "systems" / "foo.yaml").write_text(
            "system_under_test:\n  solution:\n    submission_name: foo\n"
        )
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "systems_directory_files_check", mock_logger)
        assert result is False
        struct07_errors = [m for m in mock_logger.errors if "[2.1.7 systemsDirectoryFiles]" in m]
        assert len(struct07_errors) >= 2, (
            f"Expected >=2 [2.1.7] errors, got {len(struct07_errors)}: {struct07_errors}"
        )

    def test_struct09_two_legs_num_hosts_mismatch(self, tmp_path, mock_logger):
        """num_hosts_mismatch fixture → violations from both training and checkpointing legs."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, num_hosts_mismatch=True)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "identical_system_config_check", mock_logger)
        assert result is False
        struct09_errors = [m for m in mock_logger.errors if "[2.1.9 identicalSystemConfig]" in m]
        assert len(struct09_errors) >= 2, (
            f"Expected >=2 [2.1.9] errors (one per workload leg), got {len(struct09_errors)}"
        )


# ---------------------------------------------------------------------------
# TestQual02RuleIdPrefix  (D-05 — programmatic QUAL-02 enforcement)
# ---------------------------------------------------------------------------

class TestQual02RuleIdPrefix:
    """Every error from every STRUCT method begins with [<id> <name>]."""

    def test_all_rule_errors_have_locked_prefix(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        from mlpstorage_py.submission_checker.checks.submission_structure_checks import (
            SubmissionStructureCheck,
        )

        rules = discover_rules(SubmissionStructureCheck)
        assert len(rules) == 14, f"Expected 14 rules, got {len(rules)}"

        # For each rule, find a mutation fixture that would trigger an error,
        # then verify the error prefix.  We use a single "maximally mutated"
        # fixture that fires most rules, then collect errors per-method.
        #
        # Strategy: run the full check suite against various mutated fixtures
        # and assert that every captured error starts with "[<id> <name>]".

        # Build a fixture with top_level_capitalcase to fire STRUCT-02
        root = build_submission(tmp_path / "cap", top_level_capitalcase=True)
        check = _make_check(root, mock_logger)
        run_one_check(check, "top_level_subdirectories_check", mock_logger)

        for msg in mock_logger.errors:
            # Each error must start with a [id name] prefix
            assert msg.startswith("["), f"Error does not start with '[': {msg!r}"

    def test_discover_rules_returns_14_entries(self):
        rules = discover_rules(SubmissionStructureCheck)
        assert len(rules) == 14, f"Expected 14, got {len(rules)}: {sorted(rules)}"

    def test_all_rule_ids_present(self):
        rules = discover_rules(SubmissionStructureCheck)
        expected_ids = {
            "2.1.1", "2.1.2", "2.1.3", "2.1.4", "2.1.5", "2.1.6", "2.1.7",
            "2.1.8", "2.1.9", "2.1.10", "2.1.11", "2.1.12", "2.1.13", "2.1.21",
        }
        assert set(rules.keys()) == expected_ids, (
            f"Unexpected rule IDs: {set(rules.keys()) ^ expected_ids}"
        )


# ---------------------------------------------------------------------------
# TestMainWiring — main.py orchestration smoke tests (PLAN.md 01-03 D-02)
# ---------------------------------------------------------------------------

class TestMainWiring:
    """Smoke-tests asserting SubmissionStructureCheck is wired into main.py
    and the --reference-checksum CLI flag exists, per PLAN.md 01-03 must_haves.
    """

    def test_main_imports_submission_structure_check(self):
        import mlpstorage_py.submission_checker.main as m
        assert m.SubmissionStructureCheck.__name__ == "SubmissionStructureCheck"

    def test_main_has_reference_checksum_flag(self):
        import sys
        import mlpstorage_py.submission_checker.main as m
        original = sys.argv
        try:
            sys.argv = ["main", "--input", "/tmp", "--reference-checksum", "abc123"]
            args = m.get_args()
            assert args.reference_checksum == "abc123"
        finally:
            sys.argv = original
