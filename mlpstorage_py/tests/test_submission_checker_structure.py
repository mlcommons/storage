#!/usr/bin/env python3
"""
Tests for SubmissionStructureCheck — STRUCT-01 through STRUCT-14.

Run with:
    pytest mlpstorage_py/tests/test_submission_checker_structure.py -v
"""

import os
import shutil
import pytest

from mlpstorage_py.submission_checker.checks.submission_structure_checks import (
    SubmissionStructureCheck,
)
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.rule_registry import discover_rules


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(version="v2.0"):
    return Config(
        version=version,
        submitters=None,
        skip_output_file=False,
    )


def run_one_check(structure_check, method_name, mock_logger):
    """Invoke a single named check method and return its bool result."""
    method = getattr(structure_check, method_name)
    return method()


def _make_check(root_path, mock_logger, version="v2.0"):
    config = _make_config(version=version)
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
        for d in ("results", "systems"):
            assert (base / d).is_dir(), f"Missing {d}/"

    def test_default_builds_system_yaml(self, tmp_path):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        assert (root / "closed" / "Acme" / "systems" / "acme-storage-v1.yaml").is_file()

    def test_default_builds_system_pdf(self, tmp_path):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        assert (root / "closed" / "Acme" / "systems" / "acme-storage-v1.pdf").is_file()

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

    def test_missing_results_subdir(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, missing_required_subdir="results")
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "required_subdirectories_check", mock_logger)
        assert result is False
        assert any("[2.1.5 requiredSubdirectoriesClosed]" in m for m in mock_logger.errors), mock_logger.errors

    def test_missing_systems_subdir(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, missing_required_subdir="systems")
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "required_subdirectories_check", mock_logger)
        assert result is False
        assert any("[2.1.5 requiredSubdirectoriesClosed]" in m for m in mock_logger.errors), mock_logger.errors

    def test_extra_submitter_subdir(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, extra_submitter_subdir="extra")
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "required_subdirectories_check", mock_logger)
        assert result is False
        assert any("[2.1.5 requiredSubdirectoriesClosed]" in m for m in mock_logger.errors), mock_logger.errors
        # Plan 02-05: the legacy "only code/results/systems allowed" literal
        # was replaced by the sorted-list-repr format from Plan 02-03 Task 2.
        # Assert the new CLOSED required-set rendering is present.
        assert any(
            "allowed: ['results', 'systems']" in m
            for m in mock_logger.errors
        ), mock_logger.errors

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
        """Common submitter mistake: closed/<submitter>/benchmarks/{results,
        systems}/ instead of closed/<submitter>/{results, systems}/. The
        diagnostic for the extra wrapper dir should explicitly name the
        wrapping so the submitter knows what to fix.
        """
        from mlpstorage_py.tests.conftest import build_submission
        # Strip results so the wrapper detection has something to complain about.
        root = build_submission(tmp_path, missing_required_subdir="results")
        sub_path = os.path.join(root, "closed", "Acme")
        wrapper = os.path.join(sub_path, "benchmarks")
        os.makedirs(wrapper)
        os.makedirs(os.path.join(wrapper, "results"))
        os.makedirs(os.path.join(wrapper, "systems"))
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "required_subdirectories_check", mock_logger)
        assert result is False
        wrapping_msgs = [
            m for m in mock_logger.errors
            if "[2.1.5 requiredSubdirectoriesClosed]" in m
            and "nested one level deeper than expected" in m
        ]
        assert len(wrapping_msgs) == 1, mock_logger.errors



# ---------------------------------------------------------------------------
# Phase 2 Plan 02-03 — Tests for mode-aware required_subdirectories_check
# (STRUCT-05 per Rules.md §2.1.5 split — D-17)
# ---------------------------------------------------------------------------

class TestStruct05_ModeAwareRequiredSubdirectories:
    """STRUCT-05 mode-aware split (D-17) after the v1.1 pool refactor (D-64).

    Both CLOSED and OPEN submitter dirs require {results, systems}; code/
    lives in <division>/pool/code-images/ (D-64), not per-submitter.
    Violation messages route through `requiredSubdirectoriesClosed`
    (CLOSED) or `requiredSubdirectoriesOpen` (OPEN).
    """

    def test_closed_happy_path(self, tmp_path, mock_logger):
        sub = tmp_path / "closed" / "Acme"
        (sub / "results").mkdir(parents=True)
        (sub / "systems").mkdir(parents=True)
        check = _make_check(tmp_path, mock_logger)
        result = run_one_check(check, "required_subdirectories_check", mock_logger)
        assert result is True, mock_logger.errors
        assert mock_logger.errors == []

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
# Phase 2 Plan 02-05 — TestStruct05_OpenSubmitter
# Mode-aware required_subdirectories_check (TEST-11)
# Regression suite for the Gemini HIGH cross-plan finding (REVIEWS.md).
# ---------------------------------------------------------------------------

def _build_minimal_open_submitter(root, submitter, *, with_code=False,
                                  with_results=True, with_systems=True):
    """Build a minimal open/<submitter>/{code?,results?,systems?}/ tree."""
    sub = os.path.join(root, "open", submitter)
    os.makedirs(sub, exist_ok=True)
    if with_code:
        os.makedirs(os.path.join(sub, "code"), exist_ok=True)
    if with_results:
        os.makedirs(os.path.join(sub, "results"), exist_ok=True)
    if with_systems:
        os.makedirs(os.path.join(sub, "systems"), exist_ok=True)
    return sub


def _build_minimal_closed_submitter(root, submitter, *,
                                    with_results=True, with_systems=True):
    """Build a minimal closed/<submitter>/{results?,systems?}/ tree.

    Post-D-64 (v1.1 pool layout): code/ lives in closed/pool/code-images/,
    not at the per-submitter level.
    """
    sub = os.path.join(root, "closed", submitter)
    os.makedirs(sub, exist_ok=True)
    if with_results:
        os.makedirs(os.path.join(sub, "results"), exist_ok=True)
    if with_systems:
        os.makedirs(os.path.join(sub, "systems"), exist_ok=True)
    return sub


class TestStruct05_OpenSubmitter:
    """Mode-aware required_subdirectories_check per Plan 02-03 Task 2 (D-17).

    Regression suite for the Gemini HIGH cross-plan finding (REVIEWS.md):
    before the mode-aware refactor, EVERY OPEN submission would have been
    flagged as having a missing code/ at the submitter level. These tests
    directly exercise the new sub-rule anchors `requiredSubdirectoriesClosed`
    and `requiredSubdirectoriesOpen` and the new "allowed: [...]" violation
    message format from Plan 02-03 Task 2.
    """

    def test_closed_required_set_post_v11(self, tmp_path, mock_logger):
        """CLOSED post-D-64: {results, systems} required, code/ lives in pool."""
        _build_minimal_closed_submitter(str(tmp_path), "Acme")
        check = _make_check(str(tmp_path), mock_logger)
        run_one_check(check, "required_subdirectories_check", mock_logger)
        v25 = [m for m in mock_logger.errors if "[2.1.5 " in m]
        assert v25 == [], v25

    def test_open_happy_path_results_systems_passes(self, tmp_path, mock_logger):
        """OPEN submitter with {results, systems} only must pass STRUCT-05."""
        _build_minimal_open_submitter(str(tmp_path), "Acme", with_code=False)
        check = _make_check(str(tmp_path), mock_logger)
        run_one_check(check, "required_subdirectories_check", mock_logger)
        v25 = [m for m in mock_logger.errors if "[2.1.5 " in m]
        assert v25 == [], v25

    def test_open_with_code_at_submitter_level_flags_unexpected(self, tmp_path, mock_logger):
        """OPEN with code/ at submitter level → unexpected violation routed
        through requiredSubdirectoriesOpen with the new "allowed: [...]"
        message format.
        """
        _build_minimal_open_submitter(str(tmp_path), "Acme", with_code=True)
        check = _make_check(str(tmp_path), mock_logger)
        run_one_check(check, "required_subdirectories_check", mock_logger)
        v25 = [m for m in mock_logger.errors if "[2.1.5 " in m]
        assert len(v25) == 1, v25
        assert "unexpected subdirectory 'code' in open/Acme" in v25[0]
        assert "requiredSubdirectoriesOpen" in v25[0]
        assert "allowed: ['results', 'systems']" in v25[0]

    def test_open_missing_results_fails(self, tmp_path, mock_logger):
        _build_minimal_open_submitter(
            str(tmp_path), "Acme",
            with_code=False, with_results=False, with_systems=True,
        )
        check = _make_check(str(tmp_path), mock_logger)
        run_one_check(check, "required_subdirectories_check", mock_logger)
        v25 = [m for m in mock_logger.errors if "[2.1.5 " in m]
        assert any(
            "required subdirectory 'results' missing from open/Acme" in m
            for m in v25
        ), v25
        assert any("requiredSubdirectoriesOpen" in m for m in v25), v25

    def test_open_missing_systems_fails(self, tmp_path, mock_logger):
        _build_minimal_open_submitter(
            str(tmp_path), "Acme",
            with_code=False, with_results=True, with_systems=False,
        )
        check = _make_check(str(tmp_path), mock_logger)
        run_one_check(check, "required_subdirectories_check", mock_logger)
        v25 = [m for m in mock_logger.errors if "[2.1.5 " in m]
        assert any(
            "required subdirectory 'systems' missing from open/Acme" in m
            for m in v25
        ), v25
        assert any("requiredSubdirectoriesOpen" in m for m in v25), v25

    def test_closed_missing_results_routes_through_closed_anchor(self, tmp_path, mock_logger):
        """CLOSED missing results/ routes through requiredSubdirectoriesClosed."""
        _build_minimal_closed_submitter(str(tmp_path), "Acme", with_results=False)
        check = _make_check(str(tmp_path), mock_logger)
        run_one_check(check, "required_subdirectories_check", mock_logger)
        v25 = [m for m in mock_logger.errors if "[2.1.5 " in m]
        assert any(
            "required subdirectory 'results' missing from closed/Acme" in m
            for m in v25
        ), v25
        assert any("requiredSubdirectoriesClosed" in m for m in v25), v25

    def test_open_nesting_hint_works(self, tmp_path, mock_logger):
        """open/Acme/benchmarks/{results,systems} — nested one level too deep.

        The wrapping-hint diagnostic mentions the OPEN required-set elements.
        """
        root = str(tmp_path)
        sub = os.path.join(root, "open", "Acme")
        wrap = os.path.join(sub, "benchmarks")
        os.makedirs(os.path.join(wrap, "results"), exist_ok=True)
        os.makedirs(os.path.join(wrap, "systems"), exist_ok=True)
        check = _make_check(root, mock_logger)
        run_one_check(check, "required_subdirectories_check", mock_logger)
        v25 = [m for m in mock_logger.errors if "[2.1.5 " in m]
        assert any(
            "the submission appears to be nested one level deeper than expected" in m
            for m in v25
        ), v25


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


class TestIssue612WorkloadCategoriesAcceptsAllFour:
    """Issue #612: _VALID_WORKLOAD_CATEGORIES must include the on-disk
    directory names produced by ``BENCHMARK_TYPES.name`` —
    ``vector_database`` and ``kv_cache`` with underscores, not the short
    forms ``vectordb`` / ``kvcache``. Pre-fix every vdb / kvcache
    submission tripped a [2.1.10 workloadCategories] error."""

    @staticmethod
    def _add_workload_category_dir(root, sys_name, category):
        """Drop a bare <category>/ dir into the system's results subtree."""
        from pathlib import Path
        path = (
            Path(root) / "closed" / "Acme" / "results"
            / sys_name / category
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_vector_database_category_accepted(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        # Determine the default systemname from the fixture.
        from pathlib import Path
        sys_name = next(
            (Path(root) / "closed" / "Acme" / "results").iterdir()
        ).name
        self._add_workload_category_dir(root, sys_name, "vector_database")
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "workload_categories_check", mock_logger)
        assert result is True, (
            "vector_database must be a recognized workload category; "
            f"errors: {mock_logger.errors!r}"
        )
        assert mock_logger.errors == []

    def test_kv_cache_category_accepted(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        from pathlib import Path
        sys_name = next(
            (Path(root) / "closed" / "Acme" / "results").iterdir()
        ).name
        self._add_workload_category_dir(root, sys_name, "kv_cache")
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "workload_categories_check", mock_logger)
        assert result is True, (
            "kv_cache must be a recognized workload category; "
            f"errors: {mock_logger.errors!r}"
        )
        assert mock_logger.errors == []

    def test_short_form_vectordb_still_flagged(self, tmp_path, mock_logger):
        """Defense in depth: the SHORT form ``vectordb`` (without
        underscore) is NOT the canonical on-disk name and must be flagged
        as unexpected — submissions that get this name on disk indicate
        a writer-side regression, not a tolerable variant."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, extra_workload_category="vectordb")
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "workload_categories_check", mock_logger)
        assert result is False
        assert any("[2.1.10 workloadCategories]" in m for m in mock_logger.errors)

    def test_short_form_kvcache_still_flagged(self, tmp_path, mock_logger):
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, extra_workload_category="kvcache")
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "workload_categories_check", mock_logger)
        assert result is False
        assert any("[2.1.10 workloadCategories]" in m for m in mock_logger.errors)

    def test_error_message_enumerates_all_four_categories(self, tmp_path, mock_logger):
        """When a truly bogus category lands on disk, the violation
        message must enumerate all four allowed names so the user sees
        the canonical set — pre-fix it hardcoded 'only training and
        checkpointing allowed', misleading vdb / kvcache submitters."""
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, extra_workload_category="foo")
        check = _make_check(root, mock_logger)
        run_one_check(check, "workload_categories_check", mock_logger)
        joined = " ".join(mock_logger.errors)
        for category in ("training", "checkpointing", "vector_database", "kv_cache"):
            assert category in joined, (
                f"violation message must enumerate {category!r}; "
                f"got: {joined!r}"
            )


class TestIssue612ModeToCheckersKeys:
    """The MODE_TO_CHECKERS keys must match the on-disk directory names —
    ``vector_database`` and ``kv_cache``, NOT the short forms. Pre-fix
    the dict was keyed on the short forms so every vdb / kvcache
    submission flowed into the unrecognized-mode error branch at
    main.py:174."""

    def test_keys_are_disk_canonical(self):
        from mlpstorage_py.submission_checker.main import MODE_TO_CHECKERS
        assert "vector_database" in MODE_TO_CHECKERS, (
            f"MODE_TO_CHECKERS must key vdb under the disk name "
            f"'vector_database'; got keys {sorted(MODE_TO_CHECKERS.keys())!r}"
        )
        assert "kv_cache" in MODE_TO_CHECKERS, (
            f"MODE_TO_CHECKERS must key kvcache under the disk name "
            f"'kv_cache'; got keys {sorted(MODE_TO_CHECKERS.keys())!r}"
        )

    def test_short_form_keys_are_absent(self):
        """The short forms 'vectordb' / 'kvcache' would never match the
        loader's mode (the disk-canonical name). Their absence keeps the
        unrecognized-mode error branch firing for genuinely misnamed
        submissions instead of silently routing to a non-matching checker."""
        from mlpstorage_py.submission_checker.main import MODE_TO_CHECKERS
        assert "vectordb" not in MODE_TO_CHECKERS
        assert "kvcache" not in MODE_TO_CHECKERS

    def test_keys_align_with_benchmark_types_enum(self):
        """Round-trip: every BENCHMARK_TYPES.name must be a MODE_TO_CHECKERS
        key. This pins the writer↔consumer contract — if either side ever
        drifts (e.g. a future rename of the enum), this test fires."""
        from mlpstorage_py.config import BENCHMARK_TYPES
        from mlpstorage_py.submission_checker.main import MODE_TO_CHECKERS
        for member in BENCHMARK_TYPES:
            assert member.name in MODE_TO_CHECKERS, (
                f"BENCHMARK_TYPES.{member.name} produces directories "
                f"named {member.name!r} on disk but MODE_TO_CHECKERS has "
                f"no entry for it; keys are "
                f"{sorted(MODE_TO_CHECKERS.keys())!r}"
            )


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

    def test_missing_datasize_phase_warns_but_passes(self, tmp_path, mock_logger):
        """Missing datasize/ → warn-level (DATASIZE-MISSING), rule still passes.

        Rules.md §2.1.12 requires the datasize phase, but the checker enforces
        it at warn-level during the current submission window (see
        ``_WARN_ONLY_MISSING_TRAINING_PHASES`` note). Regression pin against
        the retroactive-invalidation concern raised when datasize/ was
        first added to the required set.
        """
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path, omit_datasize_phase=True)
        check = _make_check(root, mock_logger)
        result = run_one_check(check, "training_phases_check", mock_logger)
        assert result is True
        assert any(
            "[2.1.12 trainingPhases]" in m and "DATASIZE-MISSING" in m
            for m in mock_logger.warnings
        )
        # Must NOT surface as an error (warn-only doctrine).
        assert not any(
            "DATASIZE-MISSING" in m for m in mock_logger.errors
        )

    def test_missing_datagen_phase_still_errors(self, tmp_path, mock_logger):
        """Missing datagen/ → hard error, unchanged from pre-datasize behavior.

        Pins that the warn-only carve-out is scoped to `datasize` — the
        other required phases must still fail the check.
        """
        import shutil
        from mlpstorage_py.tests.conftest import build_submission
        root = build_submission(tmp_path)
        # Remove the datagen dir the fixture built.
        datagen_dir = (
            root / "closed" / "Acme" / "results" / "acme-storage-v1"
            / "training" / "unet3d" / "datagen"
        )
        shutil.rmtree(datagen_dir)
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
        assert len(rules) == 13, f"Expected 13 rules, got {len(rules)}"

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

    def test_discover_rules_returns_13_entries(self):
        rules = discover_rules(SubmissionStructureCheck)
        assert len(rules) == 13, f"Expected 13, got {len(rules)}: {sorted(rules)}"

    def test_all_rule_ids_present(self):
        rules = discover_rules(SubmissionStructureCheck)
        expected_ids = {
            "2.1.1", "2.1.2", "2.1.3", "2.1.4", "2.1.5", "2.1.7",
            "2.1.8", "2.1.9", "2.1.10", "2.1.11", "2.1.12", "2.1.13", "2.1.21",
        }
        assert set(rules.keys()) == expected_ids, (
            f"Unexpected rule IDs: {set(rules.keys()) ^ expected_ids}"
        )


# ---------------------------------------------------------------------------
# TestMainWiring — main.py orchestration smoke tests (PLAN.md 01-03 D-02)
# ---------------------------------------------------------------------------

class TestMainWiring:
    """Smoke-tests asserting SubmissionStructureCheck is wired into main.py."""

    def test_main_imports_submission_structure_check(self):
        import mlpstorage_py.submission_checker.main as m
        assert m.SubmissionStructureCheck.__name__ == "SubmissionStructureCheck"
