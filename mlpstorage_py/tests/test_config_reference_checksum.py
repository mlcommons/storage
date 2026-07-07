#!/usr/bin/env python3
"""
Tests for REFERENCE_CHECKSUMS, RUN_TIMESTAMP_COUNT, MD5_EXCLUDE_PREFIXES,
and MD5_EXCLUDE_FILENAMES constants in constants.py.

Covers D-09, D-13, D-22 from the phase context. The Config.get_reference_checksum
precedence tests (D-10, D-12) were removed in Phase 8 (D-88); per-image
REFERENCE_CHECKSUMS lookup (CHECK-05) is deferred to a follow-up cycle.

Run with:
    pytest mlpstorage_py/tests/test_config_reference_checksum.py -v
"""

from mlpstorage_py.submission_checker.constants import (
    REFERENCE_CHECKSUMS,
    RUN_TIMESTAMP_COUNT,
    MD5_EXCLUDE_PREFIXES,
    MD5_EXCLUDE_FILENAMES,
)


class TestConstantsImport:
    """Tests that the new constants exist with correct values (D-09, D-13, D-22)."""

    def test_reference_checksums_structure(self):
        """REFERENCE_CHECKSUMS must be a dict with v2.0, v3.0, default keys all None."""
        assert REFERENCE_CHECKSUMS == {"v2.0": None, "v3.0": None, "default": None}

    def test_run_timestamp_count_value(self):
        """RUN_TIMESTAMP_COUNT must equal 6 (1 warm-up + 5 measured per Rules.md 2.1.17)."""
        assert RUN_TIMESTAMP_COUNT == 6

    def test_md5_exclude_prefixes_membership(self):
        """MD5_EXCLUDE_PREFIXES must be a tuple containing all required directory prefixes (D-13)."""
        assert isinstance(MD5_EXCLUDE_PREFIXES, tuple)
        required = {
            ".git/",
            "__pycache__/",
            ".pytest_cache/",
            ".venv/",
            "node_modules/",
            "build/",
            "dist/",
            ".tox/",
            "test/",
            "tests/",
        }
        for prefix in required:
            assert prefix in MD5_EXCLUDE_PREFIXES, f"Missing prefix: {prefix}"

    def test_md5_exclude_prefixes_exact_membership(self):
        """MD5_EXCLUDE_PREFIXES must contain exactly the expected entries — no extras (D-13 locked set).

        Kept in sync with constants.MD5_EXCLUDE_PREFIXES: when adding/removing
        entries there, update the expected tuple here so the gate keeps catching
        unintended drops as well as unintended additions.
        """
        expected = (
            ".git/",
            ".idea/",          # JetBrains IDE workspace
            ".vscode/",        # VS Code workspace
            ".claude/",        # Claude CLI runtime / settings
            ".agent/",         # Agent runtime (per project .gitignore "Coding Agents")
            ".agents/",        # Same, alternate name
            ".roo/",           # Roo agent runtime
            ".planning/",      # GSD planning artifacts (project-local)
            ".gsd-tmp/",       # GSD code-fixer worktree (project-local)
            "__pycache__/",
            ".pytest_cache/",
            ".venv/",
            "node_modules/",
            "build/",
            "dist/",
            ".tox/",
            "test/",
            "tests/",
        )
        assert MD5_EXCLUDE_PREFIXES == expected

    def test_md5_exclude_filenames_membership(self):
        """MD5_EXCLUDE_FILENAMES must be a tuple containing all required filename patterns (D-13)."""
        assert isinstance(MD5_EXCLUDE_FILENAMES, tuple)
        required = {".code-hash.json", "*.pyc", "*.pyo", ".DS_Store", "Thumbs.db"}
        for pattern in required:
            assert pattern in MD5_EXCLUDE_FILENAMES, f"Missing pattern: {pattern}"


