#!/usr/bin/env python3
"""
Tests for REFERENCE_CHECKSUMS, RUN_TIMESTAMP_COUNT, MD5_EXCLUDE_PREFIXES,
MD5_EXCLUDE_FILENAMES constants in constants.py and Config.get_reference_checksum
in configuration.py.

Covers D-09, D-10, D-12, D-13, D-22 from the phase context.

Run with:
    pytest mlpstorage_py/tests/test_config_reference_checksum.py -v
"""

import pytest

from mlpstorage_py.submission_checker.constants import (
    REFERENCE_CHECKSUMS,
    RUN_TIMESTAMP_COUNT,
    MD5_EXCLUDE_PREFIXES,
    MD5_EXCLUDE_FILENAMES,
)
from mlpstorage_py.submission_checker.configuration.configuration import Config


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
        }
        for prefix in required:
            assert prefix in MD5_EXCLUDE_PREFIXES, f"Missing prefix: {prefix}"

    def test_md5_exclude_prefixes_exact_membership(self):
        """MD5_EXCLUDE_PREFIXES must contain exactly the expected entries — no extras (D-13 locked set)."""
        expected = (
            ".git/",
            "__pycache__/",
            ".pytest_cache/",
            ".venv/",
            "node_modules/",
            "build/",
            "dist/",
            ".tox/",
        )
        assert MD5_EXCLUDE_PREFIXES == expected

    def test_md5_exclude_filenames_membership(self):
        """MD5_EXCLUDE_FILENAMES must be a tuple containing all required filename patterns (D-13)."""
        assert isinstance(MD5_EXCLUDE_FILENAMES, tuple)
        required = {"*.pyc", "*.pyo", ".DS_Store", "Thumbs.db"}
        for pattern in required:
            assert pattern in MD5_EXCLUDE_FILENAMES, f"Missing pattern: {pattern}"


class TestConfigGetReferenceChecksum:
    """Tests for Config.get_reference_checksum precedence chain (D-10, D-12)."""

    def test_returns_none_when_version_has_no_pinned_checksum(self):
        """Config.get_reference_checksum() returns None when REFERENCE_CHECKSUMS[version] is None."""
        config = Config(version="v2.0", submitters=["Acme"])
        assert config.get_reference_checksum() is None

    def test_constructor_override_beats_version_dict(self):
        """reference_checksum_override passed to constructor is returned when cli_override is absent."""
        config = Config(version="v2.0", submitters=["Acme"], reference_checksum_override="abc123")
        assert config.get_reference_checksum() == "abc123"

    def test_cli_override_beats_constructor_override(self):
        """cli_override kwarg to get_reference_checksum beats the constructor override (D-10 top precedence)."""
        config = Config(version="v2.0", submitters=["Acme"], reference_checksum_override="abc123")
        assert config.get_reference_checksum(cli_override="def456") == "def456"

    def test_backwards_compat_no_reference_checksum_override(self):
        """Config constructed without reference_checksum_override still works; skip_output_file still defaults False."""
        config = Config(version="v2.0", submitters=["Acme"])
        assert config.skip_output_file is False
        assert config.get_reference_checksum() is None

    def test_backwards_compat_with_skip_output_file(self):
        """Config(version, submitters, skip_output_file=True) still works after adding the new kwarg."""
        config = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
        assert config.skip_output_file is True
        assert config.get_reference_checksum() is None

    def test_unknown_version_returns_none_not_key_error(self):
        """An unrecognised version string returns None instead of raising KeyError (D-10 .get() usage)."""
        config = Config(version="v99.0", submitters=["Acme"])
        assert config.get_reference_checksum() is None

    def test_cli_override_none_falls_through_to_version_dict(self):
        """Explicitly passing cli_override=None falls through to constructor/version dict."""
        config = Config(version="v2.0", submitters=["Acme"], reference_checksum_override="ctor_val")
        assert config.get_reference_checksum(cli_override=None) == "ctor_val"
