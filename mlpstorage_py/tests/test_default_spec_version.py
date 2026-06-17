"""Tests for DEFAULT_SPEC_VERSION derivation in submission_checker.constants.

The default value of ``--mlperf-version`` (standalone CLI) and
``--mlperf-version`` (the top-level ``mlpstorage validate`` shim) is derived
from the package's release version's major.minor so the two never drift.

References:
  - Rules.md 2.1.1 — the spec version controls every per-version dict lookup
  - mlpstorage_py/submission_checker/constants.py: _derive_default_spec_version
"""

import pytest

from mlpstorage_py.submission_checker.constants import (
    DEFAULT_SPEC_VERSION,
    VERSIONS,
    _derive_default_spec_version,
)


class TestDeriveDefaultSpecVersion:
    """The derivation maps package major.minor → 'v<major>.<minor>' and
    falls back to the most recent supported round when no direct match
    exists."""

    def test_exact_match_in_supported_list(self):
        assert _derive_default_spec_version("3.0.8", ["v2.0", "v3.0"]) == "v3.0"

    def test_picks_earlier_version_when_supported(self):
        assert _derive_default_spec_version("2.0.1", ["v2.0", "v3.0"]) == "v2.0"

    def test_falls_back_to_latest_when_unrecognized(self):
        # 4.0.0 isn't in VERSIONS yet; should fall back to v3.0 (most recent)
        assert _derive_default_spec_version("4.0.0", ["v2.0", "v3.0"]) == "v3.0"

    def test_falls_back_to_latest_when_metadata_unknown(self):
        # _resolve_version returns "unknown" if both PEP 621 metadata and
        # pyproject.toml are unavailable; derivation must still return a
        # usable value rather than crashing.
        assert _derive_default_spec_version("unknown", ["v2.0", "v3.0"]) == "v3.0"

    def test_handles_pre_release_suffix(self):
        # PEP 440 dev/rc suffixes ("3.0.8.dev0", "3.0.8rc1") should still
        # extract the major.minor cleanly.
        assert _derive_default_spec_version("3.0.8.dev0", ["v2.0", "v3.0"]) == "v3.0"
        assert _derive_default_spec_version("3.0.8rc1", ["v2.0", "v3.0"]) == "v3.0"

    def test_empty_supported_list_returns_unknown_marker(self):
        # Defensive: with no supported versions configured, return a string
        # rather than IndexError; callers will surface the mismatch via
        # per-version dict lookups.
        assert _derive_default_spec_version("3.0.8", []) == "unknown"


class TestDefaultSpecVersionModuleLevel:
    """The module-level DEFAULT_SPEC_VERSION constant must be a value that
    every per-version dict in constants.py has an entry for — otherwise the
    default invocation crashes with KeyError (the bug that motivated this
    module)."""

    def test_default_is_in_versions_list(self):
        assert DEFAULT_SPEC_VERSION in VERSIONS, (
            f"DEFAULT_SPEC_VERSION={DEFAULT_SPEC_VERSION!r} must be present "
            f"in VERSIONS={VERSIONS!r}; otherwise per-version dict lookups "
            f"raise KeyError under the default invocation"
        )

    @pytest.mark.parametrize(
        "dict_name",
        [
            "DATAGEN_REQUIRED_FILES",
            "DATAGEN_REQUIRED_FOLDERS",
            "RUN_REQUIRED_FILES",
            "RUN_REQUIRED_FOLDERS",
            "CHECKPOINT_REQUIRED_FILES",
            "CHECKPOINT_REQUIRED_FOLDERS",
            "SYSTEM_PATH",
        ],
    )
    def test_default_spec_version_resolves_in_every_per_version_dict(self, dict_name):
        """Every per-version dict consulted via Config / DirectoryCheck must
        have an entry for DEFAULT_SPEC_VERSION, otherwise the default
        invocation crashes (the v5.1 KeyError class of bug)."""
        from mlpstorage_py.submission_checker import constants
        d = getattr(constants, dict_name)
        assert DEFAULT_SPEC_VERSION in d, (
            f"{dict_name} is missing an entry for the derived default "
            f"{DEFAULT_SPEC_VERSION!r}; populate it before releasing this "
            f"package version"
        )
