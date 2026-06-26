"""Unit tests for Phase 5 exception classes — SystemDriftError + SystemDescriptionParseError.

These two exception classes land as siblings of `FileSystemError` under
`MLPStorageException` per CONTEXT.md D-42 (SystemDriftError) and D-48
(SystemDescriptionParseError). They are imported by
`mlpstorage_py.system_description.auto_generator.write_systemname_yaml`'s
FileExistsError branch (Slice 2 wiring) to surface drift detected against
the on-disk systemname.yaml.

Test discipline:
- Each new class gets at least 5 unit tests covering: inheritance,
  default ErrorCode, ErrorCode enum-string value lock (E404 / E104),
  path attribute storage, default-suggestion text content.
- The default-suggestion tests use substring matching (NOT exact-text
  matching) so cosmetic wording tweaks don't force test churn while still
  locking the semantic intent.
- These tests have ZERO dependency on the on-disk YAML pipeline — they
  only construct the exceptions directly and inspect attributes.
"""

from __future__ import annotations

import pytest

from mlpstorage_py.errors import (
    ErrorCode,
    MLPStorageException,
    SystemDescriptionParseError,
    SystemDriftError,
)


# ---------------------------------------------------------------------------
# SystemDriftError (D-42)
# ---------------------------------------------------------------------------


class TestSystemDriftError:
    """SystemDriftError surfaces the LIFE-02/03 drift report to the operator."""

    def test_system_drift_error_inherits_mlpstorage_exception(self):
        """D-42: must inherit from MLPStorageException so main.py:262 top-level
        handler routes via the error-footer formatter and non-zero exit."""
        exc = SystemDriftError("drift detected")
        assert isinstance(exc, MLPStorageException)
        assert issubclass(SystemDriftError, MLPStorageException)

    def test_system_drift_error_default_code_is_fs_invalid_structure(self):
        """Default ErrorCode is FS_INVALID_STRUCTURE per 05-PATTERNS.md mapping
        ("the on-disk file describes a different fleet than reality")."""
        exc = SystemDriftError("drift detected")
        assert exc.code == ErrorCode.FS_INVALID_STRUCTURE

    def test_system_drift_error_default_code_is_E404(self):
        """Lock the enum string value so an accidental enum renumbering is caught."""
        exc = SystemDriftError("drift detected")
        assert exc.code.value == "E404"

    def test_system_drift_error_stores_path_attribute(self):
        """`self.path = path` per FileSystemError sibling shape (errors.py:295).
        The Slice 2 raise site passes `path=str(systemname_path)` so callers
        can inspect it without poking at the structured-error context dict."""
        exc = SystemDriftError("drift", path="/results/closed/Acme/systems/sys-v1.yaml")
        assert exc.path == "/results/closed/Acme/systems/sys-v1.yaml"

    def test_system_drift_error_default_suggestion_mentions_rename_or_remove(self):
        """Default suggestion guides the operator to rename-or-remove the on-disk
        yaml. Substring match on the verbs (`rename`/`remove` or similar) so
        wording polishes don't force test churn."""
        exc = SystemDriftError("drift")
        suggestion = exc.suggestion.lower()
        # Either "rename" or "remove" must appear; both verbs are project lingo
        # for the LIFE-02/03 remediation block already locked in diff.py's
        # format_unified_diff (the bullets are "Rename..." and "Remove...").
        assert "rename" in suggestion or "remove" in suggestion or "diff" in suggestion

    def test_system_drift_error_explicit_suggestion_overrides_default(self):
        """If the caller passes `suggestion=...`, it wins over the default
        (matches FileSystemError sibling pattern at errors.py:294)."""
        exc = SystemDriftError("drift", suggestion="bespoke help text")
        assert exc.suggestion == "bespoke help text"

    def test_system_drift_error_message_round_trips_through_str(self):
        """The drift report (a multi-line unified-diff body) survives str(exc)
        unchanged; the report IS the user-facing payload."""
        report = (
            "--- on-disk: /tmp/sys.yaml\n"
            "+++ in-memory: <computed from live MPI fleet>\n"
            "@@ clients[0].chassis.cpu_model @@\n"
            "- Old CPU\n"
            "+ New CPU\n"
        )
        exc = SystemDriftError(report, path="/tmp/sys.yaml")
        rendered = str(exc)
        assert "--- on-disk:" in rendered
        assert "@@ clients[0].chassis.cpu_model @@" in rendered

    def test_system_drift_error_path_optional(self):
        """`path` is optional (matches FileSystemError sibling shape — `path: str = None`)."""
        exc = SystemDriftError("drift no path")
        assert exc.path is None


# ---------------------------------------------------------------------------
# SystemDescriptionParseError (D-48)
# ---------------------------------------------------------------------------


class TestSystemDescriptionParseError:
    """SystemDescriptionParseError surfaces malformed on-disk YAML or missing
    required structural keys with an actionable rm-and-re-run remediation."""

    def test_system_description_parse_error_inherits_mlpstorage_exception(self):
        """D-48: must inherit from MLPStorageException so main.py:262 top-level
        handler routes via the error-footer formatter and non-zero exit."""
        exc = SystemDescriptionParseError("bad yaml")
        assert isinstance(exc, MLPStorageException)
        assert issubclass(SystemDescriptionParseError, MLPStorageException)

    def test_system_description_parse_error_default_code_is_config_parse_error(self):
        """Default ErrorCode is CONFIG_PARSE_ERROR per 05-PATTERNS.md mapping
        ("malformed config file at a known path")."""
        exc = SystemDescriptionParseError("bad yaml")
        assert exc.code == ErrorCode.CONFIG_PARSE_ERROR

    def test_system_description_parse_error_default_code_is_E104(self):
        """Lock the enum string value so an accidental enum renumbering is caught."""
        exc = SystemDescriptionParseError("bad yaml")
        assert exc.code.value == "E104"

    def test_system_description_parse_error_stores_path_attribute(self):
        """`self.path = path` per FileSystemError sibling shape."""
        exc = SystemDescriptionParseError("bad yaml", path="/results/closed/Acme/systems/sys-v1.yaml")
        assert exc.path == "/results/closed/Acme/systems/sys-v1.yaml"

    def test_system_description_parse_error_default_suggestion_mentions_rm_re_run(self):
        """Default suggestion guides the operator to rm-and-re-run. Substring
        match on the project's `rm` lingo (the file regenerates on next run)."""
        exc = SystemDescriptionParseError("bad yaml")
        suggestion = exc.suggestion.lower()
        # "rm" or "remove" or "re-run" — any of these confirms the rm-then-rerun intent.
        assert "rm " in suggestion or "remove" in suggestion or "re-run" in suggestion or "regenerat" in suggestion

    def test_system_description_parse_error_explicit_suggestion_overrides_default(self):
        """Caller-supplied suggestion wins over default (sibling-pattern parity)."""
        exc = SystemDescriptionParseError("bad yaml", suggestion="custom help")
        assert exc.suggestion == "custom help"

    def test_system_description_parse_error_path_optional(self):
        """`path` is optional (matches FileSystemError sibling shape — `path: str = None`)."""
        exc = SystemDescriptionParseError("bad yaml no path")
        assert exc.path is None

    def test_system_description_parse_error_message_includes_problem_mark_info(self):
        """The Slice-2 caller (parse_on_disk_systemname_yaml) builds the message
        with `(line N, column M)` when yaml.YAMLError surfaces a problem_mark.
        This test only verifies the message survives end-to-end as a string."""
        exc = SystemDescriptionParseError(
            "systemname.yaml at /tmp/sys.yaml (line 3, column 5) is malformed: ...",
            path="/tmp/sys.yaml",
        )
        rendered = str(exc)
        assert "(line 3, column 5)" in rendered


# ---------------------------------------------------------------------------
# Cross-class invariants (main.py top-level dispatch contract)
# ---------------------------------------------------------------------------


class TestPhase5ExceptionDispatchContract:
    """Both exception classes share the MLPStorageException base so main.py's
    existing top-level handler at mlpstorage_py/main.py:262 routes them via
    the error-footer formatter and a non-zero exit — no new dispatch needed."""

    def test_both_classes_inherit_mlpstorage_exception(self):
        """The single contract main.py:262 depends on."""
        assert issubclass(SystemDriftError, MLPStorageException)
        assert issubclass(SystemDescriptionParseError, MLPStorageException)

    def test_both_classes_raisable_and_catchable_as_mlpstorage_exception(self):
        """The actual try/except shape main.py uses."""
        with pytest.raises(MLPStorageException):
            raise SystemDriftError("x")
        with pytest.raises(MLPStorageException):
            raise SystemDescriptionParseError("x")

    def test_drift_error_is_not_a_parse_error(self):
        """They are SIBLINGS, not parent/child — drift means structurally-valid
        YAML that disagrees with reality; parse error means structurally-invalid YAML."""
        assert not issubclass(SystemDriftError, SystemDescriptionParseError)
        assert not issubclass(SystemDescriptionParseError, SystemDriftError)
