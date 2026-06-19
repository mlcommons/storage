"""Tests for BUG-02: CHECKPOINT_REQUIRED_FILES uses wrong filename prefix.

``constants.py`` lines 44-48 reference ``training_run.stdout.log`` /
``training_run.stderr.log`` inside ``CHECKPOINT_REQUIRED_FILES`` — they should
be ``checkpointing_run.stdout.log`` / ``checkpointing_run.stderr.log``.

Additionally, the dot before ``log`` must be escaped (``\\.log`` not ``.log``)
to avoid latent over-matching.

References:
  - D-E2 in Phase 2 CONTEXT.md
  - RESEARCH.md §Codebase Investigation: constants.py (BUG-02 site)
  - Rules.md 2.1.25 checkpointingFiles
"""

import pytest

from mlpstorage_py.submission_checker.constants import CHECKPOINT_REQUIRED_FILES


_ALL_VERSIONS = ("v2.0", "v3.0", "default")


class TestCheckpointRequiredFilesPrefix:
    """BUG-02: the filename prefix must be checkpointing_run, not training_run."""

    def test_checkpoint_required_files_uses_checkpointing_prefix(self):
        """All three version keys must contain at least one checkpointing_run entry."""
        for version in _ALL_VERSIONS:
            files = CHECKPOINT_REQUIRED_FILES[version]
            assert any("checkpointing_run" in p for p in files), (
                f"CHECKPOINT_REQUIRED_FILES[{version!r}] has no 'checkpointing_run' entry. "
                f"BUG-02 not yet fixed."
            )

    def test_checkpoint_required_files_no_training_run(self):
        """No entry in CHECKPOINT_REQUIRED_FILES must start with training_run."""
        for version in _ALL_VERSIONS:
            files = CHECKPOINT_REQUIRED_FILES[version]
            bad = [p for p in files if p.startswith("training_run") or "/training_run" in p]
            assert not bad, (
                f"CHECKPOINT_REQUIRED_FILES[{version!r}] still contains training_run patterns: "
                f"{bad}. BUG-02 not yet fixed."
            )


class TestCheckpointRequiredFilesEscaping:
    """The dot before 'log' must be escaped to avoid over-matching."""

    def test_checkpoint_required_files_escapes_log_extension_stdout(self):
        """The stdout regex must escape the dot before 'log'."""
        for version in _ALL_VERSIONS:
            files = CHECKPOINT_REQUIRED_FILES[version]
            assert r"checkpointing_run\.stdout\.log" in files, (
                f"CHECKPOINT_REQUIRED_FILES[{version!r}] missing "
                r"r'checkpointing_run\.stdout\.log' (with escaped dot before log). "
                "BUG-02 not yet fixed."
            )

    def test_checkpoint_required_files_escapes_log_extension_stderr(self):
        """The stderr regex must escape the dot before 'log'."""
        for version in _ALL_VERSIONS:
            files = CHECKPOINT_REQUIRED_FILES[version]
            assert r"checkpointing_run\.stderr\.log" in files, (
                f"CHECKPOINT_REQUIRED_FILES[{version!r}] missing "
                r"r'checkpointing_run\.stderr\.log' (with escaped dot before log). "
                "BUG-02 not yet fixed."
            )


class TestCheckpointRequiredFilesOtherEntries:
    """The four other required entries must remain present and unchanged."""

    @pytest.mark.parametrize("version", _ALL_VERSIONS)
    def test_checkpoint_required_files_preserves_other_entries(self, version):
        """output.json, per_epoch_stats.json, summary.json, dlio.log must be present."""
        files = CHECKPOINT_REQUIRED_FILES[version]
        required_others = [
            r".*output\.json",
            r".*per_epoch_stats\.json",
            r".*summary\.json",
            r"dlio\.log",
        ]
        for pattern in required_others:
            assert pattern in files, (
                f"CHECKPOINT_REQUIRED_FILES[{version!r}] is missing expected entry "
                f"{pattern!r}. This entry should be unchanged by BUG-02 fix."
            )
