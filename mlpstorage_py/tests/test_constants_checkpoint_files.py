"""Tests for BUG-02: the checkpoint leaf's required files use the
``checkpointing_run`` prefix, not ``training_run``.

The list lives in the rules editions table
(``mlpstorage_py/rules/editions.yaml`` -> ``editions.<id>.checker.
checkpoint_required_files``) for every edition this tool can check; it used to
be ``constants.CHECKPOINT_REQUIRED_FILES`` keyed by spec version.

Additionally, the dot before ``log`` must be escaped (``\\.log`` not ``.log``)
to avoid latent over-matching.

References:
  - D-E2 in Phase 2 CONTEXT.md
  - Rules.md 2.1.25 checkpointingFiles
"""

import pytest

from mlpstorage_py.editions import load_editions


def _checkable_editions():
    table = load_editions()
    return [eid for eid, e in table.editions.items() if e.checkable]


def _files(edition):
    return load_editions().checker_parameters(edition).checkpoint_required_files


@pytest.mark.parametrize("edition", _checkable_editions())
class TestCheckpointRequiredFilesPrefix:
    """BUG-02: the filename prefix must be checkpointing_run, not training_run."""

    def test_checkpoint_required_files_uses_checkpointing_prefix(self, edition):
        files = _files(edition)
        assert any("checkpointing_run" in p for p in files), (
            f"edition {edition}: checkpoint_required_files has no 'checkpointing_run' entry. "
            f"BUG-02 not yet fixed."
        )

    def test_checkpoint_required_files_no_training_run(self, edition):
        files = _files(edition)
        bad = [p for p in files if p.startswith("training_run") or "/training_run" in p]
        assert not bad, (
            f"edition {edition}: checkpoint_required_files still contains training_run patterns: "
            f"{bad}. BUG-02 not yet fixed."
        )


@pytest.mark.parametrize("edition", _checkable_editions())
class TestCheckpointRequiredFilesEscaping:
    """The dot before 'log' must be escaped to avoid over-matching."""

    def test_checkpoint_required_files_escapes_log_extension_stdout(self, edition):
        assert r"checkpointing_run\.stdout\.log" in _files(edition)

    def test_checkpoint_required_files_escapes_log_extension_stderr(self, edition):
        assert r"checkpointing_run\.stderr\.log" in _files(edition)


@pytest.mark.parametrize("edition", _checkable_editions())
class TestCheckpointRequiredFilesOtherEntries:
    """The four other required entries must remain present and unchanged."""

    def test_checkpoint_required_files_preserves_other_entries(self, edition):
        files = _files(edition)
        for pattern in (r".*output\.json", r".*per_epoch_stats\.json", r".*summary\.json", r"dlio\.log"):
            assert pattern in files, (
                f"edition {edition}: checkpoint_required_files is missing expected entry {pattern!r}."
            )
