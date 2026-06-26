"""Stale-capture detection for the code-image module (#505 follow-up).

PR #512 changed `capture_code_image` to hash `source_root` directly instead of
the just-made `code_tmp/` copy, eliminating the walker-parity bug between the
copy callback and the hasher. The fix worked, but anyone with a `.code-hash.json`
left over from BEFORE #512 merged held a digest computed under the old
"hash the copy" semantics. The post-fix verify path computes a NEW digest
against `source_root` and compares to the stale stored value — they don't
match, and the user gets the misleading
``changes to the codebase are not allowed in a CLOSED run`` content-mismatch
error instead of an actionable "your capture is from an older version, delete
it and re-run" message.

The follow-up shipped here bumps `_ALGORITHM` from `md5-tree-v1` to
`md5-tree-v2` so `_read_hash_file`'s existing algorithm-equality check
(`code_image.py:461`) catches the stale capture and `capture_or_verify_code_image`'s
existing `MalformedHashFile` handler emits the actionable error. These tests
pin that contract end-to-end so future hash-semantic bumps stay safe.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlpstorage_py.submission_checker.tools.code_image import (
    _ALGORITHM,
    CodeImageError,
    MalformedHashFile,
    capture_code_image,
    verify_source_against_image,
    _read_hash_file,
)


def _make_log():
    log = MagicMock()
    log.warning = MagicMock()
    log.error = MagicMock()
    log.status = MagicMock()
    log.info = MagicMock()
    log.debug = MagicMock()
    return log


def _make_source(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "pyproject.toml").write_text("[project]\nname='x'\nversion='0'\n")
    (src / "mod.py").write_text("X = 1\n")
    return src


class TestCurrentAlgorithmIsV2:
    """The bumped constant must read v2 (not v1) after this PR."""

    def test_current_algorithm_is_v2(self):
        assert _ALGORITHM == "md5-tree-v2", (
            f"Algorithm bump for #505 follow-up missed: still {_ALGORITHM!r}"
        )

    def test_fresh_capture_stamps_v2(self, tmp_path):
        src = _make_source(tmp_path)
        target = tmp_path / "out"
        target.mkdir()
        img = capture_code_image(src, target, _make_log())
        assert img.algorithm == "md5-tree-v2"
        on_disk = json.loads((target / "code" / ".code-hash.json").read_text())
        assert on_disk["algorithm"] == "md5-tree-v2"


class TestStaleV1CaptureDetection:
    """v1 .code-hash.json files left over from before PR #512 must raise
    MalformedHashFile with the existing actionable handler in
    capture_or_verify_code_image, not silently fail verify with the
    misleading content-mismatch error."""

    def _v1_payload(self) -> dict:
        return {
            "hash": "a" * 32,
            "algorithm": "md5-tree-v1",
            "captured_at": "2026-06-24T00:00:00Z",
            "mlpstorage_version": "3.0.16",
            "git_sha": None,
        }

    def test_read_hash_file_rejects_v1(self, tmp_path):
        image_dir = tmp_path / "code"
        image_dir.mkdir()
        (image_dir / ".code-hash.json").write_text(json.dumps(self._v1_payload()))

        with pytest.raises(MalformedHashFile) as exc_info:
            _read_hash_file(image_dir, _make_log())

        msg = str(exc_info.value)
        # The error must name both algorithms so the user can map their
        # situation to "I have a pre-#512 capture".
        assert "md5-tree-v1" in msg
        assert "md5-tree-v2" in msg

    def test_verify_propagates_v1_rejection_as_malformedhashfile(self, tmp_path):
        """The runtime verify path inherits the MalformedHashFile from
        _read_hash_file → load_code_image, which capture_or_verify_code_image
        then catches and turns into the user-visible delete-code-dir message."""
        src = _make_source(tmp_path)
        image_dir = tmp_path / "code"
        image_dir.mkdir()
        (image_dir / ".code-hash.json").write_text(json.dumps(self._v1_payload()))

        with pytest.raises(MalformedHashFile):
            verify_source_against_image(src, image_dir, _make_log())


class TestRoundTripStillWorks:
    """The bump must not regress the happy path: capture-then-verify on an
    unchanged source must still match. Guards against accidentally breaking
    fresh captures while fixing stale-capture detection."""

    def test_capture_then_verify_matches(self, tmp_path):
        src = _make_source(tmp_path)
        target = tmp_path / "out"
        target.mkdir()
        capture_code_image(src, target, _make_log())
        matched = verify_source_against_image(src, target / "code", _make_log())
        assert matched is True

    def test_capture_is_deterministic_across_calls(self, tmp_path):
        """Two captures of the same unchanged tree must produce the same
        digest — pins the post-#512 source_root-hashing contract."""
        src = _make_source(tmp_path)
        digests = []
        for i in range(3):
            target = tmp_path / f"out{i}"
            target.mkdir()
            img = capture_code_image(src, target, _make_log())
            digests.append(img.hash)
        assert len(set(digests)) == 1, (
            f"Capture is non-deterministic: {digests}"
        )
