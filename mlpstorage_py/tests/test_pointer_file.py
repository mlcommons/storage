#!/usr/bin/env python3
"""
Tests for the pointer-file and pool-dir-name helpers introduced in Plan 06-01.

Covers the following NEW symbols in
mlpstorage_py.submission_checker.tools.code_image:

    - _POINTER_FILENAME (module-level constant, ".mlps-code-image")
    - PointerMalformed  (exception, subclass of CodeImageError)
    - _write_pointer_atomic(run_leaf, full_hash, log) -> None
    - _read_pointer(run_leaf, log) -> tuple[str, str]
    - _pool_dir_name(full_hash) -> str

Design contracts locked here:
    - D-61: pointer file content is `md5-tree-v2:<32-hex>`; trailing newline optional
    - D-62: pool dir name is `code-<first-8-hex>`
    - D-65: pointer write is atomic via write-tmp + os.rename
    - RESEARCH Pitfall 3: reader rejects every malformed variant
    - RESEARCH Pitfall 4: tmp sibling is dot-prefixed

The RED-first protocol (D-52/D-59) requires that these tests be committed
BEFORE the production symbols exist. On this commit, `pytest` collection
of this module raises ImportError — that is the intended RED state.

Run with:
    pytest mlpstorage_py/tests/test_pointer_file.py -v
"""

import os
from pathlib import Path

import pytest

# NOTE: these imports FAIL at collection time until Task 2 (GREEN) lands.
# The RED-state failure mode is ImportError, not test failure — collection
# aborts before any test body runs. This is the intended RED state per the
# plan's <verify> block.
from mlpstorage_py.submission_checker.tools.code_image import (
    CodeImageError,
    PointerMalformed,
    _POINTER_FILENAME,
    _pool_dir_name,
    _read_pointer,
    _write_pointer_atomic,
)


# ---------------------------------------------------------------------------
# MockLogger — reused verbatim from test_capture_or_verify_code_image.py:33-64
# ---------------------------------------------------------------------------

class MockLogger:
    def __init__(self):
        self.statuses = []
        self.errors = []
        self.warnings = []
        self.infos = []
        self.debugs = []

    def status(self, msg, *args):
        self.statuses.append(msg % args if args else msg)

    def error(self, msg, *args):
        self.errors.append(msg % args if args else msg)

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    def info(self, msg, *args):
        self.infos.append(msg % args if args else msg)

    def debug(self, msg, *args):
        self.debugs.append(msg % args if args else msg)

    # Phase 1 verbose levels (unused here but kept for compatibility)
    def verbose(self, msg, *args): pass
    def verboser(self, msg, *args): pass
    def ridiculous(self, msg, *args): pass


@pytest.fixture
def log():
    return MockLogger()


# Deterministic 32-hex fixture (lowercase). Value is arbitrary but stable so
# assertions on `code-<hash8>` and pointer content are exact-string checks.
_FIXTURE_FULL_HASH = "a3f8e91b2c4d0f5e6d7c8b9a0e1f2d3c"


@pytest.fixture
def full_hash_fixture():
    return _FIXTURE_FULL_HASH


@pytest.fixture
def tmp_run_leaf(tmp_path):
    """A pre-existing run-leaf directory the pointer helpers can write into."""
    leaf = tmp_path / "run_leaf"
    leaf.mkdir()
    return leaf


# ---------------------------------------------------------------------------
# TestPointerWrite — locks SC#2, SC#5, and Pitfall 4 (dot-prefixed tmp).
# ---------------------------------------------------------------------------

class TestPointerWrite:

    def test_writes_literal_md5_tree_v2_prefix_and_full_hash(
        self, tmp_run_leaf, full_hash_fixture, log
    ):
        # SC#2: writer emits `md5-tree-v2:<32-hex>` verbatim.
        _write_pointer_atomic(tmp_run_leaf, full_hash_fixture, log)
        pointer_path = tmp_run_leaf / _POINTER_FILENAME
        assert pointer_path.is_file(), (
            f"pointer file was not created at {pointer_path}"
        )
        contents = pointer_path.read_text(encoding="utf-8")
        # Reader is permissive on trailing whitespace, but the writer produces
        # exactly the literal — no trailing newline, no leading whitespace.
        assert contents == f"md5-tree-v2:{full_hash_fixture}", (
            f"pointer content mismatch: {contents!r}"
        )

    def test_atomicity_no_partial_file_on_baseexception(
        self, tmp_run_leaf, full_hash_fixture, log, monkeypatch
    ):
        # SC#5: writer catches BaseException so KeyboardInterrupt still
        # triggers tmp cleanup. On BaseException:
        #   (a) the exception propagates
        #   (b) run_leaf contains NEITHER the pointer NOR the tmp sibling
        real_open = open

        def raising_open(path, *a, **kw):
            # Only sabotage the tmp write. Any other open() (e.g. read-back
            # in the same test) should still work.
            if str(path).endswith(f".tmp.{os.getpid()}"):
                raise KeyboardInterrupt("simulated ^C mid-write")
            return real_open(path, *a, **kw)

        monkeypatch.setattr("builtins.open", raising_open)

        with pytest.raises(KeyboardInterrupt):
            _write_pointer_atomic(tmp_run_leaf, full_hash_fixture, log)

        # Neither pointer nor tmp sibling should be visible.
        pointer_path = tmp_run_leaf / _POINTER_FILENAME
        assert not pointer_path.exists(), (
            f"pointer file leaked after BaseException: {pointer_path}"
        )
        tmp_leftovers = list(tmp_run_leaf.glob(f".{_POINTER_FILENAME}.tmp.*"))
        assert tmp_leftovers == [], (
            f"tmp sibling leaked after BaseException: {tmp_leftovers}"
        )

    def test_pre_existing_tmp_sibling_cleaned_up_before_write(
        self, tmp_run_leaf, full_hash_fixture, log
    ):
        # SC#2 (idempotency): if a stale tmp sibling from a prior crash sits
        # in the leaf, the writer removes it before writing.
        stale_tmp = tmp_run_leaf / f".{_POINTER_FILENAME}.tmp.{os.getpid()}"
        stale_tmp.write_text("stale garbage", encoding="utf-8")
        assert stale_tmp.exists()

        _write_pointer_atomic(tmp_run_leaf, full_hash_fixture, log)

        pointer_path = tmp_run_leaf / _POINTER_FILENAME
        assert pointer_path.is_file()
        assert not stale_tmp.exists(), (
            "stale tmp sibling still present after successful write"
        )
        # Extra safety: content is the fresh literal, not the stale garbage.
        assert pointer_path.read_text(encoding="utf-8") == (
            f"md5-tree-v2:{full_hash_fixture}"
        )

    def test_tmp_sibling_name_is_dot_prefixed(
        self, tmp_run_leaf, full_hash_fixture, log, monkeypatch
    ):
        # SC#2 / Pitfall 4: any intermediate tmp name MUST be dot-prefixed so
        # a downstream `Path.glob("code-*")` (Plan 06-02's pool scan) never
        # picks it up. Spy on open() to observe the tmp path mid-write.
        observed_tmp_names = []
        real_open = open

        def spy_open(path, *a, **kw):
            p = Path(path)
            if p.parent == tmp_run_leaf and p.name != _POINTER_FILENAME:
                observed_tmp_names.append(p.name)
            return real_open(path, *a, **kw)

        monkeypatch.setattr("builtins.open", spy_open)

        _write_pointer_atomic(tmp_run_leaf, full_hash_fixture, log)

        # (a) The intermediate name observed during write must be dot-prefixed.
        assert observed_tmp_names, (
            "spy_open never saw a tmp write — writer signature may have changed"
        )
        for name in observed_tmp_names:
            assert name.startswith("."), (
                f"tmp sibling name {name!r} is NOT dot-prefixed — Pitfall 4"
            )
            # It should also carry the .tmp.<pid> shape.
            assert ".tmp." in name, (
                f"tmp sibling name {name!r} lacks the .tmp.<pid> segment"
            )

        # (b) No tmp sibling remains after write.
        leftovers = list(tmp_run_leaf.glob(f".{_POINTER_FILENAME}.tmp.*"))
        assert leftovers == [], f"tmp sibling leaked post-write: {leftovers}"

    def test_no_trailing_newline_assertion(
        self, tmp_run_leaf, full_hash_fixture, log
    ):
        # SC#9: writer does not need a trailing newline; and the round-trip
        # read succeeds against the writer's raw output.
        _write_pointer_atomic(tmp_run_leaf, full_hash_fixture, log)
        pointer_path = tmp_run_leaf / _POINTER_FILENAME
        contents = pointer_path.read_text(encoding="utf-8")
        assert not contents.endswith("\n"), (
            "writer emitted a trailing newline; D-61 permits either but "
            "the plan spec locks the writer to no-trailing-newline"
        )
        # Round-trip via reader must still succeed.
        alg, full_hash = _read_pointer(tmp_run_leaf, log)
        assert (alg, full_hash) == ("md5-tree-v2", full_hash_fixture)

    def test_pointer_write_is_idempotent_across_repeated_calls(
        self, tmp_run_leaf, full_hash_fixture, log
    ):
        # Writing twice with the same hash yields the same content and leaves
        # no tmp siblings behind.
        _write_pointer_atomic(tmp_run_leaf, full_hash_fixture, log)
        first_contents = (tmp_run_leaf / _POINTER_FILENAME).read_text(
            encoding="utf-8"
        )
        _write_pointer_atomic(tmp_run_leaf, full_hash_fixture, log)
        second_contents = (tmp_run_leaf / _POINTER_FILENAME).read_text(
            encoding="utf-8"
        )
        assert first_contents == second_contents

        leftovers = list(tmp_run_leaf.glob(f".{_POINTER_FILENAME}.tmp.*"))
        assert leftovers == [], (
            f"tmp sibling leaked across repeated writes: {leftovers}"
        )


# ---------------------------------------------------------------------------
# TestPointerRead — locks SC#3, SC#6, SC#7, SC#8 and every malformed variant
# from RESEARCH Pitfall 3.
# ---------------------------------------------------------------------------

class TestPointerRead:

    def _write_raw(self, run_leaf: Path, content: str) -> Path:
        """Write `content` bytes directly into <run_leaf>/<_POINTER_FILENAME>."""
        pointer_path = run_leaf / _POINTER_FILENAME
        pointer_path.write_text(content, encoding="utf-8")
        return pointer_path

    def test_reads_writer_output_round_trip(
        self, tmp_run_leaf, full_hash_fixture, log
    ):
        # SC#3: round-trip via writer → reader returns the exact tuple.
        _write_pointer_atomic(tmp_run_leaf, full_hash_fixture, log)
        result = _read_pointer(tmp_run_leaf, log)
        assert result == ("md5-tree-v2", full_hash_fixture)

    def test_reads_with_trailing_whitespace_and_newline(
        self, tmp_run_leaf, full_hash_fixture, log
    ):
        # SC#8: `.strip()` cleans a trailing newline + trailing spaces, and
        # the reader still returns the tuple unchanged.
        self._write_raw(
            tmp_run_leaf, f"md5-tree-v2:{full_hash_fixture}\n  "
        )
        result = _read_pointer(tmp_run_leaf, log)
        assert result == ("md5-tree-v2", full_hash_fixture)

    def test_missing_pointer_file_raises_FileNotFoundError(
        self, tmp_run_leaf, log
    ):
        # Absent pointer file → Path.read_text() raises FileNotFoundError.
        # The reader does NOT wrap this in PointerMalformed — the caller
        # can distinguish "pool image not linked" from "pool image linked
        # but pointer corrupted".
        with pytest.raises(FileNotFoundError):
            _read_pointer(tmp_run_leaf, log)

    def test_malformed_empty_tail_raises_PointerMalformed(
        self, tmp_run_leaf, log
    ):
        # SC#6, Pitfall 3: `md5-tree-v2:` (empty tail) must raise
        # PointerMalformed, not silently proceed with hex_part="".
        pointer_path = self._write_raw(tmp_run_leaf, "md5-tree-v2:")
        with pytest.raises(PointerMalformed) as exc_info:
            _read_pointer(tmp_run_leaf, log)
        # SC#6: message names the offending pointer path.
        assert str(pointer_path) in str(exc_info.value)

    def test_malformed_short_hex_raises_PointerMalformed(
        self, tmp_run_leaf, log
    ):
        # SC#6: hex tail shorter than 32 chars → PointerMalformed.
        self._write_raw(tmp_run_leaf, "md5-tree-v2:abc")
        with pytest.raises(PointerMalformed):
            _read_pointer(tmp_run_leaf, log)

    def test_malformed_uppercase_hex_raises_PointerMalformed(
        self, tmp_run_leaf, log
    ):
        # SC#6: regex is `[0-9a-f]{32}` — uppercase A-F does NOT match.
        self._write_raw(tmp_run_leaf, "md5-tree-v2:" + ("A" * 32))
        with pytest.raises(PointerMalformed):
            _read_pointer(tmp_run_leaf, log)

    def test_malformed_no_colon_raises_PointerMalformed(
        self, tmp_run_leaf, log
    ):
        # SC#7: no `:` at all → PointerMalformed.
        self._write_raw(tmp_run_leaf, "md5tree-v2 abc")
        with pytest.raises(PointerMalformed):
            _read_pointer(tmp_run_leaf, log)

    def test_malformed_unknown_algorithm_raises_PointerMalformed(
        self, tmp_run_leaf, log
    ):
        # SC#7: prefix must be exactly `md5-tree-v2`; anything else raises.
        sha256_hex = "0" * 64
        self._write_raw(tmp_run_leaf, f"sha256:{sha256_hex}")
        with pytest.raises(PointerMalformed) as exc_info:
            _read_pointer(tmp_run_leaf, log)
        # Message should reference the offending algorithm.
        assert "sha256" in str(exc_info.value)

    def test_malformed_error_message_names_pointer_path(
        self, tmp_run_leaf, log
    ):
        # SC#6: every PointerMalformed carries the offending pointer path
        # in its message. Verified across multiple malformed variants.
        variants = [
            "md5-tree-v2:",                  # empty tail
            "md5-tree-v2:abc",               # short hex
            "md5-tree-v2:" + ("A" * 32),    # uppercase
            "no-colon-at-all",               # no colon
            f"sha256:{'0' * 64}",            # unknown alg
        ]
        pointer_path = tmp_run_leaf / _POINTER_FILENAME
        for content in variants:
            pointer_path.write_text(content, encoding="utf-8")
            with pytest.raises(PointerMalformed) as exc_info:
                _read_pointer(tmp_run_leaf, log)
            assert str(pointer_path) in str(exc_info.value), (
                f"variant {content!r} raised PointerMalformed but the message "
                f"did not name the pointer path: {exc_info.value!s}"
            )

    def test_PointerMalformed_is_a_CodeImageError(self):
        # SC#6: main.py's existing CodeImageError → EXIT_CODE.CODE_IMAGE_ERROR
        # mapping must catch PointerMalformed without a new handler.
        assert issubclass(PointerMalformed, CodeImageError), (
            "PointerMalformed must subclass CodeImageError so main.py's "
            "existing exit-code mapping surfaces it as CODE_IMAGE_ERROR"
        )


# ---------------------------------------------------------------------------
# TestPoolDirName — locks SC#4 (D-62 shape).
# ---------------------------------------------------------------------------

class TestPoolDirName:

    def test_returns_code_prefix_plus_first_8_hex(self):
        # SC#4: `code-<first-8-hex>` shape, verbatim.
        assert _pool_dir_name(
            "a3f8e91b2c4d0f5e6d7c8b9a0e1f2d3c"
        ) == "code-a3f8e91b"

    def test_full_hash_input_only_uses_first_8_chars(self):
        # SC#4: same 8-char prefix → same suffix, regardless of the tail.
        assert _pool_dir_name("a3f8e91b" + "0" * 24) == "code-a3f8e91b"
        assert _pool_dir_name("a3f8e91b" + "f" * 24) == "code-a3f8e91b"
