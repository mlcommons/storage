#!/usr/bin/env python3
"""
Tests for mlpstorage_py.submission_checker.tools.code_image.capture_or_verify_code_image.

Covers Phase 2 D-07..D-10, D-20, D-21 and the consensus INLINE `.`/`..`
path-traversal guard (T-02-02-05 mitigation made inline).

Run with:
    pytest mlpstorage_py/tests/test_capture_or_verify_code_image.py -v
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlpstorage_py.errors import ConfigurationError, ErrorCode
from mlpstorage_py.submission_checker.tools.code_image import (
    CodeImageError,
    MissingHashFile,
    capture_or_verify_code_image,
    _SUBMITTER_NAME_RE,
    _RESERVED_PATH_SEGMENTS,
)


# ---------------------------------------------------------------------------
# MockLogger that captures status/error calls for assertion.
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


def _make_args(*, mode, command, results_dir, benchmark="training", model="unet3d"):
    return SimpleNamespace(
        mode=mode,
        command=command,
        results_dir=str(results_dir),
        benchmark=benchmark,
        model=model,
    )


# ---------------------------------------------------------------------------
# Module-level constant sanity
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_submitter_name_regex_compiled(self):
        assert _SUBMITTER_NAME_RE.match("acme_corp.v1-2") is not None
        assert _SUBMITTER_NAME_RE.match("bad name") is None
        assert _SUBMITTER_NAME_RE.match("path/with/slash") is None

    def test_reserved_path_segments(self):
        assert _RESERVED_PATH_SEGMENTS == frozenset({".", ".."})

    def test_regex_accepts_dot_and_dotdot(self):
        # The regex `^[A-Za-z0-9._-]+$` literally matches `.` and `..` —
        # this is exactly why the additional reserved-segments guard is needed.
        assert _SUBMITTER_NAME_RE.match(".") is not None
        assert _SUBMITTER_NAME_RE.match("..") is not None


# ---------------------------------------------------------------------------
# Gating contract (D-10) — no env reads, no fs ops for non-submission modes
# ---------------------------------------------------------------------------

class TestGatingContract:
    def test_whatif_returns_none(self, tmp_path, log):
        args = _make_args(mode="whatif", command="run", results_dir=tmp_path)
        assert capture_or_verify_code_image(args, {}, log) is None
        assert log.statuses == []
        assert log.errors == []

    @pytest.mark.parametrize("mode", [
        "reports", "validate", "history", "lockfile", "version", "rules-coverage",
    ])
    def test_non_submission_modes_return_none(self, tmp_path, log, mode):
        args = _make_args(mode=mode, command="run", results_dir=tmp_path)
        assert capture_or_verify_code_image(args, {}, log) is None

    @pytest.mark.parametrize("command", [
        "configview", "validate", "datasize-something-else",
    ])
    def test_non_submission_commands_return_none(self, tmp_path, log, command):
        # mode is closed but command is not in {datasize, datagen, run} → no-op
        args = _make_args(mode="closed", command=command, results_dir=tmp_path)
        assert capture_or_verify_code_image(args, {}, log) is None


# ---------------------------------------------------------------------------
# Env-var fail-fast (D-04, D-05)
# ---------------------------------------------------------------------------

class TestEnvVarFailFast:
    def test_missing_orgname_raises_configuration_error(self, tmp_path, log):
        args = _make_args(mode="closed", command="datagen", results_dir=tmp_path)
        with pytest.raises(ConfigurationError) as exc_info:
            capture_or_verify_code_image(args, {}, log)
        assert "MLPSTORAGE_ORGNAME" in str(exc_info.value)
        assert exc_info.value.parameter == "MLPSTORAGE_ORGNAME"
        assert "mlpstorage init" in (exc_info.value.suggestion or "")

    def test_missing_systemname_raises_configuration_error(self, tmp_path, log):
        args = _make_args(mode="open", command="datagen", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "acme"}
        with pytest.raises(ConfigurationError) as exc_info:
            capture_or_verify_code_image(args, env, log)
        assert "MLPSTORAGE_SYSTEMNAME" in str(exc_info.value)
        assert exc_info.value.parameter == "MLPSTORAGE_SYSTEMNAME"

    def test_orgname_with_space_rejected(self, tmp_path, log):
        args = _make_args(mode="closed", command="run", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "bad name"}
        with pytest.raises(ConfigurationError) as exc_info:
            capture_or_verify_code_image(args, env, log)
        assert "Rules.md" in str(exc_info.value)

    def test_orgname_with_slash_rejected(self, tmp_path, log):
        args = _make_args(mode="closed", command="run", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "evil/path"}
        with pytest.raises(ConfigurationError):
            capture_or_verify_code_image(args, env, log)


# ---------------------------------------------------------------------------
# INLINE path-traversal guard (CONSENSUS FINDING — T-02-02-05)
# ---------------------------------------------------------------------------

class TestPathTraversalGuard:
    def test_orgname_dot_rejected(self, tmp_path, log):
        args = _make_args(mode="closed", command="run", results_dir=tmp_path)
        with pytest.raises(ConfigurationError) as exc_info:
            capture_or_verify_code_image(args, {"MLPSTORAGE_ORGNAME": "."}, log)
        msg = str(exc_info.value)
        assert "'.' and '..' are reserved path segments" in msg

    def test_orgname_dotdot_rejected(self, tmp_path, log):
        args = _make_args(mode="closed", command="run", results_dir=tmp_path)
        with pytest.raises(ConfigurationError) as exc_info:
            capture_or_verify_code_image(args, {"MLPSTORAGE_ORGNAME": ".."}, log)
        assert "'.' and '..' are reserved path segments" in str(exc_info.value)

    def test_systemname_dot_rejected(self, tmp_path, log):
        args = _make_args(mode="open", command="run", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "acme", "MLPSTORAGE_SYSTEMNAME": "."}
        with pytest.raises(ConfigurationError) as exc_info:
            capture_or_verify_code_image(args, env, log)
        assert "'.' and '..' are reserved path segments" in str(exc_info.value)

    def test_systemname_dotdot_rejected(self, tmp_path, log):
        args = _make_args(mode="open", command="run", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "acme", "MLPSTORAGE_SYSTEMNAME": ".."}
        with pytest.raises(ConfigurationError) as exc_info:
            capture_or_verify_code_image(args, env, log)
        assert "'.' and '..' are reserved path segments" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Capture path (CAP-01, CAP-02, CAP-06)
# ---------------------------------------------------------------------------

class TestCapturePath:
    def test_closed_first_run_captures(self, tmp_path, log):
        args = _make_args(mode="closed", command="datagen", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "acme"}
        result = capture_or_verify_code_image(args, env, log)
        # CAP-02: CLOSED tree shape
        expected_code = tmp_path / "closed" / "acme" / "code"
        assert result == expected_code
        assert expected_code.is_dir()
        assert (expected_code / ".code-hash.json").is_file()
        # CAP-06: log message starts "Captured code image at "
        assert any(s.startswith(f"Captured code image at {expected_code}") for s in log.statuses), log.statuses

    def test_open_first_run_captures_per_leaf(self, tmp_path, log):
        args = _make_args(
            mode="open", command="run", results_dir=tmp_path,
            benchmark="training", model="unet3d",
        )
        env = {"MLPSTORAGE_ORGNAME": "acme", "MLPSTORAGE_SYSTEMNAME": "rig01"}
        result = capture_or_verify_code_image(args, env, log)
        expected_code = (
            tmp_path / "open" / "acme" / "results" / "rig01" / "training" / "unet3d" / "code"
        )
        assert result == expected_code
        assert expected_code.is_dir()

    def test_open_vectordb_uses_canonical_type_name(self, tmp_path, log):
        """The CLI subparser is named 'vectordb', but the on-disk type segment
        is 'vector_database' (BENCHMARK_TYPES.name). The helper must emit that
        canonical on-disk segment so the captured code/ lives in the same
        submission tree the runtime writes results into.

        vector_database splits results by <index_type> because AISAQ results
        are not comparable to DISKANN/HNSW. The captured code/ lives at
        vector_database/<index_type>/code/ — per-leaf, same depth as
        training/checkpointing. The index directory is the UPPERCASE token,
        matching args.index_type and summary.json.index_type.
        """
        args = SimpleNamespace(
            mode="open", command="run", results_dir=str(tmp_path),
            benchmark="vectordb", index_type="DISKANN",
        )
        env = {"MLPSTORAGE_ORGNAME": "acme", "MLPSTORAGE_SYSTEMNAME": "rig01"}
        result = capture_or_verify_code_image(args, env, log)
        expected_code = (
            tmp_path / "open" / "acme" / "results" / "rig01"
            / "vector_database" / "DISKANN" / "code"
        )
        assert result == expected_code
        # And the CLI name 'vectordb' must NOT appear as a path segment.
        assert "vectordb" not in {p.name for p in result.parents}

    def test_open_kvcache_uses_canonical_type_name(self, tmp_path, log):
        """Same contract as vectordb: CLI name 'kvcache' must map to canonical
        on-disk segment 'kv_cache' (BENCHMARK_TYPES.name).

        Like vector_database, kv_cache writes <type>/<command>/<datetime>/ —
        no <model> in the runtime path — so the captured code/ also lives
        directly under <type>/.
        """
        # kvcache does have --model in CLI, but the helper must ignore it
        # because the runtime path-shape has no model segment.
        args = _make_args(
            mode="open", command="run", results_dir=tmp_path,
            benchmark="kvcache", model="llama3-8b",
        )
        env = {"MLPSTORAGE_ORGNAME": "acme", "MLPSTORAGE_SYSTEMNAME": "rig01"}
        result = capture_or_verify_code_image(args, env, log)
        expected_code = (
            tmp_path / "open" / "acme" / "results" / "rig01"
            / "kv_cache" / "code"
        )
        assert result == expected_code
        assert "kvcache" not in {p.name for p in result.parents}
        # model segment must not appear in the captured path.
        assert "llama3-8b" not in {p.name for p in result.parents}


# ---------------------------------------------------------------------------
# Verify path (VALR-01/03 success; VALR-02/04 mismatch; D-21 missing-json)
# ---------------------------------------------------------------------------

class TestVerifyPath:
    def test_matching_code_image_verifies_silently(self, tmp_path, log, monkeypatch):
        # Use an isolated source tree to keep the live-source hash deterministic
        # (the real repo's untracked / non-copytree-able files would otherwise
        # diverge between capture-via-shutil and live-source hashing).
        src = tmp_path / "iso_src"
        src.mkdir()
        (src / "a.py").write_bytes(b"A\n")
        (src / "pyproject.toml").write_bytes(b"# stub\n")

        import mlpstorage_py.submission_checker.tools.code_image as mod
        monkeypatch.setattr(mod, "find_source_root", lambda: src)

        # First call captures.
        args = _make_args(mode="closed", command="datasize", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "acme"}
        code_dir = capture_or_verify_code_image(args, env, log)
        log.statuses.clear()
        # Second call should verify and pass.
        result = capture_or_verify_code_image(args, env, log)
        assert result == code_dir
        assert any(
            f"code unchanged from on-file image at {code_dir}" in s for s in log.statuses
        ), log.statuses

    def test_closed_mismatch_raises_codeimage_error_with_literal(self, tmp_path, log, monkeypatch):
        args = _make_args(mode="closed", command="run", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "acme"}
        capture_or_verify_code_image(args, env, log)

        # Force a hash mismatch by monkeypatching verify_source_against_image to return False.
        import mlpstorage_py.submission_checker.tools.code_image as mod
        monkeypatch.setattr(mod, "verify_source_against_image", lambda *a, **k: False)

        log.errors.clear()
        with pytest.raises(CodeImageError) as exc_info:
            capture_or_verify_code_image(args, env, log)
        assert "changes to the codebase are not allowed in a CLOSED run" in str(exc_info.value)
        assert any(
            "changes to the codebase are not allowed in a CLOSED run" in e for e in log.errors
        ), log.errors

    def test_open_mismatch_raises_codeimage_error_with_literal(self, tmp_path, log, monkeypatch):
        args = _make_args(
            mode="open", command="run", results_dir=tmp_path,
            benchmark="training", model="unet3d",
        )
        env = {"MLPSTORAGE_ORGNAME": "acme", "MLPSTORAGE_SYSTEMNAME": "rig01"}
        capture_or_verify_code_image(args, env, log)

        import mlpstorage_py.submission_checker.tools.code_image as mod
        monkeypatch.setattr(mod, "verify_source_against_image", lambda *a, **k: False)

        log.errors.clear()
        with pytest.raises(CodeImageError) as exc_info:
            capture_or_verify_code_image(args, env, log)
        assert "all runs of this type must use the same codebase" in str(exc_info.value)

    def test_missing_hash_file_logs_recovery_and_reraises(self, tmp_path, log):
        # Pre-create a code/ directory without .code-hash.json
        code_dir = tmp_path / "closed" / "acme" / "code"
        code_dir.mkdir(parents=True)
        (code_dir / "dummy.py").write_text("# placeholder")

        args = _make_args(mode="closed", command="run", results_dir=tmp_path)
        env = {"MLPSTORAGE_ORGNAME": "acme"}
        with pytest.raises(MissingHashFile):
            capture_or_verify_code_image(args, env, log)
        # D-21 actionable recovery substring
        assert any(
            "either delete `code/` and re-run to re-capture" in e for e in log.errors
        ), log.errors
