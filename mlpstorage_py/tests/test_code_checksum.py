#!/usr/bin/env python3
"""
Tests for mlpstorage_py.submission_checker.tools.code_checksum.compute_code_tree_md5
and the compute_code_checksum CLI tool.

Covers D-13, D-14 predicate behaviors and D-11 CLI tool.

Run with:
    pytest mlpstorage_py/tests/test_code_checksum.py -v
"""

import os
import sys
import subprocess
import pytest


# ---------------------------------------------------------------------------
# MockLogger that captures warning() and error() calls for assertion.
# Shape mirrors test_rules.py MockLogger but extended with call-capture lists.
# ---------------------------------------------------------------------------

class MockLogger:
    """Mock logger that captures warning/error messages for assertion."""

    def __init__(self):
        self.warnings = []
        self.errors = []

    def debug(self, msg, *args): pass
    def info(self, msg, *args): pass

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    def error(self, msg, *args):
        self.errors.append(msg % args if args else msg)

    def verbose(self, msg, *args): pass
    def verboser(self, msg, *args): pass
    def ridiculous(self, msg, *args): pass


@pytest.fixture
def mock_logger():
    """Return a fresh MockLogger for each test."""
    return MockLogger()


# ---------------------------------------------------------------------------
# Helper: write a file with exact binary content
# ---------------------------------------------------------------------------

def write_binary(path, content: bytes):
    """Write bytes to path, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


# ---------------------------------------------------------------------------
# Predicate tests (D-13, D-14)
# ---------------------------------------------------------------------------

class TestComputeCodeTreeMd5:
    """Unit tests for compute_code_tree_md5 covering all nine D-14 behaviors."""

    def test_git_dir_excluded(self, tmp_path, mock_logger):
        """Behavior 1: .git/ directory is excluded — two trees differing only
        by .git/ contents produce the same digest."""
        from mlpstorage_py.submission_checker.tools.code_checksum import compute_code_tree_md5

        # Tree A: has a real .py file plus a .git directory
        tree_a = tmp_path / "tree_a"
        write_binary(tree_a / "main.py", b"print('hello')\n")
        write_binary(tree_a / ".git" / "HEAD", b"ref: refs/heads/main\n")
        write_binary(tree_a / ".git" / "refs" / "heads" / "main", b"abc123\n")

        # Tree B: only the real .py file (no .git at all)
        tree_b = tmp_path / "tree_b"
        write_binary(tree_b / "main.py", b"print('hello')\n")

        digest_a = compute_code_tree_md5(str(tree_a), mock_logger)
        digest_b = compute_code_tree_md5(str(tree_b), mock_logger)

        assert digest_a is not None
        assert digest_b is not None
        assert digest_a == digest_b

    def test_pycache_dir_excluded(self, tmp_path, mock_logger):
        """Behavior 2: __pycache__/ directory is excluded."""
        from mlpstorage_py.submission_checker.tools.code_checksum import compute_code_tree_md5

        # Tree A: has a real file plus __pycache__
        tree_a = tmp_path / "tree_a"
        write_binary(tree_a / "pkg" / "mod.py", b"x = 1\n")
        write_binary(tree_a / "pkg" / "__pycache__" / "mod.cpython-310.pyc", b"\x00\x01\x02")

        # Tree B: only the real file
        tree_b = tmp_path / "tree_b"
        write_binary(tree_b / "pkg" / "mod.py", b"x = 1\n")

        digest_a = compute_code_tree_md5(str(tree_a), mock_logger)
        digest_b = compute_code_tree_md5(str(tree_b), mock_logger)

        assert digest_a == digest_b

    def test_binary_mode_no_line_ending_normalization(self, tmp_path, mock_logger):
        """Behavior 3: binary-mode read — CRLF and LF files with identical content
        produce the same digest; LF vs CRLF byte streams produce different digests."""
        from mlpstorage_py.submission_checker.tools.code_checksum import compute_code_tree_md5

        lf_content = b"line1\nline2\n"
        crlf_content = b"line1\r\nline2\r\n"

        # Two trees with identical LF content
        tree_a = tmp_path / "tree_a"
        write_binary(tree_a / "readme.txt", lf_content)

        tree_b = tmp_path / "tree_b"
        write_binary(tree_b / "readme.txt", lf_content)

        assert compute_code_tree_md5(str(tree_a), mock_logger) == compute_code_tree_md5(str(tree_b), mock_logger)

        # Tree with CRLF — must differ from LF tree
        tree_c = tmp_path / "tree_c"
        write_binary(tree_c / "readme.txt", crlf_content)

        assert compute_code_tree_md5(str(tree_a), mock_logger) != compute_code_tree_md5(str(tree_c), mock_logger)

    def test_posix_sorted_ordering_stable(self, tmp_path, mock_logger):
        """Behavior 4: digest is stable across two calls; POSIX-byte-order sort means
        uppercase filenames (lower byte values) sort before lowercase."""
        from mlpstorage_py.submission_checker.tools.code_checksum import compute_code_tree_md5

        tree = tmp_path / "tree"
        write_binary(tree / "a.py", b"a = 1\n")
        write_binary(tree / "b.py", b"b = 2\n")
        write_binary(tree / "Z.py", b"Z = 26\n")

        digest1 = compute_code_tree_md5(str(tree), mock_logger)
        digest2 = compute_code_tree_md5(str(tree), mock_logger)

        # Digest is stable
        assert digest1 == digest2
        assert digest1 is not None

        # Rename Z.py to Y.py — content unchanged, path bytes changed → different hash
        os.rename(str(tree / "Z.py"), str(tree / "Y.py"))
        digest3 = compute_code_tree_md5(str(tree), mock_logger)

        assert digest3 != digest1

    @pytest.mark.skipif(sys.platform == "win32", reason="symlink semantics differ on Windows")
    def test_symlink_rejected_with_warning(self, tmp_path, mock_logger):
        """Behavior 5: symlink inside the tree emits a warning containing
        '[2.1.6 codeDirectoryContents]' and the symlink path, returns a valid
        digest, and skips the symlinked entry (digest equals tree-without-symlink)."""
        from mlpstorage_py.submission_checker.tools.code_checksum import compute_code_tree_md5

        tree = tmp_path / "tree"
        write_binary(tree / "pkg" / "real.py", b"real = True\n")

        # Create a symlink inside the tree
        symlink_path = tree / "pkg" / "link.py"
        os.symlink(str(tree / "pkg" / "real.py"), str(symlink_path))

        digest_with_symlink = compute_code_tree_md5(str(tree), mock_logger)

        # Must not crash — returns a valid hex digest
        assert digest_with_symlink is not None
        assert len(digest_with_symlink) == 32
        assert all(c in "0123456789abcdef" for c in digest_with_symlink)

        # Warning must contain the exact prefix
        assert any("[2.1.6 codeDirectoryContents]" in w for w in mock_logger.warnings), (
            f"Expected '[2.1.6 codeDirectoryContents]' in warnings, got: {mock_logger.warnings}"
        )

        # Warning must contain the full symlink path
        assert any(str(symlink_path) in w for w in mock_logger.warnings), (
            f"Expected symlink path in warnings, got: {mock_logger.warnings}"
        )

        # Digest equals a tree with only real.py (symlink skipped)
        tree_no_symlink = tmp_path / "tree_no_symlink"
        write_binary(tree_no_symlink / "pkg" / "real.py", b"real = True\n")
        digest_no_symlink = compute_code_tree_md5(str(tree_no_symlink), MockLogger())

        assert digest_with_symlink == digest_no_symlink

    def test_rename_detection(self, tmp_path, mock_logger):
        """Behavior 6: renaming a file (same content, different relative path)
        produces a different digest because relative-path bytes feed into the hash."""
        from mlpstorage_py.submission_checker.tools.code_checksum import compute_code_tree_md5

        # Tree A: a/b.py
        tree_a = tmp_path / "tree_a"
        write_binary(tree_a / "a" / "b.py", b"hello")

        # Tree B: a/c.py (same content, different name)
        tree_b = tmp_path / "tree_b"
        write_binary(tree_b / "a" / "c.py", b"hello")

        digest_a = compute_code_tree_md5(str(tree_a), mock_logger)
        digest_b = compute_code_tree_md5(str(tree_b), mock_logger)

        assert digest_a != digest_b

    def test_pyc_filename_excluded(self, tmp_path, mock_logger):
        """Behavior 7: *.pyc files are excluded via MD5_EXCLUDE_FILENAMES."""
        from mlpstorage_py.submission_checker.tools.code_checksum import compute_code_tree_md5

        tree_a = tmp_path / "tree_a"
        write_binary(tree_a / "pkg" / "__init__.py", b"# init\n")
        write_binary(tree_a / "pkg" / "__init__.pyc", b"\x00\x01\x02pyc_bytes")

        tree_b = tmp_path / "tree_b"
        write_binary(tree_b / "pkg" / "__init__.py", b"# init\n")

        assert compute_code_tree_md5(str(tree_a), mock_logger) == compute_code_tree_md5(str(tree_b), mock_logger)

    def test_egg_info_directory_excluded(self, tmp_path, mock_logger):
        """Behavior 8: directories ending in .egg-info are excluded by the predicate
        (not in MD5_EXCLUDE_PREFIXES constant — handled directly in code)."""
        from mlpstorage_py.submission_checker.tools.code_checksum import compute_code_tree_md5

        tree_a = tmp_path / "tree_a"
        write_binary(tree_a / "mylib.py", b"lib = 1\n")
        write_binary(tree_a / "mylib.egg-info" / "PKG-INFO", b"Name: mylib\n")

        tree_b = tmp_path / "tree_b"
        write_binary(tree_b / "mylib.py", b"lib = 1\n")

        assert compute_code_tree_md5(str(tree_a), mock_logger) == compute_code_tree_md5(str(tree_b), mock_logger)

    def test_nonexistent_root_returns_none(self, tmp_path, mock_logger):
        """Behavior 9: compute_code_tree_md5 returns None for a nonexistent path."""
        from mlpstorage_py.submission_checker.tools.code_checksum import compute_code_tree_md5

        result = compute_code_tree_md5("/no/such/path/that/does/not/exist", mock_logger)

        assert result is None


# ---------------------------------------------------------------------------
# CLI integration tests (D-11) — added in Task 3
# ---------------------------------------------------------------------------

class TestComputeCodeChecksumCli:
    """Integration tests for the compute_code_checksum CLI tool (D-11)."""

    def test_cli_prints_hex_digest_and_exits_0(self, tmp_path):
        """Test 1: CLI against a real tree prints a 32-char hex digest and exits 0."""
        write_binary(tmp_path / "main.py", b"print('hello')\n")

        result = subprocess.run(
            [sys.executable, "-m", "mlpstorage_py.submission_checker.tools.compute_code_checksum", str(tmp_path)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Expected exit 0, got {result.returncode}. stderr: {result.stderr}"
        digest = result.stdout.strip()
        assert len(digest) == 32, f"Expected 32-char hex digest, got: {digest!r}"
        assert all(c in "0123456789abcdef" for c in digest), f"Not hex: {digest!r}"

    def test_cli_nonexistent_path_exits_1(self, tmp_path):
        """Test 2: CLI against a nonexistent path exits non-zero with an error message."""
        result = subprocess.run(
            [sys.executable, "-m", "mlpstorage_py.submission_checker.tools.compute_code_checksum", "/no/such/path"],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0, "Expected non-zero exit for nonexistent path"
        # Error must appear on stderr (logging output)
        assert len(result.stderr) > 0 or len(result.stdout) == 0

    def test_cli_deterministic_across_two_runs(self, tmp_path):
        """Test 3: same tree produces identical stdout on two consecutive CLI runs."""
        write_binary(tmp_path / "a.py", b"a = 1\n")
        write_binary(tmp_path / "b.py", b"b = 2\n")

        def run_cli():
            r = subprocess.run(
                [sys.executable, "-m", "mlpstorage_py.submission_checker.tools.compute_code_checksum", str(tmp_path)],
                capture_output=True,
                text=True,
            )
            return r.stdout.strip()

        assert run_cli() == run_cli()

    def test_cli_same_digest_with_excluded_paths_added(self, tmp_path):
        """Test 4: adding .git/HEAD and __pycache__/foo.pyc does not change the digest."""
        write_binary(tmp_path / "main.py", b"main = True\n")

        def run_cli():
            r = subprocess.run(
                [sys.executable, "-m", "mlpstorage_py.submission_checker.tools.compute_code_checksum", str(tmp_path)],
                capture_output=True,
                text=True,
            )
            return r.stdout.strip()

        digest_before = run_cli()

        # Now add excluded files
        write_binary(tmp_path / ".git" / "HEAD", b"ref: refs/heads/main\n")
        write_binary(tmp_path / "__pycache__" / "main.cpython-312.pyc", b"\x00\x01pyc_magic")

        digest_after = run_cli()

        assert digest_before == digest_after

    def test_cli_digest_matches_library_digest(self, tmp_path, mock_logger):
        """Test 5: CLI digest and library digest for the same tree are identical."""
        from mlpstorage_py.submission_checker.tools.code_checksum import compute_code_tree_md5

        write_binary(tmp_path / "pkg" / "mod.py", b"value = 42\n")

        cli_result = subprocess.run(
            [sys.executable, "-m", "mlpstorage_py.submission_checker.tools.compute_code_checksum", str(tmp_path)],
            capture_output=True,
            text=True,
        )
        cli_digest = cli_result.stdout.strip()
        lib_digest = compute_code_tree_md5(str(tmp_path), mock_logger)

        assert cli_digest == lib_digest
