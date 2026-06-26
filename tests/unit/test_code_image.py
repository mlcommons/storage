"""Unit tests for ``mlpstorage_py.results_dir.code_image.capture_code_image``.

Covers LAY-06 (per-mode code-image capture, Rules.md §2.1.6):

- ``closed`` mode: ONE image at ``<rd>/closed/<orgname>/code/``; idempotent.
- ``open`` mode: per-(benchmark, command) image at
  ``<rd>/open/<orgname>/code/<benchmark>/<command>/``. Single ``code/``
  segment, mirroring closed mode (WR-05 — the duplicated suffix was a typo).
- ``whatif`` mode: returns ``None``; nothing written.
- Unknown mode: raises ``ValueError``.
- Excludes ``__pycache__/``, ``*.pyc``, ``tests/``, ``.pytest_cache/`` (V12).

Refs: 01-canonical-layout-and-init / 01-05-PLAN.md Task 1; RESEARCH.md "Per-mode
code-image capture (LAY-06)"; threat-model T-1-CI2 (symlinks=False).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest import mock

import pytest

from mlpstorage_py.results_dir.code_image import capture_code_image


def _make_fake_src(root: Path) -> Path:
    """Create a fake source tree with some excludable artifacts.

    Returns the path to the fake "package" root, which contains:
    - ``__init__.py``  (must be copied)
    - ``submod.py``    (must be copied)
    - ``__pycache__/cached.pyc``  (must be EXCLUDED, both as a `__pycache__`
      directory AND as a ``*.pyc`` filename)
    - ``stray.pyc``  (must be EXCLUDED on filename)
    - ``tests/test_x.py``  (must be EXCLUDED on dir name)
    - ``.pytest_cache/v/cache.txt``  (must be EXCLUDED on dir name)
    - ``inner/keep.py``  (must be copied)
    """
    pkg = root / "fake_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("# fake init\n")
    (pkg / "submod.py").write_text("VALUE = 1\n")
    (pkg / "__pycache__").mkdir()
    (pkg / "__pycache__" / "cached.pyc").write_bytes(b"\x00\x01")
    (pkg / "stray.pyc").write_bytes(b"\x00\x02")
    (pkg / "tests").mkdir()
    (pkg / "tests" / "test_x.py").write_text("def test_x(): pass\n")
    (pkg / ".pytest_cache").mkdir()
    (pkg / ".pytest_cache" / "v").mkdir()
    (pkg / ".pytest_cache" / "v" / "cache.txt").write_text("noop\n")
    (pkg / "inner").mkdir()
    (pkg / "inner" / "keep.py").write_text("KEEP = 1\n")
    return pkg


class TestClosedMode:
    """Closed mode: single ``<rd>/closed/<orgname>/code/`` image; idempotent."""

    def test_closed_captures_once(self, tmp_path):
        """`capture_code_image` writes the live source tree to
        ``<rd>/closed/<orgname>/code/`` on first call and returns the path."""
        dst = capture_code_image(
            str(tmp_path), "closed", "Acme", "training", "run",
        )
        expected = tmp_path / "closed" / "Acme" / "code"
        assert dst == str(expected)
        assert expected.is_dir()
        # Soft assertion: more than a few files were copied.
        all_files = list(expected.rglob("*"))
        assert len(all_files) > 10, (
            f"Expected the source tree image to contain >10 entries; got "
            f"{len(all_files)}"
        )
        # Exclude check on the real tree: no `__pycache__` in the destination.
        assert not any(
            p.name == "__pycache__" for p in expected.rglob("*")
        ), "Destination tree must not contain __pycache__"

    def test_closed_idempotent(self, tmp_path):
        """Second call with same args does NOT re-copy; ``copytree`` not invoked."""
        # First call: real copy.
        first = capture_code_image(
            str(tmp_path), "closed", "Acme", "training", "run",
        )
        assert Path(first).is_dir()

        # Second call: mock copytree and assert it was NOT called.
        with mock.patch("shutil.copytree") as fake_copytree:
            second = capture_code_image(
                str(tmp_path), "closed", "Acme", "training", "run",
            )
        assert second == first
        fake_copytree.assert_not_called()


class TestOpenMode:
    """Open mode: per-(benchmark, command) image."""

    def test_open_captures_per_command(self, tmp_path):
        """Two distinct commands produce two distinct subtrees."""
        run_dst = capture_code_image(
            str(tmp_path), "open", "Acme", "training", "run",
        )
        datagen_dst = capture_code_image(
            str(tmp_path), "open", "Acme", "training", "datagen",
        )
        # WR-05: single ``code/`` segment — was previously ``code/.../code/``.
        expected_run = tmp_path / "open" / "Acme" / "code" / "training" / "run"
        expected_datagen = tmp_path / "open" / "Acme" / "code" / "training" / "datagen"

        assert run_dst == str(expected_run)
        assert datagen_dst == str(expected_datagen)
        assert expected_run.is_dir()
        assert expected_datagen.is_dir()
        # Each tree exists independently.
        assert expected_run != expected_datagen

    def test_open_idempotent_per_tuple(self, tmp_path):
        """Repeated capture for the same (benchmark, command) does not re-copy."""
        first = capture_code_image(
            str(tmp_path), "open", "Acme", "training", "run",
        )
        with mock.patch("shutil.copytree") as fake_copytree:
            second = capture_code_image(
                str(tmp_path), "open", "Acme", "training", "run",
            )
        assert second == first
        fake_copytree.assert_not_called()


class TestWhatifMode:
    """Whatif mode: no code image; returns None; no filesystem side effects."""

    def test_whatif_skips(self, tmp_path):
        result = capture_code_image(
            str(tmp_path), "whatif", "Acme", "training", "run",
        )
        assert result is None
        # No files/dirs were created under tmp_path.
        assert list(tmp_path.iterdir()) == []


class TestErrorPaths:
    """Unknown modes raise ValueError; production never reaches this branch."""

    def test_unknown_mode_raises(self, tmp_path):
        with pytest.raises(ValueError, match="garbage"):
            capture_code_image(
                str(tmp_path), "garbage", "Acme", "training", "run",
            )


class TestExcludes:
    """Excludes apply: __pycache__/, *.pyc, tests/, .pytest_cache/."""

    def test_excludes_pycache_and_pyc_and_tests(self, tmp_path):
        """Use ``src_override`` so we control the source tree contents."""
        src_root = tmp_path / "src_root"
        src_root.mkdir()
        fake_pkg = _make_fake_src(src_root)

        # Results-dir is a separate tmp child.
        rd = tmp_path / "rd"
        rd.mkdir()

        dst = capture_code_image(
            str(rd), "closed", "Acme", "training", "run",
            src_override=str(fake_pkg),
        )
        dst_path = Path(dst)
        assert dst_path.is_dir()

        # Files that MUST be present.
        assert (dst_path / "__init__.py").is_file()
        assert (dst_path / "submod.py").is_file()
        assert (dst_path / "inner" / "keep.py").is_file()

        # Files / dirs that MUST be excluded.
        all_paths = list(dst_path.rglob("*"))
        names = {p.name for p in all_paths}
        assert "__pycache__" not in names, "must exclude __pycache__/"
        assert ".pytest_cache" not in names, "must exclude .pytest_cache/"
        assert "tests" not in names, "must exclude tests/"
        # No *.pyc anywhere.
        assert not any(p.suffix == ".pyc" for p in all_paths), (
            "must exclude *.pyc files"
        )


class TestAtomicWriteThenRename:
    """WR-01: capture uses write-then-rename so a partial tree from an
    SIGKILL/OOM'd previous run is never trusted as a completed image.

    The contract is: ``dst.exists()`` after ``capture_code_image`` returns
    success implies "complete and trustworthy". If ``copytree`` raises mid-
    copy, the partial tree must end up under a temp name (e.g. a sibling
    starting with ``.``), NOT at the final ``dst`` path.
    """

    def test_partial_copy_does_not_leave_dst_in_place(self, tmp_path):
        """If ``shutil.copytree`` fails partway, the final ``dst`` must not exist.

        Simulate a torn copy by patching ``shutil.copytree`` to create a
        partial tree at the path it was called with (mimicking what would
        be left behind by a real SIGKILL / OOM mid-copytree) and then
        raise. The implementation MUST stage into a temp location and only
        atomically rename to the final ``dst`` after success — so when the
        copy raises mid-way, the final ``dst`` is empty.

        A subsequent ``capture_code_image`` call must NOT see
        ``dst.exists()`` and early-return on a torn copy.
        """
        src_root = tmp_path / "src_root"
        src_root.mkdir()
        fake_pkg = _make_fake_src(src_root)
        rd = tmp_path / "rd"
        rd.mkdir()

        final_dst = rd / "closed" / "Acme" / "code"

        def fake_copytree(src, dst, **kwargs):
            # Mimic a real partial copytree: create dst, drop a partial file,
            # then explode. This is exactly what the kernel would leave
            # behind on SIGKILL.
            os.makedirs(dst, exist_ok=True)
            with open(os.path.join(dst, "partial_marker.txt"), "w") as f:
                f.write("partial")
            raise OSError("simulated mid-copy crash")

        with mock.patch(
            "mlpstorage_py.results_dir.code_image.shutil.copytree",
            side_effect=fake_copytree,
        ):
            with pytest.raises(OSError, match="simulated mid-copy crash"):
                capture_code_image(
                    str(rd), "closed", "Acme", "training", "run",
                    src_override=str(fake_pkg),
                )

        # The contract: after a failed copy, the FINAL dst path must NOT
        # exist. The partial tree may exist under a sibling temp name (we
        # don't pin which name here — the impl chooses), but the final
        # canonical path that future calls early-return on must not.
        assert not final_dst.exists(), (
            "WR-01: a failed mid-copy must NOT leave the final dst path in "
            "place; otherwise the next run would trust a torn copy via the "
            "idempotency early-return. The implementation should stage into "
            "a temp sibling and atomically rename only after success."
        )

    def test_successful_copy_lands_at_dst(self, tmp_path):
        """The atomic rename must land the staged tree at the final dst path.

        Belt-and-suspenders: WR-01 says torn copies must not leave dst in
        place. A successful copy MUST still leave dst in place — otherwise
        we've broken the happy path.
        """
        src_root = tmp_path / "src_root"
        src_root.mkdir()
        fake_pkg = _make_fake_src(src_root)
        rd = tmp_path / "rd"
        rd.mkdir()

        result = capture_code_image(
            str(rd), "closed", "Acme", "training", "run",
            src_override=str(fake_pkg),
        )

        expected_dst = rd / "closed" / "Acme" / "code"
        assert Path(result) == expected_dst
        assert expected_dst.is_dir()
        assert (expected_dst / "__init__.py").is_file()
        assert (expected_dst / "submod.py").is_file()


class TestDryRunSkipsCapture:
    """WR-09: ``--dry-run`` must skip ``capture_code_image`` entirely.

    Pre-fix, ``Benchmark.__init__`` invoked ``capture_code_image``
    unconditionally — so ``mlpstorage ... run --dry-run`` would write
    multi-MB code images to disk on every invocation even though no
    benchmark was executed. ``--dry-run`` is meant to be a pure
    inspection mode; it should leave the filesystem untouched.
    """

    def test_dry_run_skips_code_image(self, tmp_path, monkeypatch):
        """Constructing a benchmark with ``args.dry_run=True`` must NOT call
        ``capture_code_image`` and must set ``self.code_image_path = None``.
        """
        pytest.importorskip("mlpstorage_py.benchmarks.kvcache")
        from argparse import Namespace
        from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark

        # Track capture invocations.
        captured = {"called": False}

        def fake_capture(**kwargs):
            captured["called"] = True
            return str(tmp_path / "should-not-be-used")

        monkeypatch.setattr(
            "mlpstorage_py.results_dir.code_image.capture_code_image",
            fake_capture,
        )
        monkeypatch.setattr(
            "mlpstorage_py.benchmarks.base.Benchmark._reserve_run_directory",
            lambda self: str(tmp_path / "fake_run_dir"),
        )

        args = Namespace(
            mode="open",
            orgname="Acme",
            systemname="sys-v1",
            results_dir=str(tmp_path),
            data_dir=str(tmp_path),
            command="run",
            debug=False,
            dry_run=True,  # ← the WR-09 invariant
            model="llama3-8b",
            num_processes=1,
            stream_log_level="INFO",
            verbose=0,
            num_accelerators=1,
            accelerator_type="h100",
            allow_invalid_params=False,
            closed=False,
            open=True,
            mpi_bin="mpirun",
            mpi_extra_args="",
            exec_type=None,
            client_host_memory_in_gb=64,
            host_data_path=None,
            host_meta_path=None,
            inter_option_delay=0,
            num_trials=1,
            seed=42,
        )

        bench = None
        try:
            bench = KVCacheBenchmark(args=args, run_datetime="20260619_120000", run_number=0)
        except Exception:
            # Tolerate post-capture failures — only the capture-skip
            # invariant matters.
            pass

        assert captured["called"] is False, (
            "WR-09: capture_code_image MUST NOT be invoked on a --dry-run "
            "benchmark instantiation"
        )
        if bench is not None:
            assert getattr(bench, "code_image_path", "sentinel") is None, (
                "WR-09: --dry-run must leave self.code_image_path = None"
            )


class TestBenchmarkHook:
    """Benchmark.__init__ invokes capture_code_image after _reserve_run_directory."""

    def test_benchmark_init_calls_capture(self, tmp_path, monkeypatch):
        """Construct a benchmark with mocked _reserve_run_directory and verify
        ``capture_code_image`` is invoked AFTER it.

        We use a minimal subclass that sets BENCHMARK_TYPE so __init__ can run.
        The capture function itself is mocked so the test doesn't actually copy
        the live mlpstorage_py/ tree.
        """
        # Use kvcache benchmark which has fewer optional deps than training.
        pytest.importorskip("mlpstorage_py.benchmarks.kvcache")
        from argparse import Namespace
        from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark

        # Track call order: _reserve_run_directory MUST run before capture_code_image.
        call_log: list[str] = []

        # Patch capture in the module where it is imported (deferred import
        # inside Benchmark.__init__ — patch at the source).
        def fake_capture(**kwargs):
            call_log.append("capture")
            return str(tmp_path / "fake_code_dst")

        monkeypatch.setattr(
            "mlpstorage_py.results_dir.code_image.capture_code_image",
            fake_capture,
        )

        def fake_reserve(self):
            call_log.append("reserve")
            return str(tmp_path / "fake_run_dir")

        monkeypatch.setattr(
            "mlpstorage_py.benchmarks.base.Benchmark._reserve_run_directory",
            fake_reserve,
        )

        args = Namespace(
            mode="open",
            orgname="Acme",
            systemname="sys-v1",
            results_dir=str(tmp_path),
            data_dir=str(tmp_path),
            command="run",
            debug=False,
            model="llama3-8b",
            num_processes=1,
            stream_log_level="INFO",
            verbose=0,
            num_accelerators=1,
            accelerator_type="h100",
            allow_invalid_params=False,
            closed=False,
            open=True,
            mpi_bin="mpirun",
            mpi_extra_args="",
            exec_type=None,
            client_host_memory_in_gb=64,
            host_data_path=None,
            host_meta_path=None,
            inter_option_delay=0,
            num_trials=1,
            seed=42,
        )

        # Constructing the benchmark exercises __init__.
        try:
            KVCacheBenchmark(args=args, run_datetime="20260619_120000", run_number=0)
        except Exception:
            # Any post-capture failure is fine — we only assert ordering.
            pass

        assert "reserve" in call_log, "Benchmark.__init__ must call _reserve_run_directory"
        assert "capture" in call_log, "Benchmark.__init__ must call capture_code_image"
        # Capture must run AFTER reserve.
        assert call_log.index("reserve") < call_log.index("capture"), (
            "capture_code_image must run AFTER _reserve_run_directory"
        )


class TestSymlinkSafety:
    """T-1-CI2: capture uses ``symlinks=True`` so symlinks are preserved as
    symlinks (NOT followed) — out-of-tree targets cannot leak into the
    results-dir.

    Note on Python ``shutil.copytree`` semantics (counter-intuitive):

    * ``symlinks=True``  → symbolic links in the source tree are reproduced
                            as symbolic links in the destination tree (their
                            targets are NOT read/copied). This is the V12
                            mitigation we want.
    * ``symlinks=False`` → the **contents** of files pointed to by symbolic
                            links are copied. Out-of-tree targets get
                            materialized. This is what we want to AVOID.
    """

    def test_copytree_call_uses_symlinks_true(self, tmp_path):
        """Mock ``shutil.copytree`` and verify ``symlinks=True`` is passed.

        Note: post WR-01, ``capture_code_image`` stages into a temp sibling
        and atomically renames to ``dst``. The mock must therefore at least
        create the directory it was called with, otherwise the follow-up
        ``os.rename`` raises FileNotFoundError. We mimic real-copytree side
        effects minimally — just enough to let the rename succeed.
        """
        src_root = tmp_path / "src_root"
        src_root.mkdir()
        fake_pkg = _make_fake_src(src_root)
        rd = tmp_path / "rd"
        rd.mkdir()

        def fake_copytree_create_dst(src, dst, **kwargs):
            os.makedirs(dst, exist_ok=True)

        with mock.patch(
            "mlpstorage_py.results_dir.code_image.shutil.copytree",
            side_effect=fake_copytree_create_dst,
        ) as fake_copytree:
            capture_code_image(
                str(rd), "closed", "Acme", "training", "run",
                src_override=str(fake_pkg),
            )
        assert fake_copytree.called, "copytree must be invoked on first capture"
        # symlinks=True must be passed — preserves symlinks as symlinks so
        # out-of-tree targets are NOT followed.
        kwargs = fake_copytree.call_args.kwargs
        assert kwargs.get("symlinks") is True, (
            "shutil.copytree must be invoked with symlinks=True (T-1-CI2): "
            "preserve symlinks as symlinks so out-of-tree targets cannot leak."
        )
        assert kwargs.get("ignore") is not None, (
            "shutil.copytree must be invoked with an ignore predicate"
        )

    def test_outoftree_symlink_target_is_not_copied(self, tmp_path):
        """End-to-end: an in-tree symlink pointing at an out-of-tree file
        must NOT cause that file's contents to be materialized into the
        results-dir. The destination entry must be a symlink (broken or
        otherwise) — its target's bytes must not be read.

        This is the concrete T-1-CI2 invariant the threat model requires.
        """
        # Out-of-tree secret file — would-be exfiltration target.
        secret = tmp_path / "out_of_tree_secret.txt"
        secret.write_text("SECRET-CONTENTS\n")

        # In-tree source package with a symlink pointing at the secret.
        src_root = tmp_path / "src_root"
        src_root.mkdir()
        pkg = src_root / "fake_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("# fake init\n")
        link_path = pkg / "leaky_link.txt"
        os.symlink(secret, link_path)
        # Sanity: the symlink is set up so that following it yields the secret.
        assert os.path.islink(link_path)
        assert link_path.read_text() == "SECRET-CONTENTS\n"

        rd = tmp_path / "rd"
        rd.mkdir()

        dst = capture_code_image(
            str(rd), "closed", "Acme", "training", "run",
            src_override=str(pkg),
        )
        dst_link = Path(dst) / "leaky_link.txt"
        # The destination entry must exist AS A SYMLINK — not as a regular
        # file containing the secret's bytes.
        assert dst_link.is_symlink(), (
            "T-1-CI2: in-tree symlink must be preserved as a symlink in the "
            "destination, not materialized as a regular file with the "
            "out-of-tree target's contents."
        )
        # Belt-and-suspenders: even if the link still resolves to the secret
        # on this filesystem, the on-disk entry itself must not be a regular
        # file holding the secret bytes.
        assert not (dst_link.is_file() and not dst_link.is_symlink()), (
            "destination link must not be a plain file copy of the secret"
        )
