"""Per-mode code-image capture (Rules.md §2.1.6 / LAY-06).

When a benchmark instantiates in ``closed`` or ``open`` mode, a copy of the
running ``mlpstorage_py/`` source tree is placed alongside the results so
the submission package can be audited reproducibly. ``whatif`` mode is a
dry-run and skips this step.

Mode policy
-----------

* **closed** — ONE image per (results_dir, orgname) pair at
  ``<results_dir>/closed/<orgname>/code/``. Idempotent: re-entry early-returns
  if the destination already exists.
* **open** — ONE image per (results_dir, orgname, benchmark, command) tuple at
  ``<results_dir>/open/<orgname>/code/<benchmark>/<command>/``. Open mode
  permits per-command code variation, so each command keeps its own snapshot.
  The single ``code/`` segment mirrors the closed-mode shape — the previous
  doubled ``code/.../code/`` was a typo (WR-05).
* **whatif** — return ``None``. No filesystem side effects.

Excludes
--------

Reader-side parity with ``submission_checker.tools.code_checksum`` (see
``constants.MD5_EXCLUDE_PREFIXES``/``MD5_EXCLUDE_FILENAMES``): the destination
must omit ``__pycache__/``, ``*.pyc``, ``tests/``, and ``.pytest_cache/`` so
the captured tree matches what the submission checker re-walks.

Security
--------

* Symlink traversal (T-1-CI2): ``shutil.copytree(symlinks=True)``. Symlinks
  in the source tree are NOT followed; ``copytree`` reproduces them as
  symlinks in the destination tree, preventing reads of arbitrary out-of-tree
  files. This is the V12 ASVS mitigation registered in the threat model.

  Note on stdlib semantics (counter-intuitive): in ``shutil.copytree``,
  ``symlinks=True`` means "symbolic links in the source tree result in
  symbolic links in the destination tree"; ``symlinks=False`` means "the
  contents of files pointed to by symbolic links are copied" — i.e. the
  link IS followed. The mitigation here is therefore ``symlinks=True``.
* Disk-fill DoS (T-1-CI1): accepted. The source path is the bounded
  ``mlpstorage_py/`` package directory; not user-controlled.

Refs: 01-canonical-layout-and-init / 01-05-PLAN.md Task 1; RESEARCH.md
"Per-mode code-image capture (LAY-06)"; threat model T-1-CI1 / T-1-CI2.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

import mlpstorage_py

# Mirrors ``submission_checker.constants.MD5_EXCLUDE_PREFIXES`` for the
# directory names + ``MD5_EXCLUDE_FILENAMES`` for the filenames. Kept inline
# (small, stable) so this module has no circular-import surface against the
# submission_checker package.
_EXCLUDE_DIRS = ("__pycache__", ".pytest_cache", "tests")
_EXCLUDE_FILENAMES = ("*.pyc",)


def _resolve_source_root(src_override: Optional[str]) -> Path:
    """Return the path to the running ``mlpstorage_py/`` package directory.

    The override is exposed exclusively for testability: production callers
    pass ``src_override=None`` and the live package path is used.
    """
    if src_override is not None:
        return Path(src_override)
    return Path(mlpstorage_py.__file__).parent


def _destination_for(
    results_dir: str,
    mode: str,
    orgname: str,
    benchmark_type: str,
    command: str,
) -> Path:
    """Compute the per-mode destination path.

    Raises ``ValueError`` for any mode other than ``closed`` or ``open``.
    """
    base = Path(results_dir)
    if mode == "closed":
        return base / "closed" / orgname / "code"
    if mode == "open":
        # WR-05: single ``code/`` segment — mirrors closed mode at line above.
        # Previous shape had a duplicated ``code/.../code/`` suffix.
        return base / "open" / orgname / "code" / benchmark_type / command
    raise ValueError(f"Unknown mode: {mode!r}")


def capture_code_image(
    results_dir: str,
    mode: str,
    orgname: str,
    benchmark_type: str,
    command: str,
    src_override: Optional[str] = None,
) -> Optional[str]:
    """Capture the live ``mlpstorage_py/`` source tree alongside results.

    Args:
        results_dir: Root results directory (the user's ``--results-dir``).
        mode: One of ``"closed"``, ``"open"``, ``"whatif"``. Unknown modes
            raise ``ValueError`` so misrouted calls fail loudly.
        orgname: Submitter organization name from the sentinel.
        benchmark_type: ``BENCHMARK_TYPE.name`` of the benchmark being run
            (e.g. ``"training"``, ``"checkpointing"``). Used only in
            ``open`` mode.
        command: The CLI command (e.g. ``"run"``, ``"datagen"``). Used only
            in ``open`` mode.
        src_override: Test-only hook to redirect the copy source. Production
            callers must pass ``None``.

    Returns:
        On ``closed``/``open``: the destination directory path as a string.
        On ``whatif``: ``None``.

    Raises:
        ValueError: ``mode`` is none of ``closed``/``open``/``whatif``.
        OSError / shutil.Error: bubbled up from ``shutil.copytree``. A failure
            here means ``Benchmark.__init__`` will not complete, which is the
            correct UX — we'd rather fail loudly than ship a half-populated
            results-dir.
    """
    if mode == "whatif":
        return None

    dst = _destination_for(results_dir, mode, orgname, benchmark_type, command)

    # Idempotency: if the destination already exists, this is a re-entry
    # (closed mode: same orgname; open mode: same benchmark+command). The
    # atomic-rename pattern below ensures that ``dst.exists()`` implies
    # "completed, trustworthy" — a torn copy from an SIGKILL'd previous
    # run lives under a temp sibling, not at ``dst``. See WR-01.
    if dst.exists():
        return str(dst)

    src = _resolve_source_root(src_override)

    # The destination's PARENT may not exist yet (e.g. ``.../open/Acme/code/training/run/``
    # is several levels deep on first use). ``copytree`` itself creates the
    # final segment; everything above it must exist.
    dst.parent.mkdir(parents=True, exist_ok=True)

    # WR-01: write-then-rename for crash-safe idempotency. Stage into a temp
    # sibling first; ``os.rename`` to the final ``dst`` only after
    # ``copytree`` returns success. ``os.rename`` is atomic on the same
    # filesystem, so a successful rename means the final ``dst`` is
    # guaranteed-complete. If the process dies during ``copytree``, the
    # partial tree lives under the temp sibling — the next run does NOT
    # see ``dst.exists()`` and re-copies cleanly. We also clean up the
    # temp sibling on copy failure so we don't leak disk.
    tmp = dst.parent / f".{dst.name}.tmp.{os.getpid()}"
    # If a previous run died after we created tmp but before rename, clean
    # it now so copytree (which requires its target NOT to exist) does not
    # error out on the stale sibling.
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    try:
        shutil.copytree(
            src,
            tmp,
            # T-1-CI2: preserve symlinks as symlinks in the destination — do
            # NOT follow them. ``symlinks=True`` in ``shutil.copytree`` means
            # "reproduce the link as a link"; ``symlinks=False`` would copy
            # the link's TARGET contents, which is exactly the V12 ASVS
            # threat we mitigate against (out-of-tree exfiltration).
            symlinks=True,
            ignore=shutil.ignore_patterns(*_EXCLUDE_DIRS, *_EXCLUDE_FILENAMES),
        )
    except BaseException:
        # On any failure (OSError, KeyboardInterrupt, shutil.Error, ...) try
        # to clean up the temp sibling so a follow-up run starts fresh.
        # Best-effort: if even cleanup fails, prefer to surface the original
        # exception, not the cleanup exception.
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    # Atomic on same filesystem — after this returns, ``dst`` is complete.
    os.rename(tmp, dst)
    return str(dst)


__all__ = ["capture_code_image"]
