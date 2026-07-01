"""CAP-03 FS-separation probe — direct kernel test for filesystem separation.

Rules 3.4.2 / 4.4.2 / 5.4.2 verify that the data/checkpoint directory and
the results directory live on DIFFERENT filesystems. The pre-#601 contract
relied on grepping a ``df`` block out of the run log, which has two
problems:

  1. The producer (mlpstorage CLI) never wrote a ``df`` block, so every
     conforming file-API submission hard-failed the rule (issue #601).
  2. Text-scraping ``df`` output is a brittle proxy: multi-line device-name
     wrapping, anchor regex fragility, and substring-match weakness in the
     mount-point column (see ``submission_checker/checks/helpers.py:87-90,
     179-185``).

CAP-03 replaces the proxy with the direct kernel test: ``os.link()`` returning
``EXDEV`` is the kernel's authoritative "different filesystem" signal — the
errno is literally defined as "the operation would have crossed filesystems."

This module is the producer-side primitive. The gate that calls it lives in
``Benchmark._pre_execution_gate`` (`base.py`); the validator that reads the
sidecar lives in ``submission_checker/checks/helpers.py:read_fs_separation_sidecar``.

Public surface:
    probe_fs_separation(path_a, path_b, run_uuid, logger=None) -> dict

Locked design decisions (issue #601, agreed before implementation):
  D-601-1. Hard gate — same_filesystem=True raises FileSystemError BEFORE
           the workload starts. Same posture as CAP-01 / CAP-02.
  D-601-2. Rank-0 only. No MPI gather. (Single-host today; extend later if
           heterogeneous-results-dir setups appear.)
  D-601-3. Pre-cutover validator fallback to df-block for one release.
           This module's contract is not affected by D-601-3.
  D-601-4. finally-block sentinel cleanup; unlink failures are
           logger.warning, not raise — mirrors CAP-02 D-44.
"""

from __future__ import annotations

import datetime
import errno
import os
import socket
from typing import Optional

from mlpstorage_py.errors import ErrorCode, FileSystemError


_SENTINEL_PREFIX = ".mlpstorage-fs-sep-probe-"


def _now_iso_utc() -> str:
    """Return the current UTC time in ISO-8601 with a Z suffix."""
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _safe_unlink(path: str, logger: Optional[object]) -> None:
    """Best-effort unlink — warn on failure, never raise (D-601-4 / CAP-02 D-44).

    A leftover sentinel is cosmetic; failing the gate over an unlink error
    would mask the actual gate verdict.
    """
    try:
        os.unlink(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        if logger is not None:
            logger.warning(
                "CAP-03: unlink of probe sentinel %s failed: %s "
                "(leftover sentinel is cosmetic)",
                path, exc,
            )


def probe_fs_separation(
    path_a: str,
    path_b: str,
    run_uuid: str,
    logger: Optional[object] = None,
) -> dict:
    """Probe whether ``path_a`` and ``path_b`` live on the same filesystem.

    Mechanism: create a sentinel in ``path_a`` and ``os.link()`` it into
    ``path_b``. Linux returns ``EXDEV`` ("Invalid cross-device link") iff
    the operation would have crossed filesystems — the kernel's
    authoritative answer to the question the validator is asking.

    Args:
        path_a: First path — by convention, the data/checkpoint directory.
        path_b: Second path — by convention, the results directory.
        run_uuid: Per-instance UUID (from ``Benchmark._run_uuid``) used to
            embed in the sentinel filename so concurrent runs against the
            same data directory cannot collide on the sentinel path
            (Pitfall 7).
        logger: Optional logger. Used only for unlink-failure warnings on
            the cleanup path (D-601-4). The happy path is silent (SC#6).

    Returns:
        Dict matching the locked sidecar shape::

            {
                "version": 1,
                "method": "link_exdev",
                "data_or_chkpt_path": str,
                "results_path": str,
                "data_or_chkpt_realpath": str,
                "results_realpath": str,
                "same_filesystem": bool,
                "probed_at": "<ISO-8601 UTC with Z>",
                "probed_by_rank": int,
                "probed_by_host": str,
            }

    Raises:
        FileSystemError (FS_PERMISSION_DENIED): ``os.link()`` raised an
            errno other than ``EXDEV`` — typically ``EACCES`` on a
            restricted parent, ``EROFS`` on a read-only mount, or
            ``ENOSPC`` on a full filesystem. The gate is a safety check;
            an inability to verify FS separation MUST NOT be silently
            treated as "verified safe" (CAP-01 precedent).
    """
    sentinel_name = f"{_SENTINEL_PREFIX}{run_uuid}"
    src = os.path.join(path_a, sentinel_name)
    dst = os.path.join(path_b, sentinel_name)

    same_filesystem: Optional[bool] = None

    try:
        # Create the source sentinel — open with O_EXCL so a leftover from
        # a crashed earlier run gets a hard failure rather than silent
        # reuse (which could trigger a stale-link surprise).
        try:
            fd = os.open(src, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
        except OSError as exc:
            raise FileSystemError(
                f"CAP-03: cannot create probe sentinel at {src}: {exc}",
                path=path_a,
                operation="cap03-probe-create",
                code=ErrorCode.FS_PERMISSION_DENIED,
            ) from exc

        try:
            os.link(src, dst)
            same_filesystem = True
        except OSError as exc:
            if exc.errno == errno.EXDEV:
                same_filesystem = False
            else:
                # EACCES / EROFS / ENOSPC / ENOENT / EPERM — cannot verify.
                raise FileSystemError(
                    (
                        f"CAP-03: cannot probe filesystem separation between "
                        f"{path_a} and {path_b}: {exc}"
                    ),
                    path=path_b,
                    operation="cap03-probe-link",
                    code=ErrorCode.FS_PERMISSION_DENIED,
                ) from exc
    finally:
        # Always attempt cleanup. unlink failures are logged, not raised
        # (D-601-4 mirrors CAP-02 D-44).
        _safe_unlink(src, logger)
        _safe_unlink(dst, logger)

    return {
        "version": 1,
        "method": "link_exdev",
        "data_or_chkpt_path": path_a,
        "results_path": path_b,
        "data_or_chkpt_realpath": os.path.realpath(path_a),
        "results_realpath": os.path.realpath(path_b),
        "same_filesystem": bool(same_filesystem),
        "probed_at": _now_iso_utc(),
        "probed_by_rank": 0,
        "probed_by_host": socket.gethostname(),
    }
