"""CAP-01 capacity gate — pre-execution disk-space check.

Implements the four-field-message contract from REQUIREMENTS.md CAP-01 +
05-03 Decision D-45:

    CAP-01: insufficient disk space at <destination_path>
      available_bytes: <int>
      required_bytes:  <int>
      deficit:         <int>

This module deliberately DOES NOT call
``mlpstorage_py.validation_helpers.check_disk_space`` — the existing helper
emits a GiB-rounded single-line summary (see DATAGEN_SPACE_INSUFFICIENT in
``mlpstorage_py/error_messages.py:221-226``), but CAP-01 mandates a
four-line, byte-exact, machine-parseable body (one labeled field per line).
See 05-RESEARCH.md §"CAP-01 wrap vs augment" for the architectural
rationale. We reuse the parent-walk + ``os.statvfs`` idioms inline rather
than calling the helper so the message format is locked at THIS call site.

Public surface (Slice 3):
    check_capacity_4field(destination_path, required_bytes, logger=None) -> None

Slice 4 (CAP-02 shared-filesystem verification) and Slice 5 (end-to-end
integration tests) build on this contract without modifying the public
signature.
"""

from __future__ import annotations

import os
from typing import Optional

from mlpstorage_py.errors import ErrorCode, FileSystemError


# Defense-in-depth read cap for any future file reads from this module.
# Not used today (we only call os.statvfs), but documented for parallel
# evolution with collect_chassis_model / collect_sysctl.
_MAX_READ_BYTES = 8192


def check_capacity_4field(
    destination_path: str,
    required_bytes: int,
    logger: Optional[object] = None,
) -> None:
    """Raise FileSystemError if free space at ``destination_path`` is below
    ``required_bytes``.

    The message body on the insufficient-space branch follows the locked
    four-field format from REQUIREMENTS.md CAP-01 + D-45:

        CAP-01: insufficient disk space at <destination_path>
          available_bytes: <int>
          required_bytes:  <int>
          deficit:         <int>

    Args:
        destination_path: Filesystem path the benchmark intends to write to.
            If the path does not yet exist, the gate walks up to the nearest
            existing parent and runs statvfs against THAT — matching the
            ``check_disk_space`` parent-walk pattern at
            ``validation_helpers.py:417-427`` but with the four-field
            message body emitted from this call site.
        required_bytes: Number of bytes the benchmark needs to write.
        logger: Optional logger. NOT used on the happy path — REQUIREMENTS.md
            SC#6 mandates silent success (no logger.info, .warning, .error
            calls when free space is sufficient). The parameter is accepted
            for API symmetry with the rest of the validation helpers.

    Returns:
        ``None`` on success. Silent — no log output, no return value.

    Raises:
        FileSystemError (code=FS_PATH_NOT_FOUND): no valid parent directory
            of ``destination_path`` exists (filesystem root is unreachable).
        FileSystemError (code=FS_PERMISSION_DENIED): ``os.statvfs`` raised
            ``OSError`` (typically EACCES on a restricted parent). The gate
            is a safety check; an inability to verify free space MUST NOT be
            silently treated as "verified safe".
        FileSystemError (code=FS_DISK_FULL): ``available_bytes <
            required_bytes``. Carries the four-field message body.
    """
    # Parent-walk to the nearest existing ancestor — mirrors
    # validation_helpers.py:417-427 but DOES NOT call into it (the helper's
    # eventual message format diverges per D-45).
    check_path = destination_path
    while not os.path.exists(check_path):
        parent = os.path.dirname(check_path)
        if parent == check_path:
            raise FileSystemError(
                f"CAP-01: cannot determine free space — no valid parent for {destination_path}",
                path=destination_path,
                operation="cap01-check",
                code=ErrorCode.FS_PATH_NOT_FOUND,
            )
        check_path = parent

    try:
        stat = os.statvfs(check_path)
        available_bytes = stat.f_bavail * stat.f_frsize
    except OSError as exc:
        raise FileSystemError(
            f"CAP-01: cannot statvfs {destination_path}: {exc}",
            path=destination_path,
            operation="statvfs",
            code=ErrorCode.FS_PERMISSION_DENIED,
        ) from exc

    if available_bytes < required_bytes:
        deficit = required_bytes - available_bytes
        raise FileSystemError(
            (
                f"CAP-01: insufficient disk space at {destination_path}\n"
                f"  available_bytes: {available_bytes}\n"
                f"  required_bytes:  {required_bytes}\n"
                f"  deficit:         {deficit}"
            ),
            path=destination_path,
            operation="cap01-check",
            code=ErrorCode.FS_DISK_FULL,
        )

    # Happy path: silent. REQUIREMENTS.md SC#6 — no logger output on success.
    return None
