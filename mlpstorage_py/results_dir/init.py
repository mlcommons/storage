"""
``mlpstorage init`` CLI dispatcher — Slice 2 Task 2.

Implements the four refusal paths plus the idempotency path that LAY-01,
D-09, and D-11 mandate. Errors are raised as ``ConfigurationError``
subclasses so the existing top-level handler in ``main.py`` (~line 371-376)
prints the message + ``suggestion`` uniformly. This module never calls
``print`` directly and never catches its own exceptions — propagation to
the top-level handler is the contract.

Behavioural matrix (CONTEXT.md "Locked Decisions" + RESEARCH.md Slice 2):

==========================  ===========================  =================
State                       Caller intent                Outcome
==========================  ===========================  =================
parent missing              ``init Acme /a/b/c``         ConfigurationError
target missing, parent ok   ``init Acme /a/b``           mkdir + sentinel
target empty, no sentinel   ``init Acme /a/empty``       sentinel written
target non-empty, no sentinel  ``init Acme /a/dirty``    NonEmptyDirError
sentinel exists, orgname match  ``init Acme /a/done``   idempotent (log + 0)
sentinel exists, orgname mismatch  ``init Other /a/done``  DoubleInitError
==========================  ===========================  =================

Anti-pattern guard (RESEARCH.md "Anti-Patterns to Avoid"): this dispatcher
must NEVER touch ``cluster_collector`` / ``collect_cluster_info`` /
``collect_local_system_info``. ``init`` is filesystem-local and must
complete fast; cluster collection is the responsibility of ``run`` only.

Refs: 01-canonical-layout-and-init / 01-02-PLAN.md Task 2; CONTEXT.md
"Locked Decisions" D-09 + D-11; RESEARCH.md Pitfalls 1, 3, 7;
PATTERNS.md row ``results_dir/init.py``.
"""

from __future__ import annotations

import logging
import os

from mlpstorage_py.config import EXIT_CODE
from mlpstorage_py.errors import ConfigurationError, ErrorCode
from mlpstorage_py.results_dir.errors import (
    DoubleInitError,
    NonEmptyDirError,
)
from mlpstorage_py.results_dir.sentinel import (
    MLPERF_RESULTS_FILENAME,
    read_sentinel,
    write_sentinel,
)

logger = logging.getLogger(__name__)


def run_init(args) -> EXIT_CODE:
    """Initialize a results-dir with the canonical ``mlperf-results.yaml``
    sentinel.

    Args:
        args: Namespace with ``args.orgname`` (str) and ``args.path`` (str).
            ``args.mode`` is expected to be ``"init"`` but is not asserted —
            the caller (``main._main_impl``) routes by mode already.

    Returns:
        ``EXIT_CODE.SUCCESS`` on happy path AND on idempotent re-init with a
        matching orgname.

    Raises:
        ConfigurationError: Parent directory does not exist (D-09).
        NonEmptyDirError: Target exists, has files, no sentinel (LAY-01).
        DoubleInitError: Target has a sentinel with a different orgname
            (D-11 mismatch refusal).
    """
    target: str = args.path
    orgname: str = args.orgname

    # ── 1. D-09 — parent must already exist ────────────────────────────────
    # ``os.path.abspath`` normalises whatever the user passed (relative path,
    # trailing slash, etc.) so ``os.path.dirname`` returns a sensible parent
    # rather than an empty string on bare basenames.
    parent = os.path.dirname(os.path.abspath(target))
    if not os.path.isdir(parent):
        raise ConfigurationError(
            f"Cannot initialize {target!r}: parent directory {parent!r} "
            f"does not exist.",
            suggestion=(
                f"Run `mkdir -p {parent}` first, then re-run "
                f"`mlpstorage init {orgname} {target}`."
            ),
            code=ErrorCode.CONFIG_INVALID_VALUE,
        )

    # ── 1b. WR-02 — target exists but is NOT a directory ──────────────────
    # If we don't catch this here, the path falls through to
    # ``os.makedirs(target, exist_ok=True)`` at the end, which raises a
    # raw ``FileExistsError`` (``exist_ok=True`` only suppresses the
    # error when the path is an existing **directory**). The top-level
    # ``main()`` handler only catches ``MLPStorageException`` subclasses,
    # so the user sees an uncaught traceback instead of the friendly
    # LAY-01 message. Raise a typed error early.
    if os.path.exists(target) and not os.path.isdir(target):
        raise NonEmptyDirError(
            f"results-dir {target!r} exists but is not a directory.",
            suggestion=(
                f"Choose a different path, or remove the existing file at "
                f"{target!r} before running `mlpstorage init`."
            ),
            code=ErrorCode.CONFIG_INVALID_VALUE,
        )

    sentinel_path = os.path.join(target, MLPERF_RESULTS_FILENAME)

    # ── 2. D-11 — sentinel present: idempotent on match, refuse on mismatch ─
    if os.path.isfile(sentinel_path):
        existing = read_sentinel(target)
        # Case-sensitive equality (RESEARCH.md Pitfall 7 — no .lower()).
        if existing.orgname == orgname:
            logger.info(
                f"results-dir {target!r} already initialized as "
                f"{existing.orgname!r}; nothing to do."
            )
            return EXIT_CODE.SUCCESS
        raise DoubleInitError(
            f"results-dir {target!r} is already initialized as "
            f"{existing.orgname!r}; refusing to re-init as {orgname!r}.",
            suggestion=(
                f"Choose a different path, or remove the existing sentinel "
                f"at {sentinel_path!r} if you really intend to re-pin "
                f"the orgname."
            ),
            code=ErrorCode.CONFIG_INVALID_VALUE,
        )

    # ── 3. LAY-01 — target exists, has files, no sentinel: refuse ──────────
    # ``os.scandir`` enumerates every entry, including hidden ones (``.git``,
    # ``.gitkeep``, ``.DS_Store``, etc.). LAY-01 treats ANY entry as
    # blocking, but we surface the first few names so the user does not have
    # to run ``ls -la`` to figure out what is in the way (WR-08).
    if os.path.isdir(target):
        with os.scandir(target) as it:
            names = sorted(e.name for e in it)
        if names:
            preview = ", ".join(names[:5])
            if len(names) > 5:
                preview = f"{preview}, ..."
            raise NonEmptyDirError(
                f"results-dir {target!r} is non-empty and not initialized "
                f"(found: {preview}).",
                suggestion=(
                    "Choose an empty path, or remove the existing contents "
                    "before running `mlpstorage init`."
                ),
                code=ErrorCode.CONFIG_INVALID_VALUE,
            )

    # ── 4. Happy path — mkdir (leaf only) + write sentinel ─────────────────
    # Parent existence was verified in step 1; ``exist_ok=True`` covers the
    # "target dir already exists and is empty" branch from step 3.
    os.makedirs(target, exist_ok=True)
    written = write_sentinel(target, orgname)
    logger.info(
        f"Initialized results-dir at {target!r} as orgname={orgname!r}; "
        f"sentinel: {written!r}"
    )
    return EXIT_CODE.SUCCESS
