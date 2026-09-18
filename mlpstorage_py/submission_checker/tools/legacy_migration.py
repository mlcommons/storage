"""One-shot legacy migration coordinator for Phase 7.

Performs automatic, idempotent, crash-resumable migration of v1.0-layout
``code/`` directories into the content-addressed pool, and (PR4 of the
results-dir hygiene effort) relocation of a per-organization pool at
``<results_dir>/<orgname>/code-*/`` into the tree-wide pool at
``<results_dir>/code-images/``. Both steps are driven by
``_check_and_migrate_legacy_layout`` before every capture.

Design decisions:
- D-70: Migration invoked by an explicit pre-check before capture_or_verify_code_image.
- D-71: Crash-safety via atomic primitives (no journal). Fixed pass-2 step order:
  (1) materialize pool images, (2) write pointer files, (3) delete legacy dirs,
  (4) write sentinel. Each step is idempotent by construction.
- D-72: Sentinel ``.mlps-image-pool`` is plain text two key=value lines.
  Atomic write via tmp + os.rename.
- D-73: Strict two-pass: pass 1 (verify ALL legacy dirs) before any pass-2 writes.
  Any hash mismatch raises HandEditedCodeImage aborting before writes.
- D-74: Exactly two status-level log call-sites — header and summary.
  Per-image detail at log.debug() only.

Public API:
    migrate_legacy_layout(results_dir, orgname, log) -> None
    migrate_org_pool(results_dir, orgname, log) -> int
    _check_and_migrate_legacy_layout(args, env, log) -> None
    _read_sentinel(sentinel_path, log) -> dict[str, str]
    VerifiedLegacyImage (dataclass)
"""

from __future__ import annotations

import errno
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from mlpstorage_py.submission_checker.tools.code_image import (
    CodeImageError,  # noqa: F401 — re-exported for test imports
    HandEditedCodeImage,
    LegacyLayoutDetected,  # noqa: F401 — retained as defensive guard per D-70
    MalformedHashFile,
    MissingHashFile,
    MLPSTORAGE_VERSION,
    _SUBMISSION_COMMANDS,
    _SUBMISSION_MODES,
    _capture_new_pool_image,
    _find_matching_pool_image,
    _now_utc_iso,
    _POOL_SENTINEL_FILENAME,
    _read_hash_file,
    _scan_legacy_layout,
    _write_pointer_atomic,
    global_pool_root,
    verify_image_self_consistent,
)
from mlpstorage_py.submission_checker.tools.code_checksum import compute_code_tree_md5

# Authoritative "this directory is a pool root" signal per D-72. Defined in
# code_image (the writer needs it too); re-exported here for the readers.
_SENTINEL_FILENAME = _POOL_SENTINEL_FILENAME

# Datetime-shaped run-leaf directory names: `YYYYMMDD_HHMMSS`. Mirrors
# `_TIMESTAMP_RE` in submission_checker/checks/pool_structure_checks.py so
# migration and CHECK-01 agree on what counts as a run leaf. Without this
# filter the fixed-depth globs below match subdirectories of a leaf
# (`dlio_config/`, `collector-staging/`, `.chk_iterations/`, ...), and
# pointer files land inside `dlio_config/` — breaking the 2.1.15/2.1.20/
# 2.1.26 exact-file-set checks (#725 Bug 1).
_LEAF_NAME_RE = re.compile(r"^\d{8}_\d{6}$")


@dataclass(frozen=True)
class VerifiedLegacyImage:
    """A legacy ``code/`` dir whose contents re-hash to its own ``.code-hash.json.hash``.

    Pass 1 emits this list; pass 2 iterates without re-hashing (D-73).
    Frozen so instances are safe to store in sets.
    """

    legacy_path: Path   # e.g. <rd>/closed/Acme/code/
    live_hash: str      # 32-hex md5-tree-v2 digest (post-verify)
    payload: dict       # .code-hash.json dict (forensic use only)


# ---------------------------------------------------------------------------
# Pass 1: verify-only
# ---------------------------------------------------------------------------

def _verify_all_legacy_dirs(
    results_dir: Path, orgname: str, log
) -> list[VerifiedLegacyImage]:
    """Pass 1: discover + re-hash every legacy ``code/`` dir (D-73).

    Raises HandEditedCodeImage before returning if ANY dir fails — pass 2 is
    unreachable on any mismatch (abort before any writes).
    """
    offenders = _scan_legacy_layout(results_dir, orgname)
    if not offenders:
        return []

    verified: list[VerifiedLegacyImage] = []
    for i, legacy_path in enumerate(offenders):
        remaining = len(offenders) - i - 1

        try:
            payload = _read_hash_file(legacy_path, log)
        except MissingHashFile as e:
            raise HandEditedCodeImage(
                f"hand-edited code image detected at {str(legacy_path)!r} "
                f"(no .code-hash.json — cannot verify content); "
                f"+{remaining} more legacy code images not yet checked. "
                f"Fix or delete offending dirs, then re-run."
            ) from e
        except MalformedHashFile as e:
            raise HandEditedCodeImage(
                f"hand-edited code image detected at {str(legacy_path)!r} "
                f"(malformed .code-hash.json: {e}); "
                f"+{remaining} more legacy code images not yet checked. "
                f"Fix or delete offending dirs, then re-run."
            ) from e

        stored = payload["hash"]
        live = compute_code_tree_md5(str(legacy_path), log)
        if live is None:
            raise HandEditedCodeImage(
                f"hand-edited code image detected at {str(legacy_path)!r} "
                f"(failed to re-hash contents); "
                f"+{remaining} more legacy code images not yet checked. "
                f"Fix or delete offending dirs, then re-run."
            )

        if live != stored:
            raise HandEditedCodeImage(
                f"hand-edited code image detected at {str(legacy_path)!r} "
                f"(recorded hash {stored} vs recomputed {live}); "
                f"+{remaining} more legacy code images not yet checked. "
                f"Fix or delete offending dirs, then re-run."
            )

        log.debug("legacy code/ at %s verified (hash %s)", legacy_path, live[:8])
        verified.append(
            VerifiedLegacyImage(legacy_path=legacy_path, live_hash=live, payload=payload)
        )

    return verified


# ---------------------------------------------------------------------------
# Pass 2: materialize → pointers → delete → sentinel
# ---------------------------------------------------------------------------

def _materialize_pool_images(
    pool_root: Path,
    verified: list[VerifiedLegacyImage],
    log,
) -> dict[str, Path]:
    """Pass 2 step 1: materialize each verified image into the pool (D-71).

    Returns a live_hash→pool_dir map used by step 2.
    """
    hash_to_pool: dict[str, Path] = {}
    for v in verified:
        if v.live_hash in hash_to_pool:
            continue
        existing = _find_matching_pool_image(pool_root, v.live_hash, log)
        if existing is not None:
            log.debug("dedup: legacy %s already materialized at %s", v.legacy_path, existing)
            hash_to_pool[v.live_hash] = existing
            continue
        pool_dir = _capture_new_pool_image(pool_root, v.legacy_path, v.live_hash, log)
        log.debug("materialized legacy %s as %s", v.legacy_path, pool_dir.name)
        hash_to_pool[v.live_hash] = pool_dir
    return hash_to_pool


def _write_pointers_for_migrated_leaves(
    results_dir: Path,
    orgname: str,
    verified: list[VerifiedLegacyImage],
    hash_to_pool: dict[str, Path],
    log,
) -> None:
    """Pass 2 step 2: write ``.mlps-code-image`` in every run leaf (D-71)."""
    for v in verified:
        subtree_root = v.legacy_path.parent  # <rd>/{closed|open}/<orgname>/
        for leaf in _enumerate_run_leaves(subtree_root, log):
            leaf.mkdir(parents=True, exist_ok=True)
            _write_pointer_atomic(leaf, v.live_hash, log)
            log.debug("pointer written to %s", leaf)


def _delete_legacy_dirs(verified: list[VerifiedLegacyImage], log) -> None:
    """Pass 2 step 3: rmtree every legacy ``code/`` dir (D-71).

    Idempotent: FileNotFoundError swallowed on crash-resume path.
    """
    for v in verified:
        try:
            shutil.rmtree(v.legacy_path)
            log.debug("deleted %s", v.legacy_path)
        except FileNotFoundError:
            log.debug("legacy %s already deleted (resume path)", v.legacy_path)


def _write_sentinel_atomic(pool_root: Path, log) -> Path:
    """Pass 2 step 4 (LAST): write ``.mlps-image-pool`` via tmp + os.rename (D-65, D-72)."""
    sentinel = pool_root / _SENTINEL_FILENAME
    tmp = pool_root / f"{_SENTINEL_FILENAME}.tmp.{os.getpid()}"
    if tmp.exists():
        tmp.unlink(missing_ok=True)
    content = (
        f"mlpstorage_version={MLPSTORAGE_VERSION}\n"
        f"migration_completed_at={_now_utc_iso()}\n"
    )
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.rename(str(tmp), str(sentinel))
    log.debug("wrote sentinel %s", sentinel)
    return sentinel


# ---------------------------------------------------------------------------
# Sentinel reader
# ---------------------------------------------------------------------------

def _read_sentinel(sentinel_path: Path, log) -> dict[str, str]:
    """Read sentinel content. Forward-compatible: unknown keys ignored.

    Returns empty dict if the file cannot be read.
    """
    try:
        text = sentinel_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return {}
    result: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip()
    return result


# ---------------------------------------------------------------------------
# Run-leaf enumerator
# ---------------------------------------------------------------------------

def _enumerate_run_leaves(subtree_root: Path, log) -> list[Path]:
    """Enumerate every run leaf under ``<rd>/{closed|open}/<orgname>/``.

    Bounded fixed-depth globs cover all three benchmark shapes:
    - Training/kv_cache (5-level): ``results/<sys>/<bench>/<model>/<cmd>/<dt>/``
    - Checkpointing (4-level): ``results/<sys>/checkpointing/<model>/<dt>/``
    - Vector_database (6-level): ``results/<sys>/<bench>/<eng>/<idx>/<cmd>/<dt>/``

    Each yielded dir's basename must match ``YYYYMMDD_HHMMSS``. The globs
    overlap between shapes (e.g. training's 5-level glob also matches
    checkpointing subdirs like ``<dt>/dlio_config``), so an explicit
    filename filter is required to keep pointer writes out of leaf
    subdirectories (#725 Bug 1).
    """
    results = subtree_root / "results"
    if not results.is_dir():
        return []

    leaves: list[Path] = []
    seen: set[Path] = set()

    for glob in ("*/*/*/*", "*/*/*/*/*", "*/*/*/*/*/*"):
        for p in results.glob(glob):
            if p in seen:
                continue
            if not _LEAF_NAME_RE.match(p.name):
                continue
            if not p.is_dir():
                continue
            leaves.append(p)
            seen.add(p)

    return leaves


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def migrate_legacy_layout(results_dir: Path, orgname: str, log) -> None:
    """Discover, verify, and migrate every legacy ``code/`` dir for ``orgname``.

    Two-pass per D-70/D-71/D-73/D-74:
    Pass 1 (_verify_all_legacy_dirs): re-hash all; abort before writes on mismatch.
    Pass 2 (fixed order D-71): materialize → pointers → delete → sentinel.

    Raises:
        HandEditedCodeImage: If any legacy dir fails re-hash in pass 1 (D-73).
    """
    # Pass 1. No try/except — D-73 structural invariant: pass 2 unreachable if
    # pass 1 raises.
    verified = _verify_all_legacy_dirs(results_dir, orgname, log)

    pool_root = global_pool_root(results_dir)
    pool_root.mkdir(parents=True, exist_ok=True)

    if not verified:
        # Sentinel absent + no legacy code/ (e.g. step-3-done-but-step-4-crashed).
        # N=0 is not a migration event — no status lines (D-74).
        _write_sentinel_atomic(pool_root, log)
        return

    n = len(verified)
    log.status(f"Migrating legacy code-image layout under {orgname} ({n} images)...")

    # D-71 FIXED STEP ORDER — do not reorder.
    hash_to_pool = _materialize_pool_images(pool_root, verified, log)                         # step 1
    _write_pointers_for_migrated_leaves(results_dir, orgname, verified, hash_to_pool, log)    # step 2
    _delete_legacy_dirs(verified, log)                                                         # step 3
    _write_sentinel_atomic(pool_root, log)                                                     # step 4

    m = len(hash_to_pool)
    log.status(f"Migrated {n} legacy code images into pool ({m} unique).")


# ---------------------------------------------------------------------------
# Per-organization pool → tree-wide pool relocation
# ---------------------------------------------------------------------------

def _org_pool_images(org_root: Path) -> list[Path]:
    """``code-*`` directories directly under ``org_root`` (sorted; [] if absent)."""
    if not org_root.is_dir():
        return []
    return sorted(p for p in org_root.glob("code-*") if p.is_dir())


def migrate_org_pool(results_dir: Path, orgname: str, log) -> int:
    """Relocate ``<results_dir>/<orgname>/code-*/`` into ``<results_dir>/code-images/``.

    Earlier releases kept one pool per organization at the org root. The
    tree-wide pool replaces it; run leaves are untouched because pointers
    are hash-only. Idempotent and crash-resumable — every step leaves a
    state that a re-run completes, and readers accept both layouts
    throughout:

      1. ensure the global pool root and its sentinel exist;
      2. move each image (``os.rename``; copy+delete on EXDEV). When the
         global pool already holds an image of the same name, the kept copy
         is verified self-consistent before the duplicate is deleted;
      3. remove the org-level sentinel;
      4. remove the org directory if nothing else is in it.

    Returns the number of images moved or de-duplicated (0 when there was
    no per-organization pool). Emits exactly two status lines when N > 0
    (D-74), none otherwise.

    Raises:
        HandEditedCodeImage: an org-pool image has no usable
            ``.code-hash.json`` (nothing to name the target by), or the
            global copy that would replace a duplicate does not re-hash to
            its own recorded digest.
    """
    org_root = Path(results_dir) / orgname
    images = _org_pool_images(org_root)
    org_sentinel = org_root / _SENTINEL_FILENAME
    if not images and not org_sentinel.exists():
        return 0

    pool_root = global_pool_root(results_dir)
    n = len(images)
    if n:
        log.status(
            f"Relocating {orgname}'s code-image pool into "
            f"{pool_root.name}/ ({n} images)..."
        )

    # Step 1: global pool root + sentinel first, so a crash after any move
    # leaves a tree that validates (CHECK-04 D-91 wants the sentinel).
    pool_root.mkdir(parents=True, exist_ok=True)
    if not (pool_root / _SENTINEL_FILENAME).exists():
        _write_sentinel_atomic(pool_root, log)

    # Step 2: move / de-duplicate.
    for image in images:
        try:
            stored = _read_hash_file(image, log)["hash"]
        except (MissingHashFile, MalformedHashFile) as e:
            raise HandEditedCodeImage(
                f"cannot relocate pool image {str(image)!r}: {e}. "
                f"Fix or delete it, then re-run."
            ) from e
        target = pool_root / image.name
        if target.is_dir():
            if not verify_image_self_consistent(target, log):
                raise HandEditedCodeImage(
                    f"pool image {str(target)!r} does not re-hash to its own "
                    f".code-hash.json; refusing to delete the duplicate at "
                    f"{str(image)!r}. Fix or delete one of them, then re-run."
                )
            kept = _read_hash_file(target, log)["hash"]
            if kept != stored:
                raise HandEditedCodeImage(
                    f"pool images {str(image)!r} and {str(target)!r} share a "
                    f"name but record different hashes ({stored} vs {kept}). "
                    f"Fix or delete one of them, then re-run."
                )
            shutil.rmtree(image)
            log.debug("dropped duplicate %s (kept %s)", image, target)
            continue
        try:
            os.rename(str(image), str(target))
        except OSError as e:
            if e.errno != errno.EXDEV:
                raise
            # Pool root on another filesystem: copy byte-for-byte (keeping
            # the original .code-hash.json) into a dot-prefixed tmp sibling,
            # rename it into place, then delete the original. A crash
            # mid-copy leaves only the invisible tmp, which the next run
            # discards and redoes.
            tmp = pool_root / f".{image.name}.tmp.{os.getpid()}"
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.copytree(image, tmp, symlinks=True)
            os.rename(str(tmp), str(target))
            shutil.rmtree(image)
        log.debug("moved %s -> %s", image, target)

    # Step 3 + 4: retire the org-level pool root.
    org_sentinel.unlink(missing_ok=True)
    for stray in org_root.glob(f"{_SENTINEL_FILENAME}.tmp.*"):
        stray.unlink(missing_ok=True)
    try:
        org_root.rmdir()
    except OSError:
        log.debug("%s not empty; leaving it in place", org_root)

    if n:
        log.status(f"Relocated {n} code images into {pool_root.name}/.")
    return n


# ---------------------------------------------------------------------------
# Pre-check helper (mirrors capture_or_verify_code_image signature)
# ---------------------------------------------------------------------------

def _check_and_migrate_legacy_layout(args, env, log) -> None:
    """Fast-path pre-check per D-70. Called BEFORE ``capture_or_verify_code_image``.

    Gates on submission mode/command; resolves orgname via the same shape
    as capture_or_verify_code_image. Then, in order:

    1. legacy ``code/`` dirs under ``{closed,open}/<orgname>/`` →
       ``migrate_legacy_layout`` (materializes into ``code-images/``);
    2. a per-organization pool at ``<results_dir>/<orgname>/`` (sentinel
       and/or ``code-*`` images) → ``migrate_org_pool``;
    3. images in ``code-images/`` without its sentinel (the #716 shape:
       a capture that crashed before writing it) → write the sentinel.

    Fast path (D-70): when ``code-images/.mlps-image-pool`` exists and
    ``<results_dir>/<orgname>/`` does not, there is nothing to migrate —
    two ``exists`` probes, no scan, no writes. Fresh trees get no sentinel
    until the first capture creates the pool.
    """
    mode = getattr(args, "mode", None)
    if mode not in _SUBMISSION_MODES:
        return

    command = getattr(args, "command", None)
    if command not in _SUBMISSION_COMMANDS:
        return

    # args.orgname is pinned from the sentinel by main's LAY-03 gate; there
    # is no environment fallback.
    orgname = getattr(args, "orgname", None)
    if not orgname:
        return

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        return

    pool_root = global_pool_root(results_dir)
    global_sentinel = pool_root / _SENTINEL_FILENAME
    if global_sentinel.exists() and not (results_dir / orgname).exists():
        return

    if _scan_legacy_layout(results_dir, orgname):
        migrate_legacy_layout(results_dir, orgname, log)

    migrate_org_pool(results_dir, orgname, log)

    if not global_sentinel.exists() and _org_pool_images(pool_root):
        _write_sentinel_atomic(pool_root, log)
