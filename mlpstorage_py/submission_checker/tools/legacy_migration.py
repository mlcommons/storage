"""One-shot legacy migration coordinator for Phase 7.

Performs automatic, idempotent, crash-resumable migration of v1.0-layout
``code/`` directories into the Phase 6 content-addressed pool.

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
    _check_and_migrate_legacy_layout(args, env, log) -> None
    _read_sentinel(sentinel_path, log) -> dict[str, str]
    VerifiedLegacyImage (dataclass)
"""

from __future__ import annotations

import os
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
    _read_hash_file,
    _scan_legacy_layout,
    _write_pointer_atomic,
)
from mlpstorage_py.submission_checker.tools.code_checksum import compute_code_tree_md5
from mlpstorage_py.rules.utils import MLPSTORAGE_ORGNAME_ENVVAR

# Authoritative "migration done" signal per D-72.
_SENTINEL_FILENAME = ".mlps-image-pool"


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
    org_root: Path,
    verified: list[VerifiedLegacyImage],
    log,
) -> dict[str, Path]:
    """Pass 2 step 1: materialize each verified image into the Phase 6 pool (D-71).

    Returns a live_hash→pool_dir map used by step 2.
    """
    hash_to_pool: dict[str, Path] = {}
    for v in verified:
        if v.live_hash in hash_to_pool:
            continue
        existing = _find_matching_pool_image(org_root, v.live_hash, log)
        if existing is not None:
            log.debug("dedup: legacy %s already materialized at %s", v.legacy_path, existing)
            hash_to_pool[v.live_hash] = existing
            continue
        pool_dir = _capture_new_pool_image(org_root, v.legacy_path, v.live_hash, log)
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


def _write_sentinel_atomic(org_root: Path, log) -> Path:
    """Pass 2 step 4 (LAST): write ``.mlps-image-pool`` via tmp + os.rename (D-65, D-72)."""
    sentinel = org_root / _SENTINEL_FILENAME
    tmp = org_root / f"{_SENTINEL_FILENAME}.tmp.{os.getpid()}"
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
    """
    results = subtree_root / "results"
    if not results.is_dir():
        return []

    leaves: list[Path] = []
    seen: set[Path] = set()

    for p in results.glob("*/*/*/*/*"):
        if p.is_dir() and p not in seen:
            leaves.append(p)
            seen.add(p)

    for p in results.glob("*/*/*/*"):
        if p.is_dir() and p not in seen and "checkpointing" in p.parts:
            leaves.append(p)
            seen.add(p)

    for p in results.glob("*/*/*/*/*/*"):
        if p.is_dir() and p not in seen:
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

    org_root = results_dir / orgname
    org_root.mkdir(parents=True, exist_ok=True)

    if not verified:
        # Sentinel absent + no legacy code/ (e.g. step-3-done-but-step-4-crashed).
        # N=0 is not a migration event — no status lines (D-74).
        _write_sentinel_atomic(org_root, log)
        return

    n = len(verified)
    log.status(f"Migrating legacy code-image layout under {orgname} ({n} images)...")

    # D-71 FIXED STEP ORDER — do not reorder.
    hash_to_pool = _materialize_pool_images(org_root, verified, log)                          # step 1
    _write_pointers_for_migrated_leaves(results_dir, orgname, verified, hash_to_pool, log)    # step 2
    _delete_legacy_dirs(verified, log)                                                         # step 3
    _write_sentinel_atomic(org_root, log)                                                      # step 4

    m = len(hash_to_pool)
    log.status(f"Migrated {n} legacy code images into pool ({m} unique).")


# ---------------------------------------------------------------------------
# Pre-check helper (mirrors capture_or_verify_code_image signature)
# ---------------------------------------------------------------------------

def _check_and_migrate_legacy_layout(args, env, log) -> None:
    """Fast-path pre-check per D-70. Called BEFORE ``capture_or_verify_code_image``.

    O(2) syscalls when sentinel present (common case after first migration).
    Gates on submission mode/command; resolves orgname via same shape as
    capture_or_verify_code_image.

    Fresh-tree behaviour (A3b): sentinel absent + no legacy code/ → return without
    writing sentinel. The sentinel marks "migration ran here"; fresh trees have no
    migration event.
    """
    mode = getattr(args, "mode", None)
    if mode not in _SUBMISSION_MODES:
        return

    command = getattr(args, "command", None)
    if command not in _SUBMISSION_COMMANDS:
        return

    orgname = getattr(args, "orgname", None) or env.get(MLPSTORAGE_ORGNAME_ENVVAR)
    if not orgname:
        return

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        return

    org_root = results_dir / orgname
    sentinel = org_root / _SENTINEL_FILENAME
    if sentinel.exists():
        return

    offenders = _scan_legacy_layout(results_dir, orgname)
    if not offenders:
        # Sentinel absent + no legacy code/ dirs. If pool images already exist
        # (capture_or_verify_code_image ran first on a fresh tree), we must
        # still write the sentinel — otherwise CHECK-04 D-91 will flag the
        # tree as a partial migration forever. migrate_legacy_layout handles
        # the N=0 case by just writing the sentinel atomically.
        if org_root.is_dir() and any(org_root.glob("code-*")):
            migrate_legacy_layout(results_dir, orgname, log)
        return

    migrate_legacy_layout(results_dir, orgname, log)
