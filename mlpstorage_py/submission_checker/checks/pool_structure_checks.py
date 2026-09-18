"""PoolStructureCheck — validates the content-addressed code-image pool layout.

Implements CHECK-01..CHECK-04 (Phase 8) as a ``BaseCheck`` subclass with four
``@rule``-decorated methods. Runs as a pre-loop check in ``main.py:run()``
after ``SubmissionStructureCheck`` and ``SystemYamlSchemaCheck``.

Two pool layouts are accepted, permanently:

- the tree-wide pool at ``<root>/code-images/code-<hash8>/`` (current);
- per-organization pools at ``<root>/<org>/code-<hash8>/`` (written by
  earlier releases; the frozen v3.0 submissions tree carries 19 of them).

Each pool root carries a ``.mlps-image-pool`` sentinel. A run leaf's pointer
resolves against the global pool first, then the leaf's own org pool
(``code_image.resolve_pool_image``). Assumes Phase 7 migration has already
run: a v1.0 tree (legacy ``code/`` dirs) is a CHECK-04 failure, not a
tolerated state.

Each method follows the accumulate-don't-abort pattern (QUAL-01): it collects
ALL violations in a subtree before returning ``False``, and NEVER raises out of
its body.

All violation messages are emitted via ``self.log_violation`` (hard errors) or
``self.warn_violation`` (advisory; D-90). Format is locked to:

    ``[<rule_id> <rule_name>] <path>: <msg>``
"""

import os
from pathlib import Path

from .base import BaseCheck
from ..configuration.configuration import Config
from ..rule_registry import rule
from ..tools.code_image import (
    _find_matching_pool_image,
    _pool_dir_name,
    _POOL_SENTINEL_FILENAME,
    _read_hash_file,
    _read_pointer,
    _scan_legacy_layout,
    CodeImageError,
    CodeTreeUnreadable,
    GLOBAL_POOL_DIRNAME,
    MalformedHashFile,
    MissingHashFile,
    PointerMalformed,
    global_pool_root,
    resolve_pool_image,
    verify_image_self_consistent,
)


# Allowed top-level submission divisions (case-sensitive)
_VALID_DIVISIONS = frozenset({"closed", "open"})

# Timestamp-like pattern: 8 digits, underscore, 6 digits (YYYYMMDD_HHmmss)
# Used to identify datetime leaf directories under results/
import re
_TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")


class PoolStructureCheck(BaseCheck):
    """Validate the v1.1 content-addressed code-image pool layout.

    Checks every org's pool root for pointer integrity (CHECK-01),
    pool-image self-consistency (CHECK-02), orphan detection (CHECK-03),
    and legacy/partial-migration detection (CHECK-04).

    Constructor:
        log: Logger object with ``error``, ``warning``, ``info``, ``debug``
            methods.
        config: ``Config`` instance (unused directly, carried for symmetry with
            sibling checks).
        root_path: The submission root directory (same value as ``args.input``
            in main.py).
    """

    def __init__(self, log, config: Config, root_path: str):
        super().__init__(log=log, path=root_path)
        self.config = config
        self.root_path = root_path
        self.name = "pool structure checks"
        self.init_checks()

    def init_checks(self):
        self.checks = []
        self.checks.extend([
            self.pool_pointer_resolution_check,
            self.pool_image_self_consistency_check,
            self.pool_orphan_check,
            self.pool_legacy_check,
        ])

    # -----------------------------------------------------------------------
    # Internal helpers (inline copy from SubmissionStructureCheck to avoid
    # circular dependency)
    # -----------------------------------------------------------------------

    def _iter_submitter_dirs(self):
        """Yield (division, submitter, submitter_path) for each known division."""
        try:
            for division in sorted(os.listdir(self.root_path)):
                if division not in _VALID_DIVISIONS:
                    continue
                div_path = os.path.join(self.root_path, division)
                if not os.path.isdir(div_path):
                    continue
                for submitter in sorted(os.listdir(div_path)):
                    sub_path = os.path.join(div_path, submitter)
                    if os.path.isdir(sub_path):
                        yield division, submitter, sub_path
        except OSError:
            return

    def _iter_datetime_leaves(self, sub_path: str):
        """Walk sub_path/results/ recursively to find datetime leaf directories.

        Yields absolute path strings for every directory whose name matches
        the YYYYMMDD_HHmmss timestamp pattern.
        """
        results_path = os.path.join(sub_path, "results")
        if not os.path.isdir(results_path):
            return
        for dirpath, dirnames, _files in os.walk(results_path):
            # Walk top-down; if a dirname matches the timestamp, yield it and
            # prune it from further traversal (leaf = no need to descend).
            prune = []
            for d in dirnames:
                if _TIMESTAMP_RE.match(d):
                    yield os.path.join(dirpath, d)
                    prune.append(d)
            for d in prune:
                dirnames.remove(d)

    def _global_pool_root(self) -> Path:
        return global_pool_root(Path(self.root_path))

    def _discover_pool_orgs(self):
        """Return ``[(label, pool_root_path)]`` for every pool root carrying a
        ``.mlps-image-pool`` sentinel: the tree-wide ``code-images/`` first
        (label ``GLOBAL_POOL_DIRNAME``), then each per-organization pool.

        Skips dot-prefixed entries and the reserved names ``closed``, ``open``,
        ``systems`` (D-83 / D-85).
        """
        roots = []
        global_root = self._global_pool_root()
        if (global_root / _POOL_SENTINEL_FILENAME).exists():
            roots.append((GLOBAL_POOL_DIRNAME, global_root))
        try:
            entries = os.listdir(self.root_path)
        except OSError:
            return roots
        skip = {"closed", "open", "systems", GLOBAL_POOL_DIRNAME}
        for entry in sorted(entries):
            if entry.startswith("."):
                continue
            if entry in skip:
                continue
            candidate = Path(self.root_path) / entry
            if not candidate.is_dir():
                continue
            if (candidate / _POOL_SENTINEL_FILENAME).exists():
                roots.append((entry, candidate))
        return roots

    def _referenced_hashes_by_org(self) -> dict[str, set[str]]:
        """``{submitter: {full_hash, ...}}`` from every readable run-leaf pointer
        across closed/ and open/. Unreadable pointers are CHECK-01's job."""
        refs: dict[str, set[str]] = {}
        for _division, submitter, sub_path in self._iter_submitter_dirs():
            bucket = refs.setdefault(submitter, set())
            for leaf_path in self._iter_datetime_leaves(sub_path):
                try:
                    _alg, full_hash = _read_pointer(Path(leaf_path), self.log)
                    bucket.add(full_hash)
                except (FileNotFoundError, PointerMalformed):
                    pass
        return refs

    # -----------------------------------------------------------------------
    # CHECK-01 — poolPointerResolution
    # -----------------------------------------------------------------------

    @rule("CHECK-01", "poolPointerResolution")
    def pool_pointer_resolution_check(self):
        """CHECK-01: every datetime run-leaf must have a valid .mlps-code-image
        pointer that resolves to an existing pool image — in the tree-wide
        ``code-images/`` pool or in the org's own ``<org>/`` pool.

        D-84: if an org has closed/ or open/ entries but no pool root at all
        (neither ``code-images/`` nor ``<org>/`` carries a sentinel), emit ONE
        structural error per org instead of one per run leaf.
        D-93: missing pointer and dangling pointer both fire as CHECK-01, with
        distinct messages.
        """
        valid = True
        root = Path(self.root_path)
        global_root = self._global_pool_root()
        global_sentinel = (global_root / _POOL_SENTINEL_FILENAME).exists()
        orgs_without_pool: set[str] = set()
        seen_orgs: set[str] = set()

        for division, submitter, sub_path in self._iter_submitter_dirs():
            org_root = root / submitter
            org_sentinel = (org_root / _POOL_SENTINEL_FILENAME).exists()

            if submitter not in seen_orgs:
                seen_orgs.add(submitter)
                # D-84: no pool root anywhere → one structural error per org.
                if not global_sentinel and not org_sentinel:
                    orgs_without_pool.add(submitter)
                    self.log_violation(
                        "CHECK-01", "poolPointerResolution",
                        str(org_root),
                        "No code-image pool found for org %s: neither %s/%s nor "
                        "%s/%s exists. Auto-migration fires on the next "
                        "`mlpstorage {closed|open} <benchmark> "
                        "{datasize|datagen|run}` invocation against this "
                        "--results-dir; `mlpstorage validate` does not migrate.",
                        submitter,
                        GLOBAL_POOL_DIRNAME, _POOL_SENTINEL_FILENAME,
                        submitter, _POOL_SENTINEL_FILENAME,
                    )
                    valid = False

            if submitter in orgs_without_pool:
                # Already emitted structural error above; skip leaves.
                continue

            for leaf_path in self._iter_datetime_leaves(sub_path):
                try:
                    _alg, full_hash = _read_pointer(Path(leaf_path), self.log)
                except FileNotFoundError:
                    self.log_violation(
                        "CHECK-01", "poolPointerResolution",
                        leaf_path,
                        "run leaf %s has no .mlps-code-image pointer.",
                        leaf_path,
                    )
                    valid = False
                    continue
                except PointerMalformed as e:
                    self.log_violation(
                        "CHECK-01", "poolPointerResolution",
                        leaf_path,
                        "%s", str(e),
                    )
                    valid = False
                    continue

                if resolve_pool_image(root, full_hash, submitter) is None:
                    name = _pool_dir_name(full_hash)
                    self.log_violation(
                        "CHECK-01", "poolPointerResolution",
                        leaf_path,
                        "run leaf %s .mlps-code-image references hash %s "
                        "but %s/ not found in pool (looked in %s/ and %s/).",
                        leaf_path, full_hash[:8], name,
                        GLOBAL_POOL_DIRNAME, submitter,
                    )
                    valid = False

        return valid

    # -----------------------------------------------------------------------
    # CHECK-02 — poolImageSelfConsistency
    # -----------------------------------------------------------------------

    @rule("CHECK-02", "poolImageSelfConsistency")
    def pool_image_self_consistency_check(self):
        """CHECK-02: every pool image's directory name must match its
        .code-hash.json.hash (first 8 chars), AND its contents must re-hash
        to the digest recorded in .code-hash.json.

        Two-part check per the CHECK-02 spec:
          1. Directory name matches .code-hash.json.hash (D-62 naming contract).
          2. Contents re-hash to .code-hash.json.hash (content self-consistency).

        Uses verify_image_self_consistent from code_image.py. Also catches
        MissingHashFile / MalformedHashFile / CodeImageError / CodeTreeUnreadable.
        """
        valid = True
        for _submitter, pool_root in self._discover_pool_orgs():
            # Find all code-<hash8>/ subdirs
            try:
                pool_entries = list(pool_root.glob("code-*/"))
            except OSError:
                continue
            for pool_dir in sorted(pool_entries):
                if not pool_dir.is_dir():
                    continue
                try:
                    # Part 1: verify directory name matches hash in .code-hash.json.
                    # Read the hash file first so we can derive the expected name.
                    try:
                        hash_data = _read_hash_file(pool_dir, self.log)
                        expected_dir_name = _pool_dir_name(hash_data["hash"])
                        if pool_dir.name != expected_dir_name:
                            self.log_violation(
                                "CHECK-02", "poolImageSelfConsistency",
                                str(pool_dir),
                                "pool image directory name %r does not match expected %r "
                                "(from .code-hash.json.hash %s)",
                                pool_dir.name, expected_dir_name, hash_data["hash"][:8],
                            )
                            valid = False
                            # Still run content self-consistency below for full diagnosis.
                    except (MissingHashFile, MalformedHashFile) as e:
                        # Part 2 also depends on this read; log and skip both.
                        self.log_violation(
                            "CHECK-02", "poolImageSelfConsistency",
                            str(pool_dir),
                            "%s", str(e),
                        )
                        valid = False
                        continue

                    # Part 2: content self-consistency.
                    ok = verify_image_self_consistent(pool_dir, self.log)
                    if not ok:
                        self.log_violation(
                            "CHECK-02", "poolImageSelfConsistency",
                            str(pool_dir),
                            "pool image %s: contents do not re-hash to recorded "
                            ".code-hash.json.hash.",
                            str(pool_dir),
                        )
                        valid = False
                except (CodeImageError, CodeTreeUnreadable) as e:
                    self.log_violation(
                        "CHECK-02", "poolImageSelfConsistency",
                        str(pool_dir),
                        "%s", str(e),
                    )
                    valid = False

        return valid

    # -----------------------------------------------------------------------
    # CHECK-03 — poolOrphanCheck
    # -----------------------------------------------------------------------

    @rule("CHECK-03", "poolOrphanCheck")
    def pool_orphan_check(self):
        """CHECK-03: every pool image must be referenced by at least one run leaf.

        D-92: references are collected from ALL run leaves across closed/ AND
        open/ before checking. An image in the tree-wide ``code-images/``
        pool is referenced when ANY organization's leaf names its hash; an
        image in a per-organization pool is referenced only by that
        organization's own leaves (its pool, its runs — unchanged from the
        per-org era, so the frozen v3.0 tree validates exactly as before).
        """
        valid = True
        by_org = self._referenced_hashes_by_org()
        all_refs: set[str] = set().union(*by_org.values()) if by_org else set()

        for label, pool_root in self._discover_pool_orgs():
            referenced = all_refs if label == GLOBAL_POOL_DIRNAME else by_org.get(label, set())
            try:
                pool_entries = list(pool_root.glob("code-*/"))
            except OSError:
                continue
            for pool_dir in sorted(pool_entries):
                if not pool_dir.is_dir():
                    continue
                try:
                    stored = _read_hash_file(pool_dir, self.log)
                    stored_hash = stored["hash"]
                except (MissingHashFile, MalformedHashFile):
                    # CHECK-02 surfaces this; skip
                    continue

                if stored_hash not in referenced:
                    self.log_violation(
                        "CHECK-03", "poolOrphanCheck",
                        str(pool_dir),
                        "pool image %s is not referenced by any run leaf (orphan).",
                        str(pool_dir),
                    )
                    valid = False

        return valid

    # -----------------------------------------------------------------------
    # CHECK-04 — poolLegacyCheck
    # -----------------------------------------------------------------------

    @rule("CHECK-04", "poolLegacyCheck")
    def pool_legacy_check(self):
        """CHECK-04: no legacy unhashed code/ directories must exist; and partial
        migrations (pool images without sentinel) are also failures.

        D-81: legacy code/ dirs → actionable 'migrate first' message naming the
              first offender + count of remaining.
        D-91: pool images found but .mlps-image-pool absent → partial migration
              failure. Applies to the tree-wide ``code-images/`` root and to
              each per-organization root.
        D-90: sentinel present but no pool images → warn (not fail).

        A per-organization pool that carries its sentinel is a supported
        layout, not a legacy one: this check says nothing about it.
        """
        valid = True
        root = Path(self.root_path)

        def _pool_images(pool_root: Path) -> list[Path]:
            if not pool_root.is_dir():
                return []
            return [p for p in pool_root.glob("code-*/") if p.is_dir()]

        # Tree-wide pool root: D-91 / D-90.
        global_root = self._global_pool_root()
        global_images = _pool_images(global_root)
        global_sentinel = (global_root / _POOL_SENTINEL_FILENAME).exists()
        if global_images and not global_sentinel:
            self.log_violation(
                "CHECK-04", "poolLegacyCheck",
                str(global_root),
                "Partial migration detected: %s/ holds pool images but no "
                "%s sentinel. Auto-heal fires on the next `mlpstorage "
                "{closed|open} <benchmark> {datasize|datagen|run}` "
                "invocation against this --results-dir; `mlpstorage "
                "validate` does not migrate.",
                GLOBAL_POOL_DIRNAME, _POOL_SENTINEL_FILENAME,
            )
            valid = False
        if global_sentinel and not global_images:
            self.warn_violation(
                "CHECK-04", "poolLegacyCheck",
                str(global_root),
                "Pool sentinel present in %s/ but no pool images found "
                "— nothing to verify.",
                GLOBAL_POOL_DIRNAME,
            )

        seen_orgs: set[str] = set()
        for _division, submitter, _sub_path in self._iter_submitter_dirs():
            if submitter in seen_orgs:
                continue
            seen_orgs.add(submitter)

            org_root = root / submitter
            pool_sentinel = org_root / _POOL_SENTINEL_FILENAME

            # D-81: check for legacy code/ directories via _scan_legacy_layout
            offenders = _scan_legacy_layout(root, submitter)
            if offenders:
                first = offenders[0]
                remaining = len(offenders) - 1
                msg = (
                    "Legacy code/ layout detected at %s. "
                    "Auto-migration fires on the next `mlpstorage "
                    "{closed|open} <benchmark> {datasize|datagen|run}` "
                    "invocation against this --results-dir; `mlpstorage "
                    "validate` does not migrate."
                )
                if remaining > 0:
                    msg += " (%d additional legacy code/ directories found)" % remaining
                self.log_violation(
                    "CHECK-04", "poolLegacyCheck",
                    str(first),
                    msg,
                    str(first),
                )
                valid = False
                # Still continue to check for partial migration below

            # D-91: pool images present but .mlps-image-pool sentinel absent
            # (and no legacy code/ offenders — the more specific legacy error
            # has priority; only fire partial-migration if legacy dirs not found)
            pool_images = _pool_images(org_root)

            if pool_images and not pool_sentinel.exists() and not offenders:
                self.log_violation(
                    "CHECK-04", "poolLegacyCheck",
                    str(org_root),
                    "Partial migration detected for org %s (pool images found "
                    "but .mlps-image-pool sentinel absent). Auto-heal fires on "
                    "the next `mlpstorage {closed|open} <benchmark> "
                    "{datasize|datagen|run}` invocation against this "
                    "--results-dir; `mlpstorage validate` does not migrate.",
                    submitter,
                )
                valid = False

            # D-90: sentinel present but no pool images → advisory warning
            if pool_sentinel.exists() and not pool_images:
                self.warn_violation(
                    "CHECK-04", "poolLegacyCheck",
                    str(org_root),
                    "Pool sentinel present for %s but no pool images found "
                    "— nothing to verify.",
                    submitter,
                )
                # Not a hard failure (D-90 says warn, don't fail)

        return valid
