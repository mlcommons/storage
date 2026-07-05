"""PoolStructureCheck — validates the v1.1 content-addressed pool layout.

Implements CHECK-01..CHECK-04 (Phase 8) as a ``BaseCheck`` subclass with four
``@rule``-decorated methods. Runs as a pre-loop check in ``main.py:run()``
after ``SubmissionStructureCheck`` and ``SystemYamlSchemaCheck``.

Assumes Phase 7 migration has already run: a v1.0 tree (legacy ``code/`` dirs,
no ``.mlps-image-pool`` sentinel) is a CHECK-04 failure, not a tolerated state.

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
    _read_hash_file,
    _read_pointer,
    _scan_legacy_layout,
    CodeImageError,
    CodeTreeUnreadable,
    MalformedHashFile,
    MissingHashFile,
    PointerMalformed,
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

    def _discover_pool_orgs(self):
        """Return a list of (submitter, pool_root_path) for every org that has
        a ``.mlps-image-pool`` sentinel under the top-level results dir.

        Skips dot-prefixed entries and the reserved names ``closed``, ``open``,
        ``systems`` (D-83 / D-85).
        """
        orgs = []
        try:
            entries = os.listdir(self.root_path)
        except OSError:
            return orgs
        skip = {"closed", "open", "systems"}
        for entry in sorted(entries):
            if entry.startswith("."):
                continue
            if entry in skip:
                continue
            candidate = Path(self.root_path) / entry
            if not candidate.is_dir():
                continue
            if (candidate / ".mlps-image-pool").exists():
                orgs.append((entry, candidate))
        return orgs

    # -----------------------------------------------------------------------
    # CHECK-01 — poolPointerResolution
    # -----------------------------------------------------------------------

    @rule("CHECK-01", "poolPointerResolution")
    def pool_pointer_resolution_check(self):
        """CHECK-01: every datetime run-leaf must have a valid .mlps-code-image
        pointer that resolves to an existing pool image.

        D-84: if an org has closed/ or open/ entries but no pool root, emit ONE
        structural error per org instead of one per run leaf.
        D-93: missing pointer and dangling pointer both fire as CHECK-01, with
        distinct messages.
        """
        valid = True
        seen_orgs = set()

        for division, submitter, sub_path in self._iter_submitter_dirs():
            if submitter in seen_orgs:
                # CHECK-01 org-level structural error already emitted; still
                # walk leaves to catch dangling pointers when pool root IS found.
                pass

            org_root = Path(self.root_path) / submitter
            pool_sentinel = org_root / ".mlps-image-pool"

            if submitter not in seen_orgs:
                seen_orgs.add(submitter)
                # D-84: if pool sentinel is absent, emit one structural error
                # per org and skip per-leaf checks for this org.
                if not pool_sentinel.exists():
                    self.log_violation(
                        "CHECK-01", "poolPointerResolution",
                        str(org_root),
                        "No pool root found for org %s: missing %s/.mlps-image-pool. "
                        "Run mlpstorage to migrate.",
                        submitter, str(org_root),
                    )
                    valid = False
                    continue

            # Pool root exists; check each datetime leaf.
            if not pool_sentinel.exists():
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

                # Resolve to pool image path
                pool_dir = org_root / _pool_dir_name(full_hash)
                if not pool_dir.is_dir():
                    self.log_violation(
                        "CHECK-01", "poolPointerResolution",
                        leaf_path,
                        "run leaf %s .mlps-code-image references hash %s "
                        "but code-%s/ not found in pool.",
                        leaf_path, full_hash[:8], full_hash[:8],
                    )
                    valid = False

        return valid

    # -----------------------------------------------------------------------
    # CHECK-02 — poolImageSelfConsistency
    # -----------------------------------------------------------------------

    @rule("CHECK-02", "poolImageSelfConsistency")
    def pool_image_self_consistency_check(self):
        """CHECK-02: every pool image's contents must re-hash to the digest
        recorded in its .code-hash.json.

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
                except (MissingHashFile, MalformedHashFile, CodeImageError, CodeTreeUnreadable) as e:
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

        D-92: collect full hashes from ALL run leaves across closed/ AND open/
        for the org before checking. Cross-division dedup means a pool image
        referenced by either division is NOT an orphan.
        """
        valid = True

        for submitter, pool_root in self._discover_pool_orgs():
            # Collect all full hashes referenced by run leaves for this org
            referenced_hashes: set[str] = set()
            for _division, org_name, sub_path in self._iter_submitter_dirs():
                if org_name != submitter:
                    continue
                for leaf_path in self._iter_datetime_leaves(sub_path):
                    try:
                        _alg, full_hash = _read_pointer(Path(leaf_path), self.log)
                        referenced_hashes.add(full_hash)
                    except (FileNotFoundError, PointerMalformed):
                        # CHECK-01 already surfaces these; skip silently here
                        pass

            # Check each pool image against the referenced set
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

                if stored_hash not in referenced_hashes:
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
              failure.
        D-90: sentinel present but no pool images → warn (not fail).
        """
        valid = True
        seen_orgs: set[str] = set()

        for _division, submitter, _sub_path in self._iter_submitter_dirs():
            if submitter in seen_orgs:
                continue
            seen_orgs.add(submitter)

            org_root = Path(self.root_path) / submitter
            pool_sentinel = org_root / ".mlps-image-pool"

            # D-81: check for legacy code/ directories via _scan_legacy_layout
            offenders = _scan_legacy_layout(Path(self.root_path), submitter)
            if offenders:
                first = offenders[0]
                remaining = len(offenders) - 1
                msg = (
                    "Legacy code/ layout detected at %s. "
                    "Run mlpstorage against this results directory to "
                    "auto-migrate before revalidating."
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
            pool_images = list(org_root.glob("code-*/")) if org_root.is_dir() else []
            pool_images = [p for p in pool_images if p.is_dir()]

            if pool_images and not pool_sentinel.exists() and not offenders:
                self.log_violation(
                    "CHECK-04", "poolLegacyCheck",
                    str(org_root),
                    "Partial migration detected for org %s (pool images found "
                    "but .mlps-image-pool sentinel absent). Run mlpstorage to "
                    "complete migration.",
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
