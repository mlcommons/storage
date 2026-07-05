"""Wave-0 xfail scaffolding for MIG-03 hand-edit detection and abort tests.

Covers Phase 7 decision D-73 and requirement MIG-03:
  - MIG-03: abort before any writes when a legacy code dir has been hand-edited
    (re-hash disagrees with stored .code-hash.json hash field)
  - SC-4 abort-before-writes: all abort scenarios leave the tree byte-identical
    to the pre-migration state
  - Edge cases: missing .code-hash.json (MissingHashFile) and malformed JSON
    (MalformedHashFile) are both wrapped and surfaced as HandEditedCodeImage

Wave 0 note: every test stub raises NotImplementedError and is marked
xfail(strict=True). Wave-2 (Plan 07-03) removes xfail decorators and
populates test bodies by importing production symbols from
``mlpstorage_py.submission_checker.tools.legacy_migration`` and
``mlpstorage_py.submission_checker.tools.code_image`` (neither exists until
Plan 07-02). Symbol names appear in docstrings only — no import yet.

Refs: 07-01-PLAN.md Task 2, 07-CONTEXT.md D-73, MIG-03, SC-4,
RESEARCH §6 hand-edit detection.
"""

from __future__ import annotations

import pytest
from pathlib import Path


class TestMigrateHandEditAbort:
    """MIG-03: hand-edited legacy dir causes abort before any writes."""

    @pytest.mark.xfail(strict=True, reason="Wave 0 scaffold — implementation in Plan 07-03/07-04", raises=NotImplementedError)
    def test_hand_edited_legacy_raises_HandEditedCodeImage(
        self, tmp_path, legacy_tree_factory, log
    ):
        """hand_edit=True: migrate_legacy_layout raises HandEditedCodeImage.

        The legacy_tree_factory with hand_edit=True overwrites file1.py AFTER
        writing .code-hash.json, causing a re-hash mismatch. The production
        migrate_legacy_layout must raise HandEditedCodeImage (new exception,
        subclassing CodeImageError) with a message matching the pattern
        r"hand-edited code image detected".

        D-73 pass-1 contract: the raise happens BEFORE any pool-write, pointer-
        write, legacy-delete, or sentinel-write.
        """
        raise NotImplementedError(
            "Wave 0 stub — implementation lands with production module"
        )

    @pytest.mark.xfail(strict=True, reason="Wave 0 scaffold — implementation in Plan 07-03/07-04", raises=NotImplementedError)
    def test_hand_edit_abort_leaves_sentinel_absent(
        self, tmp_path, legacy_tree_factory, log
    ):
        """MIG-03 abort: .mlps-image-pool sentinel must NOT be written on abort.

        After HandEditedCodeImage is raised, the path
        <rd>/Acme/.mlps-image-pool must not exist — confirming that the abort
        fired before the sentinel write.
        """
        raise NotImplementedError(
            "Wave 0 stub — implementation lands with production module"
        )

    @pytest.mark.xfail(strict=True, reason="Wave 0 scaffold — implementation in Plan 07-03/07-04", raises=NotImplementedError)
    def test_hand_edit_abort_leaves_tree_byte_identical(
        self, tmp_path, legacy_tree_factory, log
    ):
        """MIG-03 abort: tree is byte-identical before and after the abort.

        Snapshot ``sorted(str(p) for p in rd.rglob("*"))`` before invoking
        migrate_legacy_layout (which raises HandEditedCodeImage) and again
        after catching the exception. The two snapshots must be equal —
        no files created, modified, or deleted during the aborted migration.
        """
        raise NotImplementedError(
            "Wave 0 stub — implementation lands with production module"
        )

    @pytest.mark.xfail(strict=True, reason="Wave 0 scaffold — implementation in Plan 07-03/07-04", raises=NotImplementedError)
    def test_hand_edit_abort_leaves_no_pool_images_materialized(
        self, tmp_path, legacy_tree_factory, log
    ):
        """MIG-03 abort: no pool images are materialized under <rd>/Acme/.

        After the abort, ``list((rd / "Acme").glob("code-*")) == []``.
        D-73 strict two-pass ordering guarantees pass 2 (materialize) is
        unreachable if pass 1 (verify) raises.
        """
        raise NotImplementedError(
            "Wave 0 stub — implementation lands with production module"
        )


class TestMigrateHashFileEdgeCases:
    """Edge cases for missing or malformed .code-hash.json files.

    MissingHashFile and MalformedHashFile (Phase 6 exceptions in code_image.py)
    are both wrapped by the migration pass-1 verifier and re-raised as
    HandEditedCodeImage so the same abort path and exit-code mapping applies.
    """

    @pytest.mark.xfail(strict=True, reason="Wave 0 scaffold — implementation in Plan 07-03/07-04", raises=NotImplementedError)
    def test_missing_code_hash_json_converts_to_HandEditedCodeImage(
        self, tmp_path, legacy_tree_factory, log
    ):
        """Legacy dir with no .code-hash.json raises HandEditedCodeImage (MissingHashFile wrapped).

        A legacy code/ dir that exists but lacks .code-hash.json is treated as
        an unverifiable (possibly hand-edited) image. Migration raises
        HandEditedCodeImage wrapping MissingHashFile — same abort path as a
        hash-mismatch.
        """
        raise NotImplementedError(
            "Wave 0 stub — implementation lands with production module"
        )

    @pytest.mark.xfail(strict=True, reason="Wave 0 scaffold — implementation in Plan 07-03/07-04", raises=NotImplementedError)
    def test_malformed_code_hash_json_converts_to_HandEditedCodeImage(
        self, tmp_path, legacy_tree_factory, log
    ):
        """Legacy dir with invalid JSON in .code-hash.json raises HandEditedCodeImage (MalformedHashFile wrapped).

        A legacy code/ dir whose .code-hash.json contains unparseable JSON is
        treated as an unverifiable image. Migration raises HandEditedCodeImage
        wrapping MalformedHashFile — same abort path as a hash-mismatch.
        """
        raise NotImplementedError(
            "Wave 0 stub — implementation lands with production module"
        )
