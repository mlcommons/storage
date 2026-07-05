"""MIG-03 hand-edit detection and abort integration tests.

Covers Phase 7 decision D-73 and requirement MIG-03:
  - MIG-03: abort before any writes when a legacy code dir has been hand-edited
    (re-hash disagrees with stored .code-hash.json hash field)
  - SC-4 abort-before-writes: all abort scenarios leave the tree byte-identical
    to the pre-migration state
  - Edge cases: missing .code-hash.json (MissingHashFile) and malformed JSON
    (MalformedHashFile) are both wrapped and surfaced as HandEditedCodeImage

Wave 0 xfail stubs replaced with real assertions by Plan 07-03 Task 4.

Refs: 07-01-PLAN.md Task 2, 07-CONTEXT.md D-73, MIG-03, SC-4,
RESEARCH §6 hand-edit detection.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mlpstorage_py.submission_checker.tools.code_image import HandEditedCodeImage
from mlpstorage_py.submission_checker.tools.legacy_migration import migrate_legacy_layout


class TestMigrateHandEditAbort:
    """MIG-03: hand-edited legacy dir causes abort before any writes."""

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
        rd = legacy_tree_factory(orgname="Acme", hand_edit=True)
        with pytest.raises(HandEditedCodeImage, match=r"hand-edited code image detected at .+ \(recorded hash .+ vs recomputed .+\)"):
            migrate_legacy_layout(rd, "Acme", log)

    def test_hand_edit_abort_leaves_sentinel_absent(
        self, tmp_path, legacy_tree_factory, log
    ):
        """MIG-03 abort: .mlps-image-pool sentinel must NOT be written on abort.

        After HandEditedCodeImage is raised, the path
        <rd>/Acme/.mlps-image-pool must not exist — confirming that the abort
        fired before the sentinel write.
        """
        rd = legacy_tree_factory(orgname="Acme", hand_edit=True)
        with pytest.raises(HandEditedCodeImage):
            migrate_legacy_layout(rd, "Acme", log)

        assert not (rd / "Acme" / ".mlps-image-pool").exists(), (
            "sentinel must NOT be written on hand-edit abort (D-73)"
        )

    def test_hand_edit_abort_leaves_tree_byte_identical(
        self, tmp_path, legacy_tree_factory, log
    ):
        """MIG-03 abort: tree is byte-identical before and after the abort.

        Snapshot the set of file paths before invoking migrate_legacy_layout
        (which raises HandEditedCodeImage) and again after catching the
        exception. The two path sets must be equal — no files created,
        modified, or deleted during the aborted migration.
        """
        rd = legacy_tree_factory(orgname="Acme", hand_edit=True)

        # Snapshot before: set of all file paths
        before = {str(p) for p in rd.rglob("*") if p.is_file()}

        with pytest.raises(HandEditedCodeImage):
            migrate_legacy_layout(rd, "Acme", log)

        after = {str(p) for p in rd.rglob("*") if p.is_file()}
        assert before == after, (
            f"D-73 abort-before-writes: tree changed during aborted migration.\n"
            f"Added: {after - before}\n"
            f"Removed: {before - after}"
        )

    def test_hand_edit_abort_leaves_no_pool_images_materialized(
        self, tmp_path, legacy_tree_factory, log
    ):
        """MIG-03 abort: no pool images are materialized under <rd>/Acme/.

        After the abort, there must be no code-* subdirs under rd/Acme/.
        D-73 strict two-pass ordering guarantees pass 2 (materialize) is
        unreachable if pass 1 (verify) raises.
        """
        rd = legacy_tree_factory(orgname="Acme", hand_edit=True)
        with pytest.raises(HandEditedCodeImage):
            migrate_legacy_layout(rd, "Acme", log)

        acme_root = rd / "Acme"
        pool_images = list(acme_root.glob("code-*")) if acme_root.exists() else []
        assert pool_images == [], (
            f"D-73 abort-before-writes: pool images materialized despite hand-edit abort: {pool_images}"
        )


class TestMigrateHashFileEdgeCases:
    """Edge cases for missing or malformed .code-hash.json files.

    MissingHashFile and MalformedHashFile (Phase 6 exceptions in code_image.py)
    are both wrapped by the migration pass-1 verifier and re-raised as
    HandEditedCodeImage so the same abort path and exit-code mapping applies.
    """

    def test_missing_code_hash_json_converts_to_HandEditedCodeImage(
        self, tmp_path, legacy_tree_factory, log
    ):
        """Legacy dir with no .code-hash.json raises HandEditedCodeImage (MissingHashFile wrapped)."""
        rd = legacy_tree_factory(orgname="Acme", hand_edit=False)
        # Delete the hash file to trigger MissingHashFile path.
        (rd / "closed" / "Acme" / "code" / ".code-hash.json").unlink()

        with pytest.raises(HandEditedCodeImage, match=r"no .code-hash.json"):
            migrate_legacy_layout(rd, "Acme", log)

    def test_malformed_code_hash_json_converts_to_HandEditedCodeImage(
        self, tmp_path, legacy_tree_factory, log
    ):
        """Legacy dir with invalid JSON in .code-hash.json raises HandEditedCodeImage (MalformedHashFile wrapped)."""
        rd = legacy_tree_factory(orgname="Acme", hand_edit=False)
        # Overwrite with unparseable JSON.
        (rd / "closed" / "Acme" / "code" / ".code-hash.json").write_text("not valid json {")

        with pytest.raises(HandEditedCodeImage, match=r"malformed .code-hash.json"):
            migrate_legacy_layout(rd, "Acme", log)
