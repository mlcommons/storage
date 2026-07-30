"""Pinned Public IDs — the ``public_ids.json`` registry.

Numbering the rows by position — sort ``results.csv``, then count from
``v3.0-0001`` — renumbers everything after any row that is inserted or
withheld. That is harmless until reviewers, issues and assignment documents
cite IDs by number, and then it is worse than an error: a stale citation does
not look broken, it points at a different, valid row.

So reportgen keeps a ``public_ids.json`` registry beside the global
``results.csv``:

- **Created automatically.** A tree without one gets one, filled from that
  run's positional assignment. Pinning is a ratchet nobody has to remember to
  engage.
- **Sticky thereafter.** A row keeps its ID no matter how the table re-sorts
  around it. A genuinely new row mints ``max + 1``, so organizations that
  arrive after the ratchet engages number above the rows already published
  rather than displacing them.
- **IDs are never reused.** A row that stops publishing leaves its ID reserved
  and a gap in the table. Reusing the number would silently point an old
  citation at a different submission. Restoring the row restores its ID.
- **Deleting the file is how you renumber.** Staff remove it once, before
  publication, to close the gaps and restore sort order for the released
  table. Nothing else in the tool renumbers.

Because deletion is the renumber trigger, the tool cannot distinguish a
deliberate renumber from a registry that was lost — a stale clone, a file
never checked out, a ``git clean``. Both produce a plausible, wrong numbering
recorded as authoritative. So the guard does not gate on intent; it reports
the consequence, and reads correctly either way: **any run that changes an ID
the tree's existing results.csv already published says so, and says how many.**
Normal pinned operation is silent, because pinning never changes a matched
row's ID and minted rows were not in the old table.

Row identity is the generator's own workload key (D-05/D-06), not the row's
position and not the display columns: ``(category, orgname, systemname, id1,
id2)`` plus the benchmark type. That matters because the CSV's ``Model`` column
is blank for kv_cache and vector_database rows while the key's ``id1``/``id2``
carry model + performance profile / engine + index type — two kv_cache profiles
for one system are two distinct rows that a display-column key would collapse.
"""

from __future__ import annotations

import json
import pathlib
import shutil
from argparse import Namespace

import pytest

from mlpstorage_py.report_generator import ReportGenerator

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_FIXTURES_ROOT = _REPO_ROOT / "tests" / "fixtures" / "sample_results"

_REGISTRY_NAME = "public_ids.json"


def _prepare_tree(tmp_path: pathlib.Path, name: str = "repo_root") -> pathlib.Path:
    dest = tmp_path / name
    shutil.copytree(_FIXTURES_ROOT / "multi_orgname", dest)
    return dest


def _run(root: pathlib.Path) -> ReportGenerator:
    gen = ReportGenerator(str(root), args=Namespace(debug=False),
                          validate_structure=False)
    rc = gen.generate_reports()
    assert rc == 0, f"generate_reports returned {rc}"
    return gen


def _global_rows(root: pathlib.Path) -> list:
    return json.loads((root / "results.json").read_text())


def _ids_by_org(root: pathlib.Path) -> dict:
    return {r["Organization"]: r["Public ID"] for r in _global_rows(root)}


def _seed_empty_registry(root: pathlib.Path) -> None:
    (root / _REGISTRY_NAME).write_text(json.dumps({"assignments": []}, indent=2))


def _registry(root: pathlib.Path) -> dict:
    return json.loads((root / _REGISTRY_NAME).read_text())


def _clone_org_as(root: pathlib.Path, src: str, dest: str) -> None:
    """Clone a whole org tree under a name that sorts before the others."""
    shutil.copytree(root / "closed" / src, root / "closed" / dest)


_BETA_RUN_LEAF = pathlib.PurePath(
    "closed/beta_corp/results/system-b/training/unet3d/run/20260706_101500"
)


def _withhold_beta_row(root: pathlib.Path) -> pathlib.Path:
    """Make beta_corp stop publishing, without removing the organization.

    Deleting the whole org directory would drop the tree from two orgs to
    one, and a single-org tree puts its global rollup (and therefore the
    registry) somewhere else entirely — the row would vanish for a reason
    that has nothing to do with pinning. Removing just the run leaf leaves
    the org in place and the layout unchanged.
    """
    leaf = root / _BETA_RUN_LEAF
    stash = root.parent / "beta_run_leaf_stash"
    shutil.copytree(leaf, stash)
    shutil.rmtree(leaf)
    return stash


def _restore_beta_row(root: pathlib.Path, stash: pathlib.Path) -> None:
    shutil.copytree(stash, root / _BETA_RUN_LEAF)


class TestCreatedAutomatically:
    def test_a_tree_without_a_registry_gets_one(self, tmp_path):
        """The ratchet engages by itself — no magic file to remember."""
        root = _prepare_tree(tmp_path)
        assert not (root / _REGISTRY_NAME).exists()
        _run(root)
        reg = _registry(root)
        assert [a["public_id"] for a in reg["assignments"]] == [
            "v3.0-0001", "v3.0-0002",
        ]
        assert reg["next_index"] == 3

    def test_creation_numbers_positionally(self, tmp_path):
        """The created registry records the sorted, contiguous numbering —
        this is what the release-time renumber relies on."""
        root = _prepare_tree(tmp_path)
        _run(root)
        assert _ids_by_org(root) == {
            "acme": "v3.0-0001",
            "beta_corp": "v3.0-0002",
        }

    def test_creation_on_a_fresh_tree_is_silent(self, tmp_path, caplog):
        """A submitter's first run has nothing to warn about."""
        root = _prepare_tree(tmp_path)
        caplog.clear()
        with caplog.at_level("WARNING"):
            _run(root)
        noisy = [
            r.getMessage() for r in caplog.records
            if "public id" in r.getMessage().lower()
        ]
        assert not noisy, f"unexpected Public ID warning on a fresh tree: {noisy}"

    def test_pinning_takes_effect_from_the_second_run(self, tmp_path):
        """Created on run 1, honored on run 2: an org sorting first takes the
        NEXT number rather than displacing acme."""
        root = _prepare_tree(tmp_path)
        _run(root)
        _clone_org_as(root, "acme", "aaa_corp")
        _run(root)
        assert _ids_by_org(root) == {
            "aaa_corp": "v3.0-0003",
            "acme": "v3.0-0001",
            "beta_corp": "v3.0-0002",
        }


class TestChangedIdGuard:
    """Any run that changes an already-published id must say so."""

    def _public_id_warnings(self, caplog) -> list:
        return [
            r.getMessage() for r in caplog.records
            if "changed the public id" in r.getMessage().lower()
        ]

    def test_deleting_the_registry_renumbers_and_warns(self, tmp_path, caplog):
        """The release-time renumber. Loud, and correct to be loud."""
        root = _prepare_tree(tmp_path)
        _run(root)
        _clone_org_as(root, "acme", "aaa_corp")
        _run(root)
        assert _ids_by_org(root)["aaa_corp"] == "v3.0-0003"

        (root / _REGISTRY_NAME).unlink()
        caplog.clear()
        with caplog.at_level("WARNING"):
            _run(root)

        assert _ids_by_org(root) == {
            "aaa_corp": "v3.0-0001",
            "acme": "v3.0-0002",
            "beta_corp": "v3.0-0003",
        }, "deleting the registry must renumber contiguously in sort order"
        warned = self._public_id_warnings(caplog)
        assert warned, "renumbering silently is the whole failure mode"
        assert "3" in warned[0], f"the count of changed rows must be named: {warned[0]}"

    def test_the_warning_says_how_to_recover(self, tmp_path, caplog):
        """It has to read correctly for the staffer who deleted the file AND
        for the one whose registry went missing."""
        root = _prepare_tree(tmp_path)
        _run(root)
        _clone_org_as(root, "acme", "aaa_corp")
        _run(root)
        (root / _REGISTRY_NAME).unlink()
        caplog.clear()
        with caplog.at_level("WARNING"):
            _run(root)
        text = " ".join(self._public_id_warnings(caplog)).lower()
        assert "restore" in text, "must tell a reader whose registry was lost what to do"

    def test_an_emptied_registry_warns_the_same_way(self, tmp_path, caplog):
        """Guarding only on absence would miss a truncated file."""
        root = _prepare_tree(tmp_path)
        _run(root)
        _clone_org_as(root, "acme", "aaa_corp")
        _run(root)
        _seed_empty_registry(root)
        caplog.clear()
        with caplog.at_level("WARNING"):
            _run(root)
        assert self._public_id_warnings(caplog), (
            "an emptied registry renumbers exactly like a deleted one"
        )

    def test_a_normal_pinned_run_is_silent(self, tmp_path, caplog):
        root = _prepare_tree(tmp_path)
        _run(root)
        caplog.clear()
        with caplog.at_level("WARNING"):
            _run(root)
        assert not self._public_id_warnings(caplog)

    def test_minting_a_new_row_does_not_trip_the_guard(self, tmp_path, caplog):
        """A new org is not a changed id — it was in no earlier table."""
        root = _prepare_tree(tmp_path)
        _run(root)
        _clone_org_as(root, "acme", "aaa_corp")
        caplog.clear()
        with caplog.at_level("WARNING"):
            _run(root)
        assert not self._public_id_warnings(caplog)

    def test_withholding_a_row_does_not_trip_the_guard(self, tmp_path, caplog):
        """A row leaving the table keeps its id reserved; nothing changed."""
        root = _prepare_tree(tmp_path)
        _run(root)
        _withhold_beta_row(root)
        caplog.clear()
        with caplog.at_level("WARNING"):
            _run(root)
        assert not self._public_id_warnings(caplog)

    def test_ambiguous_display_keys_are_excluded_not_guessed(self, tmp_path):
        """The published table is matched on its display columns, which are
        not guaranteed unique — two kv_cache performance profiles for one
        system share all five. Those rows are dropped from the comparison and
        counted, rather than being matched arbitrarily and reported as a
        changed id that never changed."""
        root = _prepare_tree(tmp_path)
        gen = _run(root)
        table = tmp_path / "published.csv"
        table.write_text(
            "Public ID,Organization,Division,Benchmark Type,Model,Name\n"
            "v3.0-0001,acme,CLOSED,kv_cache,,system-a\n"   # same 5 display cols
            "v3.0-0002,acme,CLOSED,kv_cache,,system-a\n"   # as the row above
            "v3.0-0003,beta_corp,CLOSED,training,unet3d,system-b\n"
        )
        published, ambiguous = gen._published_public_ids(str(table))
        assert ambiguous == 1, "the duplicated display key must be counted"
        assert len(published) == 1, (
            "only the unambiguous row may be used for comparison, got "
            f"{published}"
        )


class TestSeeding:
    def test_empty_registry_is_seeded_from_the_first_run(self, tmp_path):
        root = _prepare_tree(tmp_path)
        _seed_empty_registry(root)
        _run(root)
        assert _ids_by_org(root) == {
            "acme": "v3.0-0001",
            "beta_corp": "v3.0-0002",
        }
        reg = _registry(root)
        assert [a["public_id"] for a in reg["assignments"]] == [
            "v3.0-0001", "v3.0-0002",
        ]
        assert reg["next_index"] == 3

    def test_seeded_assignments_carry_the_workload_identity(self, tmp_path):
        root = _prepare_tree(tmp_path)
        _seed_empty_registry(root)
        _run(root)
        by_id = {a["public_id"]: a for a in _registry(root)["assignments"]}
        acme = by_id["v3.0-0001"]
        assert acme["category"] == "closed"
        assert acme["organization"] == "acme"
        assert acme["system"] == "system-a"
        assert acme["benchmark_type"] == "training"
        assert acme["key"], "identity key must be recorded, not just the display fields"


class TestStickiness:
    def test_new_org_sorting_first_does_not_renumber(self, tmp_path):
        """The whole point: aaa_corp sorts first but takes the NEXT number."""
        root = _prepare_tree(tmp_path)
        _seed_empty_registry(root)
        _run(root)
        _clone_org_as(root, "acme", "aaa_corp")
        _run(root)
        assert _ids_by_org(root) == {
            "aaa_corp": "v3.0-0003",
            "acme": "v3.0-0001",
            "beta_corp": "v3.0-0002",
        }

    def test_rerunning_changes_nothing(self, tmp_path):
        root = _prepare_tree(tmp_path)
        _seed_empty_registry(root)
        _run(root)
        first, reg_first = _ids_by_org(root), _registry(root)
        _run(root)
        assert _ids_by_org(root) == first
        assert _registry(root) == reg_first

    def test_pinned_id_agrees_across_every_table(self, tmp_path):
        """A pinned ID must reach the per-model and per-org tables too."""
        root = _prepare_tree(tmp_path)
        _seed_empty_registry(root)
        _run(root)
        _clone_org_as(root, "acme", "aaa_corp")
        _run(root)
        for org, system, expected in (
            ("acme", "system-a", "v3.0-0001"),
            ("aaa_corp", "system-a", "v3.0-0003"),
        ):
            per_model = json.loads(
                (root / "closed" / org / "results" / system / "training"
                 / "unet3d" / "run" / "results.json").read_text()
            )
            org_rollup = json.loads(
                (root / "closed" / org / "results" / "results.json").read_text()
            )
            assert [r["Public ID"] for r in per_model] == [expected]
            assert [r["Public ID"] for r in org_rollup] == [expected]


class TestIdsAreNeverReused:
    def test_withheld_row_leaves_its_id_reserved_and_gaps_the_table(self, tmp_path):
        root = _prepare_tree(tmp_path)
        _seed_empty_registry(root)
        _run(root)
        assert _ids_by_org(root)["beta_corp"] == "v3.0-0002"

        _withhold_beta_row(root)
        _run(root)
        assert _ids_by_org(root) == {"acme": "v3.0-0001"}, (
            "beta_corp's row is gone; acme must keep 0001"
        )
        reserved = {a["public_id"] for a in _registry(root)["assignments"]}
        assert "v3.0-0002" in reserved, (
            "a retired ID must stay in the registry so it is never reused"
        )

    def test_a_later_row_mints_above_the_gap_not_into_it(self, tmp_path):
        root = _prepare_tree(tmp_path)
        _seed_empty_registry(root)
        _run(root)
        _withhold_beta_row(root)
        _run(root)
        _clone_org_as(root, "acme", "aaa_corp")
        _run(root)
        assert _ids_by_org(root) == {
            "aaa_corp": "v3.0-0003",
            "acme": "v3.0-0001",
        }, "0002 is retired — reusing it would redirect every old citation"

    def test_restoring_a_withheld_row_restores_its_original_id(self, tmp_path):
        """The submitter fixes the data; the row comes back as itself."""
        root = _prepare_tree(tmp_path)
        _seed_empty_registry(root)
        _run(root)
        stash = _withhold_beta_row(root)
        _run(root)
        assert "beta_corp" not in _ids_by_org(root)
        _restore_beta_row(root, stash)
        _run(root)
        assert _ids_by_org(root)["beta_corp"] == "v3.0-0002"


class TestPinnedIdsHonorAHandWrittenRegistry:
    def test_preassigned_ids_are_used_verbatim(self, tmp_path):
        """The migration case: a registry authored from an older numbering
        must override what a positional pass would produce."""
        root = _prepare_tree(tmp_path)
        _seed_empty_registry(root)
        _run(root)
        reg = _registry(root)
        # Swap the two orgs' numbers and push them out of the contiguous range.
        remap = {"v3.0-0001": "v3.0-0042", "v3.0-0002": "v3.0-0007"}
        for a in reg["assignments"]:
            a["public_id"] = remap[a["public_id"]]
        reg["next_index"] = 43
        (root / _REGISTRY_NAME).write_text(json.dumps(reg, indent=2))

        _run(root)
        assert _ids_by_org(root) == {
            "acme": "v3.0-0042",
            "beta_corp": "v3.0-0007",
        }

    def test_next_index_is_repaired_when_it_lags_the_highest_id(self, tmp_path):
        """A hand-edited registry must not be able to mint a duplicate."""
        root = _prepare_tree(tmp_path)
        _seed_empty_registry(root)
        _run(root)
        reg = _registry(root)
        for a in reg["assignments"]:
            if a["public_id"] == "v3.0-0002":
                a["public_id"] = "v3.0-0050"
        reg["next_index"] = 2  # stale / wrong on purpose
        (root / _REGISTRY_NAME).write_text(json.dumps(reg, indent=2))

        _clone_org_as(root, "acme", "aaa_corp")
        _run(root)
        ids = set(_ids_by_org(root).values())
        assert len(ids) == 3, f"minted a duplicate ID: {sorted(ids)}"
        assert _ids_by_org(root)["aaa_corp"] == "v3.0-0051"


class TestAmbiguousIdentityIsReported:
    def test_two_rows_with_one_identity_stay_uniquely_numbered(
        self, tmp_path, caplog,
    ):
        """The registry cannot tell two identical identities apart. Say so
        out loud and keep the IDs unique rather than issuing one twice."""
        root = _prepare_tree(tmp_path)
        _seed_empty_registry(root)
        gen = _run(root)

        shared = ("closed", "acme", "system-a", "unet3d", "h100")
        rows = [
            {"__workload_key__": shared, "benchmark_type": "training",
             "category": "closed", "orgname": "acme", "systemname": "system-a",
             "model": "unet3d", "accelerator": "h100"},
            {"__workload_key__": shared, "benchmark_type": "training",
             "category": "closed", "orgname": "acme", "systemname": "system-a",
             "model": "unet3d", "accelerator": "h100"},
        ]
        caplog.clear()
        with caplog.at_level("WARNING"):
            gen._assign_public_ids(rows)

        ids = [r["sut_public_id"] for r in rows]
        assert len(set(ids)) == 2, f"the same ID was issued twice: {ids}"
        assert any(
            "ambiguous identity" in rec.getMessage().lower()
            for rec in caplog.records
        ), (
            "an ambiguous identity must be reported by _assign_public_ids "
            f"itself, not silently resolved; got {[r.getMessage() for r in caplog.records]}"
        )
