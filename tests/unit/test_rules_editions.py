"""Rules editions table + declared comparability classes
(``mlpstorage_py/rules/editions.yaml``, loader ``mlpstorage_py/editions.py``,
checker ``EditionCheck`` EDN-01/02/03; design in
.planning/rules-editions-and-comparability-classes.md).

A comparability class is the WG's hand-asserted statement that runs of one
(family, model, emulated accelerator) whose ``core-config-v1`` hash is one of
the class's listed hashes executed the same workload, in every rules edition
the class lists. Rows in one class compare; nothing else does. The table is
data, edited by PR; the tool only reads it.

Covered here:
- the table loads, is internally consistent and pinned to ``RULES_EDITION``;
- the seed: v3.0 classes equal the shipped workload templates, llama3-8b is
  one class across v2.0 and v3.0, v2.0 CLOSED subset-mode checkpoints are a
  separate class, v1.0/v2.0 spelling drift is one class with two hashes;
- ``classify`` / ``classify_stamp`` / ``accepts_dlio``;
- ``EditionCheck``: rule ids, wiring, silence on derived stamps, EDN-01
  unknown edition (leaf + manifest), EDN-02 unclassified closed run leaf is
  an error / open is info / non-run leaves and non-DLIO families skipped,
  EDN-03 unaccepted DLIO revision is info, accumulation;
- ``mlpstorage runs show`` prints the class;
- Rules.md and ManPage.md document the rules and the table.
"""

from __future__ import annotations

import copy
import inspect
import json
import re
from pathlib import Path

import pytest
import yaml

from mlpstorage_py.config import EXIT_CODE, RULES_EDITION
from mlpstorage_py.provenance import (
    PROVENANCE_FILENAME,
    UNKNOWN,
    RunProvenance,
    compute_core_config,
    load_allowlists,
    write_leaf_provenance,
    write_submission_manifest,
)
from mlpstorage_py.submission_checker.rule_registry import discover_rules

from tests.unit.test_leaf_provenance import (  # noqa: E402
    DLIO_SHA, HASH_A, UNET3D_B200, V2_LLAMA70B, V3_LLAMA70B,
    _Log, _image, _leaf, _org, _stamp,
)
from tests.unit.test_submission_checker_pool_structure import _make_config
from tests.unit.test_runs_management import (  # noqa: E402  (fixtures by import)
    LEAF_TRAIN_RUN, _main, tree, xdg,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

UNET3D_HASH = compute_core_config(UNET3D_B200, "training")["hash"]
LLAMA70B_HASH = compute_core_config(V3_LLAMA70B, "checkpointing")["hash"]

# v2.0 CLOSED llama3-70b (closed/Samsung/.../20250624_155100): subset mode.
V2_LLAMA70B_SUBSET = copy.deepcopy(V2_LLAMA70B)
V2_LLAMA70B_SUBSET["checkpoint"]["mode"] = "subset"
V2_LLAMA70B_SUBSET["model"]["parallelism"]["data"] = 8
LLAMA70B_SUBSET_HASH = compute_core_config(V2_LLAMA70B_SUBSET, "checkpointing")["hash"]

# closed/ZettaLane/.../training/unet3d/run (v3.0 tree): workflow.checkpoint = 'True'
UNET3D_ZETTALANE = copy.deepcopy(UNET3D_B200)
UNET3D_ZETTALANE["workflow"]["checkpoint"] = "True"
UNET3D_ZETTALANE_HASH = compute_core_config(UNET3D_ZETTALANE, "training")["hash"]

LEAF_CLOSED_RUN = "closed/Acme/results/sys-1/training/unet3d/run/20260901_100000"
LEAF_CLOSED_RUN_2 = "closed/Acme/results/sys-1/training/unet3d/run/20260902_100000"
LEAF_CLOSED_DATAGEN = "closed/Acme/results/sys-1/training/unet3d/datagen/20260831_090000"
LEAF_OPEN_RUN = "open/Acme/results/sys-1/training/unet3d/run/20260901_100000"
LEAF_CLOSED_CKPT = "closed/Acme/results/sys-1/checkpointing/llama3-70b/20260903_110000"
LEAF_CLOSED_KV = "closed/Acme/results/sys-1/kv_cache/llama3.1-8b/run/20260905_130000"


def _table():
    from mlpstorage_py.editions import load_editions
    return load_editions()


def _training_stamp(core_hash: str = UNET3D_HASH, **over) -> dict:
    d = _stamp(core_config={"algorithm": "core-config-v1", "allowlist": "training@1",
                            "hash": core_hash, "keys": ["framework"]})
    d.update(over)
    return d


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

class TestTable:
    def test_file_and_schema(self):
        from mlpstorage_py.editions import EDITIONS_FILE, EDITIONS_SCHEMA
        assert EDITIONS_FILE.name == "editions.yaml"
        assert EDITIONS_FILE.parent.name == "rules"
        assert EDITIONS_SCHEMA == "mlps-rules-editions/1"
        assert _table().schema == EDITIONS_SCHEMA

    def test_current_edition_is_pinned_to_the_tool_constant(self):
        t = _table()
        assert t.current_edition == RULES_EDITION
        assert RULES_EDITION in t.editions

    def test_every_round_is_an_edition(self):
        t = _table()
        assert set(t.editions) >= {"0.5", "1.0", "2.0", "3.0"}
        for eid, e in t.editions.items():
            assert e.id == eid
            assert e.status in ("historical", "current")
            assert e.results_repo
        assert t.editions["3.0"].status == "current"
        assert t.editions["2.0"].status == "historical"

    def test_classes_are_internally_consistent(self):
        t = _table()
        allowlists = load_allowlists()["allowlists"]
        ids = [c.id for c in t.classes]
        assert ids == sorted(ids) and len(ids) == len(set(ids)), "class ids sorted and unique"
        for c in t.classes:
            assert re.fullmatch(r"[a-z0-9.]+(-[a-z0-9]+)*-[A-Z]", c.id), c.id
            assert c.family in ("training", "checkpointing"), c.id
            assert c.accelerator in ("b200", "mi355", "h100", "a100", "any"), c.id
            assert c.allowlist in allowlists, c.id
            assert c.editions and set(c.editions) <= set(t.editions), c.id
            assert c.core_configs and all(re.fullmatch(r"[0-9a-f]{16}", h) for h in c.core_configs), c.id
            assert len(set(c.core_configs)) == len(c.core_configs), c.id
        # a hash means one thing: no two classes of one (family, model, accelerator, edition) share it
        seen = {}
        for c in t.classes:
            for e in c.editions:
                for h in c.core_configs:
                    key = (c.family, c.model, c.accelerator, e, h)
                    assert key not in seen, f"{c.id} and {seen[key]} both claim {key}"
                    seen[key] = c.id

    def test_edition_dlio_revisions_are_full_shas(self):
        t = _table()
        for e in t.editions.values():
            for rev in e.dlio_revisions:
                assert re.fullmatch(r"[0-9a-f]{40}", rev["commit"]), (e.id, rev)
                assert rev["source"].startswith("https://"), (e.id, rev)
        assert t.accepts_dlio("3.0", DLIO_SHA)
        assert not t.accepts_dlio("3.0", "f" * 40)
        assert not t.accepts_dlio("3.0", UNKNOWN)
        assert not t.accepts_dlio("9.9", DLIO_SHA)

    def test_edition_workload_vocabulary(self):
        e = _table().editions["3.0"]
        assert e.workloads["closed"]["training"]["unet3d"] == ["b200", "mi355"]
        assert e.workloads["closed"]["training"]["retinanet"] == ["b200", "mi355"]
        assert set(e.workloads["closed"]["checkpointing"]) == {"llama3-8b", "llama3-70b", "llama3-405b", "llama3-1t"}
        assert "kv_cache" in e.workloads["closed"] and "vector_database" in e.workloads["closed"]
        assert e.layout_versions == [2, 3]
        e2 = _table().editions["2.0"]
        assert set(e2.workloads["closed"]["training"]) == {"unet3d", "resnet50", "cosmoflow"}
        assert e2.workloads["closed"]["training"]["unet3d"] == ["a100", "h100"]

    def test_malformed_table_is_rejected(self, tmp_path):
        from mlpstorage_py.editions import EditionsError, load_editions
        bad = tmp_path / "editions.yaml"
        bad.write_text("schema: nope\n")
        with pytest.raises(EditionsError):
            load_editions(bad)
        good = yaml.safe_load(Path(inspect.getmodule(load_editions).EDITIONS_FILE).read_text())
        good["classes"].append(dict(good["classes"][0], id="zzz-dup-A", editions=["9.9"]))
        bad.write_text(yaml.safe_dump(good))
        with pytest.raises(EditionsError, match="9.9"):
            load_editions(bad)


# ---------------------------------------------------------------------------
# The seed
# ---------------------------------------------------------------------------

class TestSeed:
    def test_v30_classes_equal_the_shipped_workload_templates(self):
        """F-5: every class with a `reference` template hashes to that template."""
        t = _table()
        checked = 0
        for c in t.classes:
            if not c.reference:
                continue
            y = yaml.safe_load((PROJECT_ROOT / c.reference).read_text())
            got = compute_core_config(y.get("workload", y), c.family)["hash"]
            assert got in c.core_configs, (c.id, c.reference, got)
            checked += 1
        assert checked >= 8  # unet3d/retinanet x b200 (+mi355 retinanet), 4 llama, resnet50/cosmoflow x2

    def test_v30_unet3d_b200(self):
        c = _table().classify(family="training", model="unet3d", accelerator="b200",
                              core_config=UNET3D_HASH, edition="3.0")
        assert c is not None and c.id == "unet3d-b200-A"
        assert c.editions == ("3.0",)

    def test_v30_zettalane_variant_is_unclassified(self):
        """F-2: workflow.checkpoint='True' is not the sanctioned unet3d workload."""
        assert _table().classify(family="training", model="unet3d", accelerator="b200",
                                 core_config=UNET3D_ZETTALANE_HASH, edition="3.0") is None

    def test_llama3_8b_is_one_class_across_v20_and_v30(self):
        t = _table()
        c = next(c for c in t.classes if c.id == "llama3-8b-A")
        assert c.accelerator == "any"
        assert set(c.editions) == {"2.0", "3.0"}
        assert c.core_configs == ("68df04b7240e3fb0",)

    def test_v20_closed_subset_checkpoints_are_a_separate_class(self):
        """F-1: v2.0 CLOSED 70b/405b/1t ran checkpoint.mode=subset; v3.0 did not."""
        t = _table()
        a = t.classify(family="checkpointing", model="llama3-70b", accelerator="unknown",
                       core_config=LLAMA70B_HASH, edition="3.0")
        b = t.classify(family="checkpointing", model="llama3-70b", accelerator="any",
                       core_config=LLAMA70B_SUBSET_HASH, edition="2.0")
        assert a is not None and a.id == "llama3-70b-A"
        assert b is not None and b.id == "llama3-70b-B" and b.editions == ("2.0",)
        assert t.classify(family="checkpointing", model="llama3-70b", accelerator="b200",
                          core_config=LLAMA70B_SUBSET_HASH, edition="3.0") is None
        # v2.0 OPEN (ANL) ran the full config: same class as v3.0
        assert "2.0" in a.editions
        assert compute_core_config(V2_LLAMA70B, "checkpointing")["hash"] == LLAMA70B_HASH

    def test_v10_and_v20_spelling_drift_is_one_class_with_two_hashes(self):
        """F-4: dataset.record_length -> record_length_bytes etc."""
        t = _table()
        c = next(c for c in t.classes if c.id == "cosmoflow-h100-A")
        assert set(c.editions) == {"1.0", "2.0"}
        assert set(c.core_configs) == {"54541a1d49e87763", "4b7b36339a2a5b0d"}
        for eid, h in (("1.0", "54541a1d49e87763"), ("2.0", "4b7b36339a2a5b0d")):
            got = t.classify(family="training", model="cosmoflow", accelerator="h100",
                             core_config=h, edition=eid)
            assert got is c
        # class-level editions: every listed spelling is valid in every listed edition
        assert t.classify(family="training", model="cosmoflow", accelerator="h100",
                          core_config="54541a1d49e87763", edition="2.0") is c
        assert t.classify(family="training", model="cosmoflow", accelerator="h100",
                          core_config="54541a1d49e87763", edition="3.0") is None


# ---------------------------------------------------------------------------
# classify / classify_stamp
# ---------------------------------------------------------------------------

class TestClassify:
    def test_edition_none_matches_any_edition(self):
        t = _table()
        assert t.classify(family="training", model="unet3d", accelerator="b200",
                          core_config=UNET3D_HASH).id == "unet3d-b200-A"

    def test_unknown_hash_or_edition_never_classifies(self):
        t = _table()
        assert t.classify(family="training", model="unet3d", accelerator="b200",
                          core_config=UNKNOWN, edition="3.0") is None
        assert t.classify(family="training", model="unet3d", accelerator="b200",
                          core_config=UNET3D_HASH, edition=UNKNOWN) is None
        assert t.classify(family="training", model="unet3d", accelerator="mi355",
                          core_config=UNET3D_HASH, edition="3.0") is None

    def test_classify_stamp(self):
        t = _table()
        stamp = RunProvenance.from_dict(_training_stamp())
        assert t.classify_stamp(stamp, family="training", model="unet3d", accelerator="b200").id == "unet3d-b200-A"
        derived = RunProvenance.from_dict(_training_stamp(rules_edition=UNKNOWN))
        assert t.classify_stamp(derived, family="training", model="unet3d", accelerator="b200") is None

    def test_classes_for_edition(self):
        t = _table()
        ids = {c.id for c in t.classes_for("3.0")}
        assert ids == {"unet3d-b200-A", "retinanet-b200-A", "retinanet-mi355-A",
                       "llama3-8b-A", "llama3-70b-A", "llama3-405b-A", "llama3-1t-A"}
        assert t.classes_for("0.5") == []


# ---------------------------------------------------------------------------
# EditionCheck — EDN-01 / EDN-02 / EDN-03
# ---------------------------------------------------------------------------

def _check(root: Path):
    from mlpstorage_py.submission_checker.checks.edition_checks import EditionCheck
    log = _Log()
    return EditionCheck(log=log, config=_make_config(), root_path=str(root)), log


def _rule_lines(lines):
    """Only the locked-format findings; BaseCheck adds a 'Some ... checks failed' summary."""
    return [l for l in lines if l.startswith("[EDN-")]


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "sub"
    _org(root)
    _image(root, HASH_A)
    return root


class TestEditionCheck:
    def test_rule_ids(self):
        from mlpstorage_py.submission_checker.checks.edition_checks import EditionCheck
        found = discover_rules(EditionCheck)
        assert found["EDN-01"][0] == "rulesEdition"
        assert found["EDN-02"][0] == "comparabilityClass"
        assert found["EDN-03"][0] == "dlioRevision"

    def test_wired_after_provenance_check(self):
        from mlpstorage_py.submission_checker import main as checker_main
        src = inspect.getsource(checker_main.run)
        assert "EditionCheck(log, config, args.input)" in src
        assert src.index("ProvenanceCheck(log, config, args.input)") < src.index("EditionCheck(log, config, args.input)")

    def test_pre_stamp_tree_is_silent(self, tmp_path):
        root = _root(tmp_path)
        _leaf(root, LEAF_CLOSED_RUN, parameters=UNET3D_ZETTALANE)   # would be unclassified if stamped
        _leaf(root, LEAF_CLOSED_CKPT, parameters=V3_LLAMA70B)
        check, log = _check(root)
        assert check() is True
        assert log.lines == []

    def test_sanctioned_stamped_tree_is_silent(self, tmp_path):
        root = _root(tmp_path)
        _leaf(root, LEAF_CLOSED_RUN, parameters=UNET3D_B200, stamp=_training_stamp(), declare=True)
        ckpt = _stamp(core_config={"algorithm": "core-config-v1", "allowlist": "checkpointing@1",
                                   "hash": LLAMA70B_HASH, "keys": ["framework"]})
        _leaf(root, LEAF_CLOSED_CKPT, parameters=V3_LLAMA70B, stamp=ckpt, declare=True, accelerator="mi355")
        check, log = _check(root)
        assert check() is True
        assert log.lines == []

    def test_unknown_edition_on_a_leaf(self, tmp_path):
        root = _root(tmp_path)
        _leaf(root, LEAF_CLOSED_RUN, parameters=UNET3D_B200,
              stamp=_training_stamp(rules_edition="9.9"), declare=True)
        check, log = _check(root)
        assert check() is False
        errors = _rule_lines(log.errors)
        assert len(errors) == 1
        assert errors[0].startswith("[EDN-01 rulesEdition] ")
        assert "9.9" in errors[0] and LEAF_CLOSED_RUN in errors[0]
        # EDN-02/03 do not pile on for an edition the table cannot look up
        assert not any("[EDN-02" in l or "[EDN-03" in l for l in log.lines)

    def test_unknown_edition_in_a_manifest(self, tmp_path):
        root = _root(tmp_path)
        _leaf(root, LEAF_CLOSED_RUN, parameters=UNET3D_B200, stamp=_training_stamp(), declare=True)
        path = write_submission_manifest(root, "closed", "Acme")
        data = yaml.safe_load(path.read_text())
        data["rules_edition"] = "9.9"
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        check, log = _check(root)
        assert check() is False
        assert [l for l in log.errors if l.startswith("[EDN-01 rulesEdition] ") and "submission.yaml" in l and "9.9" in l]

    def test_unclassified_closed_run_leaf_is_an_error(self, tmp_path):
        root = _root(tmp_path)
        _leaf(root, LEAF_CLOSED_RUN, parameters=UNET3D_ZETTALANE,
              stamp=_training_stamp(UNET3D_ZETTALANE_HASH), declare=True)
        check, log = _check(root)
        assert check() is False
        errors = _rule_lines(log.errors)
        assert len(errors) == 1
        e = errors[0]
        assert e.startswith("[EDN-02 comparabilityClass] ")
        assert LEAF_CLOSED_RUN in e and UNET3D_ZETTALANE_HASH in e and "3.0" in e
        assert "unet3d" in e and "b200" in e
        assert log.infos == [] and log.warnings == []

    def test_unclassified_open_run_leaf_is_info(self, tmp_path):
        root = _root(tmp_path)
        _org(root, mode="open")
        _leaf(root, LEAF_OPEN_RUN, parameters=UNET3D_ZETTALANE,
              stamp=_training_stamp(UNET3D_ZETTALANE_HASH), declare=True)
        check, log = _check(root)
        assert check() is True
        assert log.errors == []
        assert len(log.infos) == 1 and log.infos[0].startswith("[EDN-02 comparabilityClass] ")

    def test_non_run_leaves_and_non_dlio_families_are_skipped(self, tmp_path):
        root = _root(tmp_path)
        _leaf(root, LEAF_CLOSED_DATAGEN, parameters=UNET3D_ZETTALANE,
              stamp=_training_stamp(UNET3D_ZETTALANE_HASH), declare=True)
        kv = _stamp(core_config={"algorithm": "core-config-v1", "allowlist": UNKNOWN,
                                 "hash": UNKNOWN, "keys": []})
        _leaf(root, LEAF_CLOSED_KV, parameters={}, stamp=kv, declare=True)
        check, log = _check(root)
        assert check() is True
        assert log.lines == []

    def test_unaccepted_dlio_revision_is_info(self, tmp_path):
        root = _root(tmp_path)
        stamp = _training_stamp()
        stamp["dlio"] = dict(stamp["dlio"], commit="f" * 40)
        _leaf(root, LEAF_CLOSED_RUN, parameters=UNET3D_B200, stamp=stamp, declare=True)
        check, log = _check(root)
        assert check() is True
        assert log.errors == [] and log.warnings == []
        assert len(log.infos) == 1
        i = log.infos[0]
        assert i.startswith("[EDN-03 dlioRevision] ") and "ffffffff" in i and "3.0" in i

    def test_unknown_dlio_revision_is_silent(self, tmp_path):
        root = _root(tmp_path)
        stamp = _training_stamp()
        stamp["dlio"] = {"version": UNKNOWN, "source": UNKNOWN, "commit": UNKNOWN}
        _leaf(root, LEAF_CLOSED_RUN, parameters=UNET3D_B200, stamp=stamp, declare=True)
        check, log = _check(root)
        assert check() is True
        assert log.lines == []

    def test_accumulates_every_leaf(self, tmp_path):
        root = _root(tmp_path)
        for rel in (LEAF_CLOSED_RUN, LEAF_CLOSED_RUN_2):
            _leaf(root, rel, parameters=UNET3D_ZETTALANE,
                  stamp=_training_stamp(UNET3D_ZETTALANE_HASH), declare=True)
        check, log = _check(root)
        assert check() is False
        assert len(_rule_lines(log.errors)) == 2


# ---------------------------------------------------------------------------
# mlpstorage runs show
# ---------------------------------------------------------------------------

class TestRunsShow:
    def test_show_prints_the_class(self, tree, capsys):
        write_leaf_provenance(Path(tree) / LEAF_TRAIN_RUN, RunProvenance.from_dict(_training_stamp()))
        assert _main(["runs", "show", "2"]) == EXIT_CODE.SUCCESS
        out = capsys.readouterr().out
        assert re.search(r"class:\s+unet3d-b200-A\b", out)

    def test_show_says_unclassified(self, tree, capsys):
        write_leaf_provenance(Path(tree) / LEAF_TRAIN_RUN,
                              RunProvenance.from_dict(_training_stamp(UNET3D_ZETTALANE_HASH)))
        assert _main(["runs", "show", "2"]) == EXIT_CODE.SUCCESS
        out = capsys.readouterr().out
        assert re.search(r"class:\s+unclassified", out)

    def test_show_matches_a_derived_stamp_by_hash(self, tree, capsys):
        leaf = Path(tree) / LEAF_TRAIN_RUN
        md = next(leaf.glob("*_metadata.json"))
        body = json.loads(md.read_text())
        body["parameters"] = UNET3D_B200
        md.write_text(json.dumps(body, indent=2))
        assert _main(["runs", "show", "2"]) == EXIT_CODE.SUCCESS
        out = capsys.readouterr().out
        assert "provenance:  derived" in out
        assert re.search(r"class:\s+unet3d-b200-A \(matched by hash", out)


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

class TestDocs:
    def test_rules_md_documents_the_rules_and_the_table(self):
        text = (PROJECT_ROOT / "Rules.md").read_text()
        for needle in ("EDN-01 (`rulesEdition`)", "EDN-02 (`comparabilityClass`)",
                       "EDN-03 (`dlioRevision`)", "mlpstorage_py/rules/editions.yaml",
                       "comparability class"):
            assert needle in text, needle

    def test_manpage_documents_the_table_and_runs_show(self):
        text = (PROJECT_ROOT / "ManPage.md").read_text()
        assert "mlpstorage_py/rules/editions.yaml" in text
        assert "EDN-01" in text and "EDN-02" in text and "EDN-03" in text
        assert "comparability class" in text
