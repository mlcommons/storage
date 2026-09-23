"""``mlpstorage submit`` (status-and-submit PR 3).

Design (2026-09-23, decision 4): ``submit`` is the submitter's verb over
their own initialized results-dir -- the same checker ``validate`` runs, but
with the refusal rendered as the ``status`` table, and on a pass a package
(tarball + manifest + checksum), a ledger entry under ``.mlps/`` and the
manual upload instructions. ``--dry-run`` does everything but write the
package and the ledger line. Rollup tables (``reports reportgen``) are
regenerated first in both modes, because the rules about them (2.1.16,
2.1.22, RPT-01, PROV-02) can only be judged on fresh files.

Covered here:
- the parser: ``submit`` is a top-level leaf with ``--results-dir``,
  ``--dry-run`` and ``--out``;
- refusal: exit 1 and the status table whenever any error remains (short,
  invalid, paperwork, tree problems, or a checker error the table does not
  carry); warnings never block; whatif never counts; nothing packaged;
- the package: ``<org>/{closed,open}/<org>/**`` and only the code images
  the packaged leaves point at, under ``<org>/code-images/``; never
  ``whatif/``, ``.mlps/`` or the sentinel; sha256 sidecar; manifest with a
  per-file inventory; ``.mlps/submissions.jsonl`` ledger; ``--out``;
- ``--dry-run``: same check, says what it would write, writes nothing;
- ``status``'s "Next:" line now ends in ``submit --dry-run`` when all ready;
- ``readiness.evaluate`` accepting pre-collected findings and the
  ``errors`` list it now carries;
- ``--help_all`` tree, block, synopsis and context tokens.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
from unittest.mock import patch

import pytest

from mlpstorage_py.config import EXIT_CODE

from tests.unit.test_readiness import (  # noqa: F401  (fixtures by import)
    HASH_A,
    ORG,
    SYS,
    _leaf,
    _system,
    _training,
    _ts,
    rd,
    scripted,
    xdg,
)
from tests.unit.test_runs_management import tree  # noqa: F401  (fixture by import)

HASH_B = "b" * 32
HASH_C = "c" * 32
PACKAGE_NAME_RE = re.compile(rf"^{ORG}-3\.0-\d{{8}}_\d{{6}}\.tar\.gz$")


def _main(argv):
    from mlpstorage_py import main as main_mod

    with patch("sys.argv", ["mlpstorage"] + list(argv)), \
         patch.object(main_mod, "apply_logging_options"):
        return main_mod.main()


def _lines(text: str):
    return [line.rstrip() for line in text.splitlines()]


def _global_image(rd, full_hash, *, marker=True):
    """One image in the tree-wide ``code-images/`` pool."""
    pool = os.path.join(rd, "code-images")
    os.makedirs(pool, exist_ok=True)
    if marker:
        with open(os.path.join(pool, ".mlps-image-pool"), "w") as fh:
            fh.write("mlpstorage_version=3.0.46\nmigration_completed_at=2026-09-01T00:00:00Z\n")
    image = os.path.join(pool, f"code-{full_hash[:8]}")
    os.makedirs(os.path.join(image, "mlpstorage_py"), exist_ok=True)
    with open(os.path.join(image, ".code-hash.json"), "w") as fh:
        json.dump({"hash": full_hash, "algorithm": "md5-tree-v2"}, fh)
    with open(os.path.join(image, "mlpstorage_py", "main.py"), "w") as fh:
        fh.write(f"# image {full_hash[:8]}\n")
    return image


def _legacy_image(rd, full_hash, org=ORG):
    """One image in the v3.0 per-organization pool ``<rd>/<org>/``."""
    pool = os.path.join(rd, org)
    os.makedirs(pool, exist_ok=True)
    with open(os.path.join(pool, ".mlps-image-pool"), "w") as fh:
        fh.write("version=1\ncreated=2026-09-01T00:00:00Z\n")
    image = os.path.join(pool, f"code-{full_hash[:8]}")
    os.makedirs(image, exist_ok=True)
    with open(os.path.join(image, ".code-hash.json"), "w") as fh:
        json.dump({"hash": full_hash, "algorithm": "md5-tree-v2"}, fh)
    with open(os.path.join(image, "setup.py"), "w") as fh:
        fh.write("# legacy image\n")
    return image


@pytest.fixture
def reportgen(monkeypatch):
    """Replace the rollup regeneration with a recorder. Returns the list of
    calls; set ``calls.fail`` to make it raise."""
    import mlpstorage_py.submit as submit

    class Calls(list):
        fail = False

    calls = Calls()

    def fake(results_dir, logger):
        calls.append(("reportgen", results_dir))
        if calls.fail:
            raise RuntimeError("reportgen exploded")
        return []

    monkeypatch.setattr(submit, "regenerate_rollups", fake)
    return calls


@pytest.fixture
def ready(rd, scripted, reportgen):
    """closed/sys-1 unet3d: six ok runs, complete paperwork, one global pool
    image the runs point at; the checker has nothing to say."""
    _training(rd, "unet3d", 6)
    _system(rd, "closed", SYS)
    _global_image(rd, HASH_A)
    scripted([])
    return rd


def _packages(rd):
    d = os.path.join(rd, ".mlps", "packages")
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


def _tarball(rd):
    names = [n for n in _packages(rd) if n.endswith(".tar.gz")]
    assert len(names) == 1, names
    return os.path.join(rd, ".mlps", "packages", names[0])


def _members(path):
    with tarfile.open(path, "r:gz") as tar:
        return {m.name: m for m in tar.getmembers()}


def _ledger(rd):
    path = os.path.join(rd, ".mlps", "submissions.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _systems_finding(rd, division, name, loc):
    return ("error", "2.1.7", "systemYamlValid",
            os.path.join(rd, division, ORG, "systems", f"{name}.yaml"), f"{loc}: Field required")


# ===========================================================================
# Parser and help
# ===========================================================================

class TestSubmitParser:
    def _leaf(self):
        import argparse
        from mlpstorage_py.cli_parser import build_parser
        parser = build_parser()
        top = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        return top.choices["submit"]

    def test_submit_is_a_top_level_leaf(self):
        import argparse
        sub = self._leaf()
        assert not any(isinstance(a, argparse._SubParsersAction) for a in sub._actions)

    def test_flags(self):
        sub = self._leaf()
        longs = {o for a in sub._actions for o in a.option_strings if o.startswith("--")}
        assert {"--results-dir", "--dry-run", "--out"} <= longs
        shorts = {o for a in sub._actions for o in a.option_strings if not o.startswith("--")}
        assert "-rd" in shorts

    def test_defaults(self):
        ns = self._leaf().parse_args([])
        assert ns.dry_run is False and ns.out is None


class TestHelpAll:
    def test_block_tree_entry_and_synopsis(self):
        from mlpstorage_py.cli.help_formatter import HELP_ALL_TEXT
        assert "\nSUBMIT\n" in HELP_ALL_TEXT
        assert "├── submit" in HELP_ALL_TEXT
        assert "mlpstorage submit [--dry-run] [--out PATH]" in HELP_ALL_TEXT

    def test_parity_maps_submit_to_its_block(self):
        from tests.unit.test_help_all_parity import _block_for
        assert _block_for(("submit",)) == "SUBMIT"

    def test_context_tokens(self):
        from mlpstorage_py.cli.help_formatter import get_context_help_tokens
        assert "| submit |" in get_context_help_tokens([])
        assert get_context_help_tokens(["submit"]) is None


# ===========================================================================
# Refusal
# ===========================================================================

class TestRefusal:
    @pytest.fixture
    def short_and_paperwork(self, rd, scripted, reportgen):
        _training(rd, "unet3d", 6)
        _training(rd, "retinanet", 2)
        _system(rd, "closed", SYS, pdf=False)
        _global_image(rd, HASH_A)
        scripted([_systems_finding(rd, "closed", SYS, "system_under_test -> vendor")])
        return rd

    @pytest.mark.parametrize("extra", [[], ["--dry-run"]])
    def test_exit_one_with_the_status_table(self, short_and_paperwork, capsys, extra):
        rc = _main(["submit", "-rd", short_and_paperwork] + extra)
        assert rc == EXIT_CODE.GENERAL_ERROR
        out = capsys.readouterr().out
        rows = [l for l in _lines(out) if l.startswith(SYS)]
        assert any("retinanet" in r and "2/6" in r and "short" in r for r in rows)
        assert any("unet3d" in r and "6/6" in r and "paperwork" in r for r in rows)
        assert any(l.startswith("Paperwork for") for l in _lines(out))
        assert any(l.startswith("Not submittable") for l in _lines(out))
        assert _lines(out)[-1].startswith("Next:") and "submit --dry-run" in _lines(out)[-1]
        assert _packages(short_and_paperwork) == []
        assert _ledger(short_and_paperwork) == []

    def test_checker_error_outside_the_table_blocks(self, ready, scripted, capsys):
        run_dir = os.path.join(ready, "closed", ORG, "results", SYS, "training", "unet3d", "run")
        scripted([("error", "2.1.16", "runResultsJson", run_dir, "results.json is missing")])
        rc = _main(["submit", "--dry-run", "-rd", ready])
        assert rc == EXIT_CODE.GENERAL_ERROR
        out = capsys.readouterr().out
        assert "[2.1.16]" in out and "results.json is missing" in out
        assert any(l.startswith("Not submittable") for l in _lines(out))

    def test_whatif_errors_never_block(self, ready, scripted, capsys):
        _training(ready, "unet3d", 1, division="whatif")
        whatif_leaf = os.path.join(ready, "whatif", ORG, "results", SYS, "training", "unet3d", "run", _ts(0))
        scripted([("error", "2.1.19", "runLeafFiles", whatif_leaf, "missing 0_output.json")])
        assert _main(["submit", "--dry-run", "-rd", ready]) == EXIT_CODE.SUCCESS
        out = capsys.readouterr().out
        assert "never packaged" in out

    def test_warnings_never_block(self, ready, scripted, capsys):
        leaf = os.path.join(ready, "closed", ORG, "results", SYS, "training", "unet3d", "run", _ts(0))
        scripted([("warning", "EDN-03", "dlioRevision", leaf, "DLIO commit not in the accepted list")])
        assert _main(["submit", "--dry-run", "-rd", ready]) == EXIT_CODE.SUCCESS
        assert "1 warning" in capsys.readouterr().out

    def test_whatif_only_tree_is_refused(self, rd, scripted, reportgen, capsys):
        _training(rd, "unet3d", 6, division="whatif")
        scripted([])
        assert _main(["submit", "-rd", rd]) == EXIT_CODE.GENERAL_ERROR
        assert "Nothing to submit" in capsys.readouterr().out
        assert _packages(rd) == []

    def test_empty_tree_is_refused(self, rd, scripted, reportgen, capsys):
        scripted([])
        assert _main(["submit", "-rd", rd]) == EXIT_CODE.GENERAL_ERROR
        assert "Nothing to submit" in capsys.readouterr().out

    def test_reportgen_failure_blocks(self, ready, reportgen, capsys, caplog):
        import logging
        reportgen.fail = True
        with caplog.at_level(logging.ERROR, logger="MLPerfStorage"):
            rc = _main(["submit", "-rd", ready])
        assert rc == EXIT_CODE.GENERAL_ERROR
        text = " ".join(r.getMessage() for r in caplog.records) + capsys.readouterr().out
        assert "rollup" in text.lower() and "reportgen exploded" in text
        assert _packages(ready) == []

    def test_reportgen_runs_before_the_check(self, ready, monkeypatch):
        import mlpstorage_py.readiness as readiness
        import mlpstorage_py.submit as submit
        order = []
        original = readiness.collect_findings

        def spy(results_dir):
            order.append("checker")
            return original(results_dir)

        monkeypatch.setattr(readiness, "collect_findings", spy)
        monkeypatch.setattr(submit, "regenerate_rollups",
                            lambda results_dir, logger: order.append("reportgen") or [])
        _main(["submit", "--dry-run", "-rd", ready])
        assert order == ["reportgen", "checker"]


# ===========================================================================
# Dry run
# ===========================================================================

class TestDryRun:
    def test_writes_nothing_and_says_what_it_would(self, ready, capsys):
        assert _main(["submit", "--dry-run", "-rd", ready]) == EXIT_CODE.SUCCESS
        out = capsys.readouterr().out
        assert "1 of 1 result ready" in out
        dry = [l for l in _lines(out) if l.startswith("Dry run")]
        assert len(dry) == 1
        assert "no package written" in dry[0].lower()
        assert "mlpstorage submit" in out
        would = [l for l in _lines(out) if "would write" in l.lower()]
        assert would and ".mlps/packages/" in would[0] and PACKAGE_NAME_RE.match(
            would[0].rsplit("/", 1)[1].split()[0])
        assert f"closed/{ORG}" in out and "code-images" in out and "1 image" in out
        assert _packages(ready) == []
        assert _ledger(ready) == []
        assert not os.path.exists(os.path.join(ready, ".mlps", "packages"))


# ===========================================================================
# The package
# ===========================================================================

class TestPackage:
    def test_tarball_layout(self, ready):
        assert _main(["submit", "-rd", ready]) == EXIT_CODE.SUCCESS
        path = _tarball(ready)
        assert PACKAGE_NAME_RE.match(os.path.basename(path))
        members = _members(path)
        assert all(n == ORG or n.startswith(f"{ORG}/") for n in members)
        assert f"{ORG}/closed/{ORG}/systems/{SYS}.yaml" in members
        assert f"{ORG}/closed/{ORG}/systems/{SYS}.pdf" in members
        leaf = f"{ORG}/closed/{ORG}/results/{SYS}/training/unet3d/run/{_ts(0)}"
        assert f"{leaf}/summary.json" in members
        assert f"{leaf}/.mlps-code-image" in members
        assert f"{ORG}/code-images/.mlps-image-pool" in members
        assert f"{ORG}/code-images/code-aaaaaaaa/.code-hash.json" in members
        assert f"{ORG}/code-images/code-aaaaaaaa/mlpstorage_py/main.py" in members
        assert not any("/.mlps/" in n or n.endswith("mlperf-results.yaml") or "/whatif/" in n
                       for n in members)
        assert members[f"{leaf}/summary.json"].isfile()

    def test_whatif_and_unreferenced_images_are_left_out(self, ready):
        _training(ready, "unet3d", 1, division="whatif", pointer=HASH_B)
        _global_image(ready, HASH_B)   # only whatif points at it
        _global_image(ready, HASH_C)   # nobody points at it
        assert _main(["submit", "-rd", ready]) == EXIT_CODE.SUCCESS
        members = _members(_tarball(ready))
        assert any("code-aaaaaaaa" in n for n in members)
        assert not any("code-bbbbbbbb" in n or "code-cccccccc" in n or "/whatif/" in n
                       for n in members)

    def test_legacy_per_org_pool_image_lands_under_code_images(self, rd, scripted, reportgen):
        _training(rd, "unet3d", 6)
        _system(rd, "closed", SYS)
        _legacy_image(rd, HASH_A)
        scripted([])
        assert _main(["submit", "-rd", rd]) == EXIT_CODE.SUCCESS
        members = _members(_tarball(rd))
        assert f"{ORG}/code-images/code-aaaaaaaa/setup.py" in members
        assert f"{ORG}/code-images/.mlps-image-pool" in members
        assert not any(n.startswith(f"{ORG}/{ORG}/") for n in members)

    def test_open_division_is_packaged_too(self, ready):
        _training(ready, "unet3d", 6, division="open")
        _system(ready, "open", SYS)
        assert _main(["submit", "-rd", ready]) == EXIT_CODE.SUCCESS
        members = _members(_tarball(ready))
        assert f"{ORG}/open/{ORG}/systems/{SYS}.yaml" in members
        assert f"{ORG}/open/{ORG}/results/{SYS}/training/unet3d/run/{_ts(0)}/summary.json" in members

    def test_checksum_and_manifest_sidecars(self, ready):
        assert _main(["submit", "-rd", ready]) == EXIT_CODE.SUCCESS
        path = _tarball(ready)
        stem = path[: -len(".tar.gz")]
        digest = _sha256(path)
        with open(stem + ".sha256") as fh:
            assert fh.read() == f"{digest}  {os.path.basename(path)}\n"
        with open(stem + ".manifest.json") as fh:
            manifest = json.load(fh)
        assert manifest["schema"] == "mlps-submission-package/1"
        assert manifest["orgname"] == ORG
        assert manifest["rules_edition"] == "3.0"
        assert manifest["package"] == os.path.basename(path)
        assert manifest["sha256"] == digest
        assert manifest["size_bytes"] == os.path.getsize(path)
        assert manifest["divisions"] == ["closed"]
        assert manifest["systems"] == {"closed": [SYS]}
        assert manifest["code_images"] == ["code-aaaaaaaa"]
        (result,) = manifest["results"]
        assert {k: result[k] for k in ("division", "systemname", "benchmark", "model", "accelerator", "runs")} == {
            "division": "closed", "systemname": SYS, "benchmark": "training", "model": "unet3d",
            "accelerator": "b200", "runs": 6}
        assert len(result["run_ids"]) == 6 and result["run_ids"] == sorted(result["run_ids"])
        assert len(result["leaves"]) == 6 and all(l.startswith(f"closed/{ORG}/results/") for l in result["leaves"])
        files = {f["path"]: f for f in manifest["files"]}
        regular = [n for n, m in _members(path).items() if m.isfile()]
        assert manifest["file_count"] == len(regular) == len(files)
        yaml_entry = files[f"{ORG}/closed/{ORG}/systems/{SYS}.yaml"]
        src = os.path.join(ready, "closed", ORG, "systems", f"{SYS}.yaml")
        assert yaml_entry["size"] == os.path.getsize(src)
        assert yaml_entry["sha256"] == _sha256(src)

    def test_ledger_records_each_package(self, ready):
        assert _main(["submit", "-rd", ready]) == EXIT_CODE.SUCCESS
        assert _main(["submit", "-rd", ready, "--out", os.path.join(ready, "elsewhere")]) == EXIT_CODE.SUCCESS
        entries = _ledger(ready)
        assert [e["id"] for e in entries] == [1, 2]
        assert all(e["event"] == "packaged" for e in entries)
        first = entries[0]
        assert os.path.isabs(first["package"]) and first["package"] == _tarball(ready)
        assert first["sha256"] == _sha256(first["package"])
        assert first["size_bytes"] == os.path.getsize(first["package"])
        assert first["rules_edition"] == "3.0" and first["orgname"] == ORG
        assert first["results"] == 1 and first["runs"] == 6 and first["file_count"] > 0
        assert first["at"]
        assert entries[1]["package"].startswith(os.path.join(ready, "elsewhere") + os.sep)

    def test_out_directory(self, ready, tmp_path):
        out = tmp_path / "pkgs"
        assert _main(["submit", "-rd", ready, "--out", str(out)]) == EXIT_CODE.SUCCESS
        names = sorted(os.listdir(out))
        tarballs = [n for n in names if n.endswith(".tar.gz")]
        assert len(tarballs) == 1 and PACKAGE_NAME_RE.match(tarballs[0])
        stem = tarballs[0][: -len(".tar.gz")]
        assert names == sorted([tarballs[0], stem + ".manifest.json", stem + ".sha256"])
        assert _packages(ready) == []

    def test_out_file(self, ready, tmp_path):
        out = tmp_path / "pkgs" / "acme-final.tar.gz"
        assert _main(["submit", "-rd", ready, "--out", str(out)]) == EXIT_CODE.SUCCESS
        assert out.is_file()
        assert (tmp_path / "pkgs" / "acme-final.manifest.json").is_file()
        assert (tmp_path / "pkgs" / "acme-final.sha256").is_file()
        assert _ledger(ready)[0]["package"] == str(out)

    def test_output_names_the_files_and_the_upload_step(self, ready, capsys):
        assert _main(["submit", "-rd", ready]) == EXIT_CODE.SUCCESS
        out = capsys.readouterr().out
        lines = _lines(out)
        path = _tarball(ready)
        assert any(l.startswith("Package:") and path in l for l in lines)
        assert any(l.startswith("Checksum:") and path[: -len(".tar.gz")] + ".sha256" in l for l in lines)
        assert any(l.startswith("Manifest:") and path[: -len(".tar.gz")] + ".manifest.json" in l
                   for l in lines)
        assert any("Recorded as submission 1" in l and ".mlps/submissions.jsonl" in l for l in lines)
        assert "MLCommons" in out and "upload" in out.lower() and "replaces" in out
        assert "1 of 1 result ready" in out

    def test_recorded_in_history(self, ready):
        from mlpstorage_py.history import history_file_for
        _main(["submit", "--dry-run", "-rd", ready])
        with open(history_file_for(ready)) as fh:
            assert "submit" in fh.read()


# ===========================================================================
# readiness hooks
# ===========================================================================

class TestReadinessHooks:
    def test_evaluate_accepts_precollected_findings(self, rd, monkeypatch):
        import mlpstorage_py.readiness as readiness
        _training(rd, "unet3d", 6)
        _system(rd, "closed", SYS, pdf=False)

        def boom(_rd):
            raise AssertionError("collect_findings must not run when findings are given")

        monkeypatch.setattr(readiness, "collect_findings", boom)
        finding = readiness.Finding("error", "2.1.8", "systemPdfPresent",
                                    os.path.join(rd, "closed", ORG, "systems", f"{SYS}.yaml"),
                                    f"closed/{ORG}/systems/{SYS}.yaml has no matching {SYS}.pdf")
        sub = readiness.evaluate(rd, findings=[finding])
        assert sub.results[0].submit == "paperwork"
        assert sub.errors == [finding]
        assert not sub.submittable

    def test_errors_exclude_whatif_and_warnings(self, rd, scripted):
        import mlpstorage_py.readiness as readiness
        _training(rd, "unet3d", 6)
        _training(rd, "unet3d", 1, division="whatif")
        _system(rd, "closed", SYS)
        whatif_leaf = os.path.join(rd, "whatif", ORG, "results", SYS, "training", "unet3d", "run", _ts(0))
        run_dir = os.path.join(rd, "closed", ORG, "results", SYS, "training", "unet3d", "run")
        scripted([
            ("error", "2.1.19", "runLeafFiles", whatif_leaf, "missing 0_output.json"),
            ("warning", "EDN-03", "dlioRevision", run_dir, "old DLIO"),
            ("error", "2.1.16", "runResultsJson", run_dir, "results.json is missing"),
        ])
        sub = readiness.evaluate(rd)
        assert [f.rule_id for f in sub.errors] == ["2.1.16"]
        assert [f.rule_id for f in sub.warnings] == ["EDN-03"]
        assert sub.results[0].submit == "ready"   # the table carries no rollup rule
        assert not sub.submittable                 # but the package is not clean
        assert sub.to_dict()["checker_errors"] == 1


    def test_repeated_workload_findings_collapse_in_the_note(self, rd, scripted):
        import mlpstorage_py.readiness as readiness
        _training(rd, "unet3d", 6)
        _system(rd, "closed", SYS)
        workload = os.path.join(rd, "closed", ORG, "results", SYS, "training", "unet3d")
        scripted([("error", "3.1.1", "datasetParams", workload, "dataset parameters not found")] * 6
                 + [("error", "3.1.2", "recordLength", workload, "record length is 0")])
        note = readiness.evaluate(rd).results[0].note
        assert note.count("[3.1.1]") == 1 and "(x6)" in note
        assert "[3.1.2] record length is 0" in note and "(x1)" not in note


# ===========================================================================
# status: the Next line
# ===========================================================================

class TestStatusNextLine:
    def test_all_ready_points_at_submit_dry_run(self, rd, scripted, capsys):
        _training(rd, "unet3d", 6)
        _system(rd, "closed", SYS)
        scripted([])
        _main(["status", "-rd", rd])
        nxt = [l for l in _lines(capsys.readouterr().out) if l.startswith("Next:")][0]
        assert "mlpstorage submit --dry-run" in nxt
        assert "validate" not in nxt


# ===========================================================================
# main gates
# ===========================================================================

class TestSubmitGates:
    def test_uses_recorded_default_results_dir(self, tree, scripted, reportgen, capsys):
        scripted([])
        rc = _main(["submit", "--dry-run"])
        assert rc == EXIT_CODE.GENERAL_ERROR   # that tree has a failed run: refused, not crashed
        assert f"closed/{ORG}" in capsys.readouterr().out

    def test_uninitialized_results_dir_is_refused(self, tmp_path, xdg, caplog):
        import logging
        bare = tmp_path / "bare"
        bare.mkdir()
        with caplog.at_level(logging.ERROR, logger="MLPerfStorage"):
            rc = _main(["submit", "-rd", str(bare)])
        assert rc != EXIT_CODE.SUCCESS
        assert "has not been initialized" in " ".join(r.getMessage() for r in caplog.records)
        assert not (bare / ".mlps").exists()

    def test_no_results_dir_anywhere_is_actionable(self, xdg, caplog):
        import logging
        with caplog.at_level(logging.ERROR, logger="MLPerfStorage"):
            rc = _main(["submit"])
        assert rc != EXIT_CODE.SUCCESS
        assert "mlpstorage init" in " ".join(r.getMessage() for r in caplog.records)


# ===========================================================================
# The real checker and reportgen (no stubs)
# ===========================================================================

class TestRealPipeline:
    def test_dod_tree_is_refused_with_fresh_rollups(self, tmp_path, xdg, capsys):
        """The definition-of-done fixture fails validation (no DLIO logs, no
        pool): ``submit`` must regenerate the rollups, run the real checker,
        print the table and refuse -- and never write a package."""
        from tests.unit.test_readiness import _dod_tree
        rd = _dod_tree(tmp_path)
        rc = _main(["submit", "--dry-run", "-rd", rd])
        assert rc == EXIT_CODE.GENERAL_ERROR
        out = capsys.readouterr().out
        assert any(l.startswith("Not submittable") for l in _lines(out))
        assert "CHECK-01" in out
        assert "SKIPPED RUN DIRECTORIES" not in out   # reportgen's own report stays off stdout
        assert _lines(out)[0].startswith("closed/Acme")
        assert os.path.isfile(os.path.join(rd, "closed", "Acme", "submission.yaml"))
        assert _packages(rd) == []
