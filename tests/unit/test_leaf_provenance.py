"""Leaf provenance stamps + per-submission manifest (PR5 of the results-dir
hygiene effort; schema in .planning/pr5-version-stamps-and-manifest-schema.md).

Every run leaf gets a ``provenance.json`` sidecar that says which rules
edition, tool version, layout version, DLIO revision, storage library and
core-config hash produced it. ``reportgen`` writes one
``<mode>/<org>/submission.yaml`` per organization in a live results-dir.
Leaves written before this change (the whole frozen v3.0 tree) are never
rewritten: the reader derives an in-memory stamp from ``.code-hash.json``,
the image's ``uv.lock`` and the leaf's own metadata instead.

Covered here:
- the three version constants (rules edition, layout version, schema ids);
- ``core-config-v1``: allowlist filtering, canonical numeric normalisation,
  the llama3-70b v2.0 <-> v3.0 test vector, unet3d site tunables ignored,
  emulated accelerator separating classes, non-DLIO families stamp unknown;
- the stamp file: fixed key order, round trip, ``"unknown"`` never null,
  closed provenance vocabulary;
- runtime collection from the installed packages (PEP 610 direct_url);
- ``Benchmark.write_metadata`` writes the sidecar and declares it in
  metadata.json;
- derivation for layout-2 leaves against the global pool, a per-org pool,
  and with no results root given;
- ``submission.yaml``: contents, determinism, the sentinel gate, and the
  reportgen call site;
- PROV-01 / PROV-02 in ``ProvenanceCheck`` (silent on pre-stamp trees);
- ``mlpstorage runs show`` prints the stamp.
"""

from __future__ import annotations

import copy
import json
import os
import re
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from mlpstorage_py import VERSION
from mlpstorage_py.config import EXIT_CODE, RULES_EDITION
from mlpstorage_py.results_dir import MLPERF_RESULTS_VERSION, write_sentinel
from mlpstorage_py import provenance as prov
from mlpstorage_py.provenance import (
    CORE_CONFIG_ALGORITHM,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA,
    METADATA_PROVENANCE_KEY,
    PROVENANCE_FILENAME,
    PROVENANCE_SCHEMA,
    PROVENANCE_TAGS,
    STAMP_FIELDS,
    UNKNOWN,
    ProvenanceError,
    RunProvenance,
    collect_runtime_provenance,
    compute_core_config,
    load_allowlists,
    read_leaf_provenance,
    read_submission_manifest,
    write_leaf_provenance,
    write_manifests_if_results_dir,
    write_submission_manifest,
)
from mlpstorage_py.submission_checker.checks.provenance_checks import ProvenanceCheck
from mlpstorage_py.submission_checker.rule_registry import discover_rules
from mlpstorage_py.submission_checker.tools.code_image import _write_pointer_atomic

from tests.unit.test_submission_checker_pool_structure import _make_config


# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

HASH_A = "a" * 32
HASH_B = "b" * 32
DLIO_SHA = "edaf4bb3d6343ef2fb6b9f6dbafa6102032ae8bf"
GIT_SHA = "a1dcbaa34c210823d7f97325cc4b2fd623f0ac7f"

# closed/Alluxio/.../checkpointing/llama3-70b/20260708_075034 (v3.0 tree)
V3_LLAMA70B = {
    "checkpoint": {"checkpoint_folder": "/mnt/alluxio/ckpt/wf70b2/llama3-70b", "fsync": True,
                   "num_checkpoints_read": 0, "num_checkpoints_write": 10,
                   "time_between_checkpoints": 5},
    "framework": "pytorch",
    "model": {"model_datatype": "fp16", "name": "llama_70b", "num_layers": 80,
              "optimizer_datatype": "fp32",
              "parallelism": {"pipeline": 1, "tensor": 8, "zero_stage": 3},
              "transformer": {"ffn_hidden_size": 28672, "hidden_size": 8192,
                              "num_attention_heads": 128, "num_kv_heads": 8,
                              "vocab_size": 128256},
              "type": "transformer"},
    "workflow": {"checkpoint": True, "generate_data": False, "train": False},
}
# open/ANL/.../checkpointing/llama3-70b/20250628_051907/dlio_config/config.yaml (v2.0 tree)
V2_LLAMA70B = copy.deepcopy(V3_LLAMA70B)
V2_LLAMA70B["checkpoint"].update({
    "checkpoint_folder": ".//checkpoints/n256x8/llama3-70b",
    "num_checkpoints_read": 10, "num_checkpoints_write": 0,
})

# closed/TTA/.../training/unet3d/run/20260724_123243 (v3.0 tree)
UNET3D_B200 = {
    "checkpoint": {"checkpoint_after_epoch": 5, "checkpoint_folder": "checkpoints/unet3d",
                   "epochs_between_checkpoints": 2},
    "dataset": {"data_folder": "/mnt/seahorse/v3train-1007b/unet3d", "format": "npz",
                "listing_validation_interval": 100, "num_files_train": 148000,
                "num_samples_per_file": 1, "num_subfolders_train": 14,
                "record_length_bytes": 146600628, "record_length_bytes_resize": 2097152,
                "record_length_bytes_stdev": 68341808, "skip_listing": True},
    "framework": "pytorch",
    "metric": {"au": 0.9},
    "model": {"model_size": 499153191, "name": "unet3d", "type": "cnn"},
    "reader": {"batch_size": 7, "data_loader": "pytorch", "file_shuffle": "seed",
               "multiprocessing_context": "fork", "odirect": True, "read_threads": 32,
               "sample_shuffle": "seed"},
    "train": {"computation_time": 0.162, "epochs": 5},
    "workflow": {"checkpoint": False, "generate_data": False, "train": True},
}

UV_LOCK = f'''version = 1
requires-python = ">=3.12"

[[package]]
name = "dgen-py"
version = "0.1.0"
source = {{ registry = "https://pypi.org/simple" }}

[[package]]
name = "dlio-benchmark"
version = "3.0.4"
source = {{ git = "https://github.com/mlcommons/DLIO_local_changes.git?rev={DLIO_SHA}#{DLIO_SHA}" }}
dependencies = [
    {{ name = "dgen-py", marker = "sys_platform == 'linux'" }},
]

[[package]]
name = "s3dlio"
version = "0.9.112"
source = {{ registry = "https://pypi.org/simple" }}
dependencies = [
    {{ name = "numpy", marker = "sys_platform == 'linux'" }},
]
'''


class _Log:
    def __init__(self):
        self.errors, self.warnings, self.infos = [], [], []

    def _fmt(self, msg, args):
        return msg % args if args else msg

    def error(self, msg, *args):
        self.errors.append(self._fmt(msg, args))

    def warning(self, msg, *args):
        self.warnings.append(self._fmt(msg, args))

    def info(self, msg, *args):
        self.infos.append(self._fmt(msg, args))

    def debug(self, msg, *args):
        pass

    status = info

    @property
    def lines(self):
        return self.errors + self.warnings + self.infos


def _stamp(**over) -> dict:
    d = {
        "schema": PROVENANCE_SCHEMA,
        "rules_edition": "3.0",
        "layout_version": 3,
        "tool": {"name": "mlpstorage", "version": "3.0.46", "git_sha": GIT_SHA,
                 "code_image": f"md5-tree-v2:{HASH_A}"},
        "dlio": {"version": "3.0.4",
                 "source": "https://github.com/mlcommons/DLIO_local_changes.git",
                 "commit": DLIO_SHA},
        "storage_library": {"name": "s3dlio", "version": "0.9.112"},
        "core_config": {"algorithm": CORE_CONFIG_ALGORITHM, "allowlist": "training@1",
                        "hash": "0123456789abcdef", "keys": ["framework", "model.name"]},
        "stamped_at": "2026-09-19T00:00:00Z",
        "stamped_by": "mlpstorage 3.0.46",
        "provenance": {"rules_edition": "tool-constant", "layout_version": "tool-constant",
                       "tool": "runtime", "dlio": "direct-url",
                       "storage_library": "package-metadata", "core_config": "runtime"},
    }
    d.update(over)
    return d


def _image(root: Path, full_hash: str, *, org: str | None = None, uv_lock: str | None = UV_LOCK,
           git_sha: str | None = GIT_SHA, version: str = "3.0.38") -> Path:
    pool = root / (org if org else "code-images")
    pool.mkdir(parents=True, exist_ok=True)
    (pool / ".mlps-image-pool").write_text("mlpstorage_version=3.0.0\n")
    image = pool / f"code-{full_hash[:8]}"
    image.mkdir(exist_ok=True)
    (image / ".code-hash.json").write_text(json.dumps({
        "hash": full_hash, "algorithm": "md5-tree-v2", "captured_at": "2026-07-17T04:11:34Z",
        "mlpstorage_version": version, "git_sha": git_sha}, indent=2) + "\n")
    (image / "pyproject.toml").write_text("[project]\nname='mlpstorage'\n")
    if uv_lock is not None:
        (image / "uv.lock").write_text(uv_lock)
    return image


def _leaf(root: Path, rel: str, *, parameters: dict | None = None, protocol: str | None = "file",
          pointer: str | None = HASH_A, stamp: dict | None = None, declare: bool = False,
          accelerator: str = "b200") -> Path:
    leaf = root / rel
    leaf.mkdir(parents=True, exist_ok=True)
    ts = rel.rsplit("/", 1)[1]
    family = rel.split("/results/")[1].split("/")[1]
    body = {"benchmark_type": family, "run_datetime": ts, "command": "run",
            "accelerator": accelerator, "parameters": parameters if parameters is not None else {},
            "args": {"data_access_protocol": protocol}, "exit_status": 0}
    if declare:
        body[METADATA_PROVENANCE_KEY] = PROVENANCE_FILENAME
    (leaf / f"{family}_{ts}_metadata.json").write_text(json.dumps(body, indent=2))
    if pointer:
        (leaf / ".mlps-code-image").write_text(f"md5-tree-v2:{pointer}")
    if stamp is not None:
        (leaf / PROVENANCE_FILENAME).write_text(json.dumps(stamp, indent=2) + "\n")
    return leaf


def _org(root: Path, org: str = "Acme", mode: str = "closed", systems=("sys-1",)) -> Path:
    org_dir = root / mode / org
    (org_dir / "results").mkdir(parents=True, exist_ok=True)
    sysd = org_dir / "systems"
    sysd.mkdir(exist_ok=True)
    for s in systems:
        (sysd / f"{s}.yaml").write_text("system_under_test: {}\n")
        (sysd / f"{s}.pdf").write_bytes(b"%PDF-1.4\n")
    return org_dir


LEAF_RUN = "closed/Acme/results/sys-1/training/unet3d/run/20260901_100000"
LEAF_RUN_2 = "closed/Acme/results/sys-1/training/unet3d/run/20260902_100000"
LEAF_CKPT = "closed/Acme/results/sys-1/checkpointing/llama3-70b/20260903_110000"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_rules_edition_is_3_0_until_the_wg_names_the_next_edition(self):
        assert RULES_EDITION == "3.0"

    def test_layout_version_bumped_to_3(self):
        assert MLPERF_RESULTS_VERSION == 3

    def test_file_and_schema_names(self):
        assert PROVENANCE_FILENAME == "provenance.json"
        assert MANIFEST_FILENAME == "submission.yaml"
        assert PROVENANCE_SCHEMA == "mlps-run-provenance/1"
        assert MANIFEST_SCHEMA == "mlps-submission-manifest/1"
        assert CORE_CONFIG_ALGORITHM == "core-config-v1"
        assert UNKNOWN == "unknown"
        assert METADATA_PROVENANCE_KEY == "provenance_file"

    def test_provenance_vocabulary_is_closed(self):
        assert PROVENANCE_TAGS == frozenset({
            "runtime", "tool-constant", "package-metadata", "direct-url",
            "code-hash-json", "uv-lock", "declared", "inferred", "n/a", "unknown"})
        assert STAMP_FIELDS == ("rules_edition", "layout_version", "tool", "dlio",
                                "storage_library", "core_config")


# ---------------------------------------------------------------------------
# core-config-v1
# ---------------------------------------------------------------------------

class TestCoreConfig:
    def test_allowlists_load_with_family_bindings(self):
        table = load_allowlists()
        assert table["families"] == {"training": "training@1", "checkpointing": "checkpointing@1"}
        ck = table["allowlists"]["checkpointing@1"]
        assert "checkpoint.checkpoint_folder" in ck["exclude"]
        assert "checkpoint.num_checkpoints_write" in ck["exclude"]
        assert "checkpoint.num_checkpoints_read" in ck["exclude"]
        tr = table["allowlists"]["training@1"]
        assert "train.computation_time" in tr["include"]
        assert "dataset.num_files_train" not in tr["include"]
        assert "reader.read_threads" not in tr["include"]

    def test_llama70b_v2_and_v3_share_a_class(self):
        a = compute_core_config(V2_LLAMA70B, "checkpointing")
        b = compute_core_config(V3_LLAMA70B, "checkpointing")
        assert a == b
        assert a["algorithm"] == CORE_CONFIG_ALGORITHM
        assert a["allowlist"] == "checkpointing@1"
        assert re.fullmatch(r"[0-9a-f]{16}", a["hash"])
        assert "checkpoint.checkpoint_folder" not in a["keys"]
        assert "checkpoint.num_checkpoints_write" not in a["keys"]
        assert "model.parallelism.tensor" in a["keys"]
        assert a["keys"] == sorted(a["keys"])

    def test_unet3d_site_tunables_do_not_change_the_hash(self):
        base = compute_core_config(UNET3D_B200, "training")
        variant = copy.deepcopy(UNET3D_B200)
        variant["dataset"].update({"num_files_train": 96000, "num_subfolders_train": 0,
                                   "data_folder": "/other", "listing_validation_interval": 1,
                                   "skip_listing": False})
        variant["reader"].update({"read_threads": 8, "odirect": False,
                                  "multiprocessing_context": "spawn"})
        variant["storage"] = {"storage_type": "s3", "storage_root": "bucket",
                              "storage_options": {"prefetch_window": 4}}
        variant["checkpoint"]["checkpoint_folder"] = "/elsewhere"
        assert compute_core_config(variant, "training") == base
        for k in ("dataset.num_files_train", "reader.read_threads", "storage.storage_type",
                  "dataset.data_folder", "checkpoint.checkpoint_folder"):
            assert k not in base["keys"]
        for k in ("train.computation_time", "reader.batch_size", "dataset.record_length_bytes",
                  "model.name", "workflow.train", "metric.au", "checkpoint.checkpoint_after_epoch"):
            assert k in base["keys"]

    def test_emulated_accelerator_separates_classes(self):
        mi355 = copy.deepcopy(UNET3D_B200)
        mi355["train"]["computation_time"] = 0.636
        assert compute_core_config(mi355, "training")["hash"] != \
            compute_core_config(UNET3D_B200, "training")["hash"]

    def test_numeric_strings_and_integral_floats_normalise(self):
        a = compute_core_config({"reader": {"batch_size": 7}, "train": {"epochs": 5}}, "training")
        b = compute_core_config({"reader": {"batch_size": "7"}, "train": {"epochs": 5.0}}, "training")
        c = compute_core_config({"reader": {"batch_size": 8}, "train": {"epochs": 5}}, "training")
        assert a["hash"] == b["hash"] != c["hash"]

    def test_hash_is_reproducible_from_the_stamped_keys(self):
        cc = compute_core_config(UNET3D_B200, "training")
        flat = dict(prov.flatten_parameters(UNET3D_B200))
        assert prov.core_config_hash({k: flat[k] for k in cc["keys"]}) == cc["hash"]

    @pytest.mark.parametrize("family", ["kv_cache", "vector_database"])
    def test_non_dlio_families_stamp_unknown(self, family):
        cc = compute_core_config({"model": "llama3.1-8b", "num_users": 10}, family)
        assert cc == {"algorithm": CORE_CONFIG_ALGORITHM, "allowlist": UNKNOWN,
                      "hash": UNKNOWN, "keys": []}

    def test_no_allowlisted_keys_present_is_unknown_not_empty_hash(self):
        cc = compute_core_config({}, "training")
        assert cc["hash"] == UNKNOWN and cc["keys"] == [] and cc["allowlist"] == "training@1"


# ---------------------------------------------------------------------------
# Stamp file
# ---------------------------------------------------------------------------

class TestStampFile:
    def test_write_fixed_key_order_and_round_trip(self, tmp_path):
        stamp = RunProvenance.from_dict(_stamp())
        path = write_leaf_provenance(tmp_path, stamp)
        assert path == tmp_path / PROVENANCE_FILENAME
        text = path.read_text()
        assert text.endswith("\n")
        assert list(json.loads(text).keys()) == [
            "schema", "rules_edition", "layout_version", "tool", "dlio", "storage_library",
            "core_config", "stamped_at", "stamped_by", "provenance"]
        assert read_leaf_provenance(tmp_path) == stamp
        assert not list(tmp_path.glob(".provenance.json.tmp.*"))

    def test_notes_only_when_present(self, tmp_path):
        stamp = RunProvenance.from_dict(_stamp(notes="imported from the v3.0 tree"))
        write_leaf_provenance(tmp_path, stamp)
        d = json.loads((tmp_path / PROVENANCE_FILENAME).read_text())
        assert list(d)[-1] == "notes" and d["notes"] == "imported from the v3.0 tree"

    @pytest.mark.parametrize("mutate", [
        lambda d: d.pop("dlio"),
        lambda d: d.__setitem__("schema", "mlps-run-provenance/9"),
        lambda d: d.__setitem__("rules_edition", None),
        lambda d: d.__setitem__("layout_version", 0),
        lambda d: d["tool"].__setitem__("git_sha", None),
        lambda d: d["dlio"].pop("commit"),
        lambda d: d["core_config"].__setitem__("keys", "framework"),
        lambda d: d["provenance"].__setitem__("dlio", "guessed"),
        lambda d: d["provenance"].pop("tool"),
        lambda d: d["provenance"].__setitem__("extra", "runtime"),
        lambda d: d.__setitem__("storage_library", None),
    ], ids=["missing-stamp", "bad-schema", "null-edition", "layout-0", "null-git-sha",
            "missing-commit", "keys-not-list", "bad-tag", "missing-tag", "extra-tag",
            "null-storage-library"])
    def test_from_dict_rejects(self, mutate):
        d = _stamp()
        mutate(d)
        with pytest.raises(ProvenanceError):
            RunProvenance.from_dict(d)

    def test_unknown_strings_are_valid(self):
        d = _stamp(rules_edition=UNKNOWN,
                   tool={"name": "mlpstorage", "version": UNKNOWN, "git_sha": UNKNOWN,
                         "code_image": UNKNOWN},
                   dlio={"version": UNKNOWN, "source": UNKNOWN, "commit": UNKNOWN},
                   storage_library={"name": "none"},
                   core_config={"algorithm": CORE_CONFIG_ALGORITHM, "allowlist": UNKNOWN,
                                "hash": UNKNOWN, "keys": []})
        d["provenance"].update({"rules_edition": "unknown", "storage_library": "n/a"})
        assert RunProvenance.from_dict(d).to_dict() == d

    def test_malformed_file_raises(self, tmp_path):
        (tmp_path / PROVENANCE_FILENAME).write_text("{not json")
        with pytest.raises(ProvenanceError):
            read_leaf_provenance(tmp_path)


# ---------------------------------------------------------------------------
# Runtime collection
# ---------------------------------------------------------------------------

def _patched_packages(dlio=True, s3dlio=True):
    direct = {"dlio_benchmark": {"url": "https://github.com/mlcommons/DLIO_local_changes.git",
                                 "vcs_info": {"vcs": "git", "commit_id": DLIO_SHA,
                                              "requested_revision": DLIO_SHA}}}
    versions = {"dlio_benchmark": "3.0.4", "s3dlio": "0.9.112"}
    if not dlio:
        direct.pop("dlio_benchmark"); versions.pop("dlio_benchmark")
    if not s3dlio:
        versions.pop("s3dlio")
    return (patch.object(prov, "_read_direct_url", side_effect=lambda n: direct.get(n)),
            patch.object(prov, "_package_version", side_effect=lambda n: versions.get(n)))


class TestRuntimeCollection:
    def test_object_run_with_code_image(self):
        p1, p2 = _patched_packages()
        with p1, p2:
            s = collect_runtime_provenance(family="training", parameters=UNET3D_B200,
                                           code_image_hash=HASH_A, data_access_protocol="object")
        d = s.to_dict()
        assert d["schema"] == PROVENANCE_SCHEMA
        assert d["rules_edition"] == RULES_EDITION
        assert d["layout_version"] == MLPERF_RESULTS_VERSION
        assert d["tool"]["name"] == "mlpstorage" and d["tool"]["version"] == VERSION
        assert d["tool"]["code_image"] == f"md5-tree-v2:{HASH_A}"
        assert re.fullmatch(r"[0-9a-f]{40}|unknown", d["tool"]["git_sha"])
        assert d["dlio"] == {"version": "3.0.4",
                             "source": "https://github.com/mlcommons/DLIO_local_changes.git",
                             "commit": DLIO_SHA}
        assert d["storage_library"] == {"name": "s3dlio", "version": "0.9.112"}
        assert d["core_config"] == compute_core_config(UNET3D_B200, "training")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", d["stamped_at"])
        assert d["stamped_by"] == f"mlpstorage {VERSION}"
        assert d["provenance"] == {"rules_edition": "tool-constant",
                                   "layout_version": "tool-constant", "tool": "runtime",
                                   "dlio": "direct-url", "storage_library": "package-metadata",
                                   "core_config": "runtime"}

    @pytest.mark.parametrize("protocol", ["file", None])
    def test_file_run_has_no_storage_library(self, protocol):
        p1, p2 = _patched_packages()
        with p1, p2:
            s = collect_runtime_provenance(family="training", parameters=UNET3D_B200,
                                           code_image_hash=HASH_A, data_access_protocol=protocol)
        assert s.storage_library == {"name": "none"}
        assert s.provenance["storage_library"] == "n/a"

    def test_no_code_image_stamps_unknown(self):
        p1, p2 = _patched_packages()
        with p1, p2:
            s = collect_runtime_provenance(family="kv_cache", parameters={"model": "x"},
                                           code_image_hash=None, data_access_protocol=None)
        assert s.tool["code_image"] == UNKNOWN
        assert s.core_config["hash"] == UNKNOWN and s.provenance["core_config"] == "unknown"

    def test_dlio_wheel_without_direct_url_falls_back_to_package_metadata(self):
        p1, p2 = _patched_packages(dlio=False)
        with p1, patch.object(prov, "_package_version", side_effect=lambda n: {"dlio_benchmark": "3.0.4"}.get(n)):
            s = collect_runtime_provenance(family="training", parameters=UNET3D_B200,
                                           code_image_hash=HASH_A, data_access_protocol="file")
        assert s.dlio == {"version": "3.0.4", "source": UNKNOWN, "commit": UNKNOWN}
        assert s.provenance["dlio"] == "package-metadata"

    def test_dlio_not_installed_is_unknown(self):
        p1, p2 = _patched_packages(dlio=False, s3dlio=False)
        with p1, p2:
            s = collect_runtime_provenance(family="training", parameters=UNET3D_B200,
                                           code_image_hash=HASH_A, data_access_protocol="object")
        assert s.dlio == {"version": UNKNOWN, "source": UNKNOWN, "commit": UNKNOWN}
        assert s.provenance["dlio"] == "unknown"
        assert s.storage_library == {"name": "s3dlio", "version": UNKNOWN}
        assert s.provenance["storage_library"] == "unknown"

    def test_real_venv_exposes_dlio_commit(self):
        """The PEP 610 mechanism itself: this venv installs DLIO from git."""
        info = prov._read_direct_url("dlio_benchmark")
        if info is None:
            pytest.skip("dlio_benchmark not installed from a VCS URL here")
        assert re.fullmatch(r"[0-9a-f]{40}", info["vcs_info"]["commit_id"])


# ---------------------------------------------------------------------------
# Benchmark.write_metadata writes the sidecar
# ---------------------------------------------------------------------------

class TestBenchmarkWritesStamp:
    @pytest.fixture
    def benchmark(self, tmp_path):
        from tests.unit.test_benchmarks_base import ConcreteBenchmark
        args = Namespace(mode="closed", dry_run=False, orgname="Acme", systemname="sys-v1",
                         debug=False, verbose=False, what_if=False, stream_log_level="INFO",
                         results_dir=str(tmp_path), model="unet3d", command="run",
                         num_processes=8, accelerator_type="b200", data_access_protocol="file")
        with patch("mlpstorage_py.benchmarks.base.generate_output_location") as gen:
            gen.return_value = str(tmp_path / "output")
            return ConcreteBenchmark(args, run_datetime="20260919_120000")

    def test_sidecar_written_and_declared(self, benchmark):
        benchmark.combined_params = copy.deepcopy(UNET3D_B200)
        p1, p2 = _patched_packages()
        with p1, p2:
            benchmark.write_metadata()
        leaf = Path(benchmark.run_result_output)
        metadata = json.loads(Path(benchmark.metadata_file_path).read_text())
        assert metadata[METADATA_PROVENANCE_KEY] == PROVENANCE_FILENAME
        stamp = read_leaf_provenance(leaf)
        assert stamp.stamped_by == f"mlpstorage {VERSION}"
        assert stamp.tool["code_image"] == UNKNOWN  # no capture in this unit test
        assert stamp.core_config == compute_core_config(UNET3D_B200, "training")
        assert stamp.dlio["commit"] == DLIO_SHA

    def test_sidecar_names_the_pointed_image(self, benchmark):
        _write_pointer_atomic(Path(benchmark.run_result_output), HASH_A, benchmark.logger)
        p1, p2 = _patched_packages()
        with p1, p2:
            benchmark.write_metadata()
        assert read_leaf_provenance(Path(benchmark.run_result_output)).tool["code_image"] \
            == f"md5-tree-v2:{HASH_A}"

    def test_sidecar_failure_does_not_break_metadata(self, benchmark):
        with patch.object(prov, "write_leaf_provenance", side_effect=OSError("disk full")):
            benchmark.write_metadata()
        assert os.path.exists(benchmark.metadata_file_path)


# ---------------------------------------------------------------------------
# Derivation for layout-2 leaves (no sidecar, never rewritten)
# ---------------------------------------------------------------------------

class TestDerivedProvenance:
    def _check_common(self, s: RunProvenance):
        assert s.rules_edition == UNKNOWN and s.provenance["rules_edition"] == "unknown"
        assert s.layout_version == 2 and s.provenance["layout_version"] == "inferred"
        assert s.tool == {"name": "mlpstorage", "version": "3.0.38", "git_sha": GIT_SHA,
                          "code_image": f"md5-tree-v2:{HASH_A}"}
        assert s.provenance["tool"] == "code-hash-json"
        assert s.dlio == {"version": "3.0.4",
                          "source": "https://github.com/mlcommons/DLIO_local_changes.git",
                          "commit": DLIO_SHA}
        assert s.provenance["dlio"] == "uv-lock"
        assert s.core_config == compute_core_config(UNET3D_B200, "training")
        assert s.provenance["core_config"] == "runtime"
        assert "derived" in s.stamped_by and s.notes

    def test_global_pool(self, tmp_path):
        _image(tmp_path, HASH_A)
        leaf = _leaf(tmp_path, LEAF_RUN, parameters=UNET3D_B200, protocol="object")
        s = read_leaf_provenance(leaf, tmp_path)
        self._check_common(s)
        assert s.storage_library == {"name": "s3dlio", "version": "0.9.112"}
        assert s.provenance["storage_library"] == "uv-lock"
        assert not (leaf / PROVENANCE_FILENAME).exists()  # never written by a read

    def test_per_org_pool_and_ancestor_walk(self, tmp_path):
        _image(tmp_path, HASH_A, org="Acme")
        leaf = _leaf(tmp_path, LEAF_RUN, parameters=UNET3D_B200, protocol="file")
        s = read_leaf_provenance(leaf)  # no results_root: walks up
        self._check_common(s)
        assert s.storage_library == {"name": "none"} and s.provenance["storage_library"] == "n/a"

    def test_missing_image_and_lock(self, tmp_path):
        _image(tmp_path, HASH_A, uv_lock=None, git_sha=None)
        leaf = _leaf(tmp_path, LEAF_RUN, parameters=UNET3D_B200, pointer=HASH_B)
        s = read_leaf_provenance(leaf, tmp_path)
        assert s.tool == {"name": "mlpstorage", "version": UNKNOWN, "git_sha": UNKNOWN,
                          "code_image": f"md5-tree-v2:{HASH_B}"}
        assert s.provenance["tool"] == "unknown"
        assert s.dlio["commit"] == UNKNOWN and s.provenance["dlio"] == "unknown"
        leaf2 = _leaf(tmp_path, LEAF_RUN_2, parameters=UNET3D_B200, pointer=HASH_A)
        s2 = read_leaf_provenance(leaf2, tmp_path)
        assert s2.tool["version"] == "3.0.38" and s2.tool["git_sha"] == UNKNOWN
        assert s2.dlio == {"version": UNKNOWN, "source": UNKNOWN, "commit": UNKNOWN}

    def test_no_pointer_no_metadata(self, tmp_path):
        leaf = tmp_path / LEAF_RUN
        leaf.mkdir(parents=True)
        s = read_leaf_provenance(leaf, tmp_path)
        assert s.tool["code_image"] == UNKNOWN
        assert s.core_config["hash"] == UNKNOWN
        assert set(s.provenance.values()) <= PROVENANCE_TAGS


# ---------------------------------------------------------------------------
# submission.yaml
# ---------------------------------------------------------------------------

def _results_dir(tmp_path: Path) -> Path:
    rd = tmp_path / "results"
    rd.mkdir()
    write_sentinel(str(rd), "Acme")
    return rd


class TestSubmissionManifest:
    def test_contents(self, tmp_path):
        rd = _results_dir(tmp_path)
        _org(rd, systems=("sys-1",))
        _image(rd, HASH_A)
        _image(rd, HASH_B)
        stamped = _stamp()
        _leaf(rd, LEAF_RUN, parameters=UNET3D_B200, stamp=stamped, declare=True)
        _leaf(rd, LEAF_CKPT, parameters=V3_LLAMA70B, pointer=HASH_B)  # derived
        log = _Log()
        path = write_submission_manifest(str(rd), "closed", "Acme", log=log)
        assert path == rd / "closed" / "Acme" / MANIFEST_FILENAME
        m = read_submission_manifest(path)
        assert list(m)[:6] == ["schema", "orgname", "mode", "rules_edition", "layout_version",
                               "generated_at"]
        assert m["schema"] == MANIFEST_SCHEMA
        assert (m["orgname"], m["mode"]) == ("Acme", "closed")
        assert m["rules_edition"] == RULES_EDITION
        assert m["layout_version"] == MLPERF_RESULTS_VERSION
        assert m["generated_by"] == f"mlpstorage {VERSION}"
        assert m["tool_versions"] == ["3.0.38", "3.0.46"]
        assert m["dlio_commits"] == [DLIO_SHA]
        assert m["systems"] == [{"name": "sys-1", "description": "systems/sys-1.yaml",
                                 "pdf": "systems/sys-1.pdf"}]
        assert m["code_images"] == [f"md5-tree-v2:{HASH_A}", f"md5-tree-v2:{HASH_B}"]
        runs = m["runs"]
        assert [r["leaf"] for r in runs] == [
            "results/sys-1/checkpointing/llama3-70b/20260903_110000",
            "results/sys-1/training/unet3d/run/20260901_100000"]
        ckpt, run = runs
        assert run == {"leaf": "results/sys-1/training/unet3d/run/20260901_100000",
                       "benchmark": "training", "model": "unet3d", "command": "run",
                       "accelerator": "b200", "system": "sys-1",
                       "rules_edition": "3.0", "core_config": "0123456789abcdef",
                       "code_image": f"md5-tree-v2:{HASH_A}",
                       "provenance": "results/sys-1/training/unet3d/run/20260901_100000/provenance.json"}
        assert ckpt["benchmark"] == "checkpointing" and ckpt["command"] == "run"
        assert ckpt["rules_edition"] == UNKNOWN
        assert ckpt["core_config"] == compute_core_config(V3_LLAMA70B, "checkpointing")["hash"]
        assert ckpt["provenance"] == "derived"
        assert m["provenance"] == {"rules_edition": "tool-constant", "runs": "walked"}

    def test_deterministic_apart_from_generated_at(self, tmp_path):
        rd = _results_dir(tmp_path)
        _org(rd)
        _image(rd, HASH_A)
        _leaf(rd, LEAF_RUN, parameters=UNET3D_B200, stamp=_stamp(), declare=True)
        a = write_submission_manifest(str(rd), "closed", "Acme").read_text()
        b = write_submission_manifest(str(rd), "closed", "Acme").read_text()
        strip = lambda t: "\n".join(l for l in t.splitlines() if not l.startswith("generated_at:"))
        assert strip(a) == strip(b)
        assert a.count("generated_at:") == 1

    def test_missing_pdf_and_no_systems(self, tmp_path):
        rd = _results_dir(tmp_path)
        org = _org(rd, systems=("sys-1",))
        (org / "systems" / "sys-1.pdf").unlink()
        m = read_submission_manifest(write_submission_manifest(str(rd), "closed", "Acme"))
        assert m["systems"][0]["pdf"] == "missing"
        assert m["runs"] == [] and m["tool_versions"] == [] and m["code_images"] == []

    def test_gate_writes_only_under_a_sentinel(self, tmp_path):
        bare = tmp_path / "submissions"
        _org(bare)
        _org(bare, org="Beta", mode="open")
        assert write_manifests_if_results_dir(str(bare)) == []
        assert not list(bare.rglob(MANIFEST_FILENAME))
        rd = _results_dir(tmp_path)
        _org(rd)
        _org(rd, org="Beta", mode="open")
        (rd / "whatif" / "Acme" / "results").mkdir(parents=True)
        written = write_manifests_if_results_dir(str(rd))
        assert written == [rd / "closed" / "Acme" / MANIFEST_FILENAME,
                           rd / "open" / "Beta" / MANIFEST_FILENAME,
                           rd / "whatif" / "Acme" / MANIFEST_FILENAME]

    def test_reportgen_calls_the_gate_with_the_requested_root(self, tmp_path):
        import shutil
        from mlpstorage_py.report_generator import ReportGenerator
        src = Path(__file__).resolve().parents[1] / "fixtures" / "sample_results" / "multi_orgname"
        root = tmp_path / "repo_root"
        shutil.copytree(src, root)
        with patch("mlpstorage_py.report_generator.write_manifests_if_results_dir",
                   return_value=[]) as gate:
            gen = ReportGenerator(str(root), args=Namespace(debug=False), validate_structure=False)
            assert gen.generate_reports() == 0
        gate.assert_called_once()
        assert gate.call_args.args[0] == str(root)


# ---------------------------------------------------------------------------
# ProvenanceCheck: PROV-01 leafProvenance, PROV-02 submissionManifest
# ---------------------------------------------------------------------------

def _check(root: Path):
    log = _Log()
    return ProvenanceCheck(log=log, config=_make_config(), root_path=str(root)), log


class TestProvenanceCheck:
    def test_rule_ids(self):
        assert discover_rules(ProvenanceCheck) == {
            "PROV-01": ("leafProvenance", "leaf_provenance_check"),
            "PROV-02": ("submissionManifest", "submission_manifest_check")}

    def test_wired_into_the_checker(self):
        import inspect
        from mlpstorage_py.submission_checker import main as checker_main
        assert "ProvenanceCheck(log, config, args.input)" in inspect.getsource(checker_main.run)

    def test_pre_stamp_tree_is_silent(self, tmp_path):
        _org(tmp_path)
        _image(tmp_path, HASH_A)
        _leaf(tmp_path, LEAF_RUN, parameters=UNET3D_B200)
        _leaf(tmp_path, LEAF_CKPT, parameters=V3_LLAMA70B)
        check, log = _check(tmp_path)
        assert check() is True
        assert log.lines == []

    def test_valid_stamped_tree_passes(self, tmp_path):
        rd = _results_dir(tmp_path)
        _org(rd)
        _image(rd, HASH_A)
        _leaf(rd, LEAF_RUN, parameters=UNET3D_B200, stamp=_stamp(), declare=True)
        write_submission_manifest(str(rd), "closed", "Acme")
        check, log = _check(rd)
        assert check() is True
        assert log.errors == [] and log.warnings == []

    def test_declared_but_missing_sidecar(self, tmp_path):
        _org(tmp_path)
        _leaf(tmp_path, LEAF_RUN, parameters=UNET3D_B200, declare=True)
        check, log = _check(tmp_path)
        assert check.leaf_provenance_check() is False
        assert len(log.errors) == 1
        assert log.errors[0].startswith("[PROV-01 leafProvenance] ")
        assert "provenance.json" in log.errors[0] and LEAF_RUN in log.errors[0]

    def test_malformed_sidecar(self, tmp_path):
        _org(tmp_path)
        leaf = _leaf(tmp_path, LEAF_RUN, parameters=UNET3D_B200)
        (leaf / PROVENANCE_FILENAME).write_text('{"schema": "nope"}\n')
        check, log = _check(tmp_path)
        assert check.leaf_provenance_check() is False
        assert len(log.errors) == 1 and "[PROV-01 leafProvenance]" in log.errors[0]

    def test_sidecar_disagrees_with_pointer(self, tmp_path):
        _org(tmp_path)
        _leaf(tmp_path, LEAF_RUN, parameters=UNET3D_B200, pointer=HASH_B, stamp=_stamp())
        check, log = _check(tmp_path)
        assert check.leaf_provenance_check() is False
        assert len(log.errors) == 1
        assert HASH_A[:8] in log.errors[0] and HASH_B[:8] in log.errors[0]

    def test_accumulates_every_leaf(self, tmp_path):
        _org(tmp_path)
        _leaf(tmp_path, LEAF_RUN, parameters=UNET3D_B200, declare=True)
        _leaf(tmp_path, LEAF_RUN_2, parameters=UNET3D_B200, pointer=HASH_B, stamp=_stamp())
        _leaf(tmp_path, LEAF_CKPT, parameters=V3_LLAMA70B, stamp=_stamp())  # fine
        check, log = _check(tmp_path)
        assert check.leaf_provenance_check() is False
        assert len(log.errors) == 2

    def test_manifest_edition_mismatch(self, tmp_path):
        rd = _results_dir(tmp_path)
        _org(rd)
        _leaf(rd, LEAF_RUN, parameters=UNET3D_B200, stamp=_stamp(rules_edition="2.0"), declare=True)
        write_submission_manifest(str(rd), "closed", "Acme")
        check, log = _check(rd)
        assert check.submission_manifest_check() is False
        assert len(log.errors) == 1
        assert log.errors[0].startswith("[PROV-02 submissionManifest] ")
        assert "2.0" in log.errors[0] and RULES_EDITION in log.errors[0]

    def test_manifest_stale(self, tmp_path):
        rd = _results_dir(tmp_path)
        _org(rd)
        _leaf(rd, LEAF_RUN, parameters=UNET3D_B200, stamp=_stamp(), declare=True)
        write_submission_manifest(str(rd), "closed", "Acme")
        _leaf(rd, LEAF_RUN_2, parameters=UNET3D_B200, stamp=_stamp(), declare=True)
        check, log = _check(rd)
        assert check.submission_manifest_check() is False
        assert len(log.errors) == 1
        assert "20260902_100000" in log.errors[0] and "reportgen" in log.errors[0]

    def test_manifest_malformed(self, tmp_path):
        rd = _results_dir(tmp_path)
        org = _org(rd)
        (org / MANIFEST_FILENAME).write_text("schema: something-else\n")
        check, log = _check(rd)
        assert check.submission_manifest_check() is False
        assert len(log.errors) == 1 and "[PROV-02 submissionManifest]" in log.errors[0]

    def test_derived_leaves_do_not_trip_the_edition_check(self, tmp_path):
        rd = _results_dir(tmp_path)
        _org(rd)
        _image(rd, HASH_A)
        _leaf(rd, LEAF_RUN, parameters=UNET3D_B200)  # pre-stamp leaf, edition unknown
        write_submission_manifest(str(rd), "closed", "Acme")
        check, log = _check(rd)
        assert check() is True and log.errors == []


# ---------------------------------------------------------------------------
# mlpstorage runs show
# ---------------------------------------------------------------------------

from tests.unit.test_runs_management import (  # noqa: E402  (fixtures by import)
    LEAF_TRAIN_RUN, _main, tree, xdg,
)


class TestRunsShow:
    def test_show_prints_a_stamp(self, tree, capsys):
        write_leaf_provenance(Path(tree) / LEAF_TRAIN_RUN, RunProvenance.from_dict(_stamp()))
        assert _main(["runs", "show", "2"]) == EXIT_CODE.SUCCESS
        out = capsys.readouterr().out
        assert "provenance:  provenance.json" in out
        assert "rules edition: 3.0" in out
        assert "dlio:" in out and DLIO_SHA[:8] in out
        assert "core config:   0123456789abcdef" in out

    def test_show_says_derived_for_a_pre_stamp_leaf(self, tree, capsys):
        assert _main(["runs", "show", "2"]) == EXIT_CODE.SUCCESS
        out = capsys.readouterr().out
        assert "provenance:  derived" in out
        assert "rules edition: unknown" in out
