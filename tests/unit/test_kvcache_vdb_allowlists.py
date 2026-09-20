"""core-config-v1 allowlists for the two non-DLIO families
(``kv_cache@1`` / ``vector_database@1`` in ``mlpstorage_py/rules/core_config_keys.yaml``)
and their comparability classes (``llama3.1-8b-A`` / ``milvus-diskann-A`` in
``mlpstorage_py/rules/editions.yaml``).

Until now both families stamped ``core_config.hash = "unknown"`` and had no
class, because what the runtime wrote into ``metadata['parameters']`` did not
describe the workload:

* a CLOSED kv_cache run executes the three Table KVCache-1 Options (Rules.md
  6.3.1.1) through ``mlperf_wrapper.py``; ``parameters`` carried the CLI
  placeholders (``gpu_mem_gb 16``, ``enable_rag True``...) that never reach
  ``kv-cache.py`` in CLOSED (issue #537 keys, kept for the run checker);
* a vector_database leaf carried the four reportgen identity keys at best
  (``{}`` on the v3.0 tree).

Now the runtime records the workload it actually drives -- kv_cache: the
per-Option parameters emitted to ``kv-cache.py``, the sequence locks and the
global feature flags forwarded; vector_database: the resolved ``database`` /
``index`` / ``dataset`` / ``benchmark`` sections -- and the allowlists hash the
Rules.md-fixed subset. Leaves written before this (every v3.0 leaf) are
reconstructed at read time from what they did record: the wrapper command
lines in ``command_output_files`` (kv_cache), ``summary.json`` /
``config.json`` / ``result_verdict.json`` (vector_database). A leaf that
recorded nothing usable keeps ``unknown``.

Covered here:
- the allowlists load with four family bindings and exclude the site knobs;
- kv_cache runtime block (CLOSED = Table KVCache-1 + locks, features empty;
  OPEN = per-key supersede + forwarded features), hash stability against the
  scaling axis, hash change on a workload change;
- kv_cache reconstruction from the recorded wrapper command lines (CLOSED and
  OPEN shapes of the v3.0 tree), and ``unknown`` when they are absent;
- vector_database runtime block (datagen from CLI, run from YAML + CLI), hash
  stability against the 5.6.4 tunables, hash change on scale / index / recall;
- vector_database reconstruction from the two v3.0 leaf shapes (MPI leaves
  with ``summary.json``; single-node leaves with ``config.json`` +
  ``result_verdict.json``), datagen leaves hash over the load keys only;
- the seeded classes match the CLOSED runtime blocks and the shipped
  ``configs/vectordbbench/default.yaml``; EDN-02 fires on a CLOSED variant;
  ``runs show`` prints the class; Rules.md / ManPage.md name the allowlists.
"""
from __future__ import annotations

import copy
import json
import re
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from mlpstorage_py.config import EXIT_CODE
from mlpstorage_py.provenance import (
    CORE_CONFIG_ALGORITHM,
    UNKNOWN,
    compute_core_config,
    derive_leaf_provenance,
    load_allowlists,
    workload_parameters,
)
from tests.unit.test_leaf_provenance import HASH_A, _image, _leaf, _org, _stamp
from tests.unit.test_rules_editions import LEAF_CLOSED_KV, _check, _rule_lines, _table
from tests.unit.test_runs_management import (  # noqa: F401  (fixtures by import)
    LEAF_VDB, _main, tree, xdg,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LEAF_CLOSED_VDB = "closed/Acme/results/sys-1/vector_database/milvus/DISKANN/run/20260904_120000"
LEAF_CLOSED_VDB_DATAGEN = "closed/Acme/results/sys-1/vector_database/milvus/DISKANN/datagen/20260903_120000"
LEAF_OPEN_KV = "open/Acme/results/sys-1/kv_cache/llama3.1-8b/run/20260905_130000"

# --- the CLOSED kv_cache workload: Rules.md Table KVCache-1 + 6.3.2.1 locks ---
KV_OPTION_1 = {"model": "llama3.1-8b", "num_users": 200, "duration": 300, "gpu_mem_gb": 0,
               "cpu_mem_gb": 0, "max_concurrent_allocs": 16, "generation_mode": "none"}
KV_OPTION_2 = {"model": "llama3.1-8b", "num_users": 100, "duration": 300, "gpu_mem_gb": 0,
               "cpu_mem_gb": 4, "max_concurrent_allocs": 16, "generation_mode": "none"}
KV_OPTION_3 = {"model": "llama3.1-70b-instruct", "num_users": 70, "duration": 300, "gpu_mem_gb": 0,
               "cpu_mem_gb": 0, "max_concurrent_allocs": 4, "generation_mode": "none"}
KV_CLOSED_PARAMETERS = {
    "options": {"1": KV_OPTION_1, "2": KV_OPTION_2, "3": KV_OPTION_3},
    "sequence": {"seed": 42, "trials": 3, "inter_option_delay_s": 90, "config": "default"},
    "features": {},
}
KV_CLOSED_HASH = compute_core_config(KV_CLOSED_PARAMETERS, "kv_cache")["hash"]

# --- the CLOSED vector_database workload: configs/vectordbbench/default.yaml ---
VDB_RUN_PARAMETERS = {
    "database": {"database": "milvus"},
    "index": {"index_type": "DISKANN"},
    "dataset": {"num_vectors": 1000000, "dimension": 1536},
    "benchmark": {"recall_k": 10, "search_limit": 10},
}
VDB_RUN_HASH = compute_core_config(VDB_RUN_PARAMETERS, "vector_database")["hash"]
VDB_DATAGEN_PARAMETERS = {k: v for k, v in VDB_RUN_PARAMETERS.items() if k != "benchmark"}
VDB_DATAGEN_HASH = compute_core_config(VDB_DATAGEN_PARAMETERS, "vector_database")["hash"]

# The flat #537 placeholder block a v3.0 CLOSED kv_cache leaf carries.
KV_V30_PLACEHOLDER = {"model": "llama3.1-8b", "num_users": 100, "duration": 60, "gpu_mem_gb": 16.0,
                      "cpu_mem_gb": 32.0, "cache_dir": "/mnt/nvme/kvcache",
                      "generation_mode": "realistic", "performance_profile": "throughput"}


def _wrapper_cmd(option: int, trial: int, params: dict, *, seed: int = 42,
                 config: str = "/opt/storage/kv_cache_benchmark/config.yaml",
                 features: str = "") -> str:
    """One recorded ``command_output_files[*].command`` of a v3.0 kv_cache leaf."""
    p = params
    return (f"mpirun -n 20 -host h1:5,h2:5,h3:5,h4:5 --npernode 5 --bind-to none --map-by node "
            f"--allow-run-as-root --mca orte_abort_on_non_zero_status 0 /opt/storage/.venv/bin/python3 "
            f"/opt/storage/kv_cache_benchmark/mlperf_wrapper.py "
            f"--rank-output-base /results/closed/Acme/results/sys-1/kv_cache/llama3.1-8b/run/20260905_130000/option_{option}/trial_{trial} "
            f"--rank-cache-base /mnt/nvme/kvcache --seed-base {seed} --config {config} "
            f"--model {p['model']} --num-users {p['num_users']} --duration {p['duration']} "
            f"--gpu-mem-gb {p['gpu_mem_gb']} --cpu-mem-gb {p['cpu_mem_gb']} "
            f"--max-concurrent-allocs {p['max_concurrent_allocs']} --generation-mode {p['generation_mode']}"
            f"{' ' + features if features else ''}")


def _closed_kv_command_lines(trials: int = 3) -> list:
    probe = {"command": "mpirun -n 4 python3 -c 'probe'", "stdout": "/x/probe.stdout.log",
             "stderr": "/x/probe.stderr.log"}
    out = [probe]
    for option, params in ((1, KV_OPTION_1), (2, KV_OPTION_2), (3, KV_OPTION_3)):
        for trial in range(trials):
            out.append({"command": _wrapper_cmd(option, trial, params),
                        "stdout": f"/x/kvcache_opt{option}_trial{trial}.stdout.log",
                        "stderr": f"/x/kvcache_opt{option}_trial{trial}.stderr.log"})
    return out


def _kv_leaf(root: Path, rel: str, *, parameters: dict, command_output_files=None,
             command: str = "run", args: dict | None = None, pointer: str | None = HASH_A,
             stamp: dict | None = None, declare: bool = False) -> Path:
    """A kv_cache leaf the way the v3.0 tool wrote it (no provenance.json unless given)."""
    leaf = root / rel
    leaf.mkdir(parents=True, exist_ok=True)
    ts = rel.rsplit("/", 1)[1]
    mode = rel.split("/", 1)[0]
    body = {"benchmark_type": "kv_cache", "model": "llama3.1-8b", "command": command,
            "run_datetime": ts, "accelerator": None, "num_processes": 20,
            "parameters": parameters, "override_parameters": {},
            "args": {"mode": mode, "benchmark": "kvcache", "command": command,
                     "inter_option_delay": 90, "trials": 3, "seed": 42, **(args or {})},
            "command_output_files": command_output_files if command_output_files is not None else [],
            "exit_status": 0}
    if declare:
        body["provenance_file"] = "provenance.json"
    (leaf / f"kv_cache_{ts}_metadata.json").write_text(json.dumps(body, indent=2))
    if pointer:
        (leaf / ".mlps-code-image").write_text(f"md5-tree-v2:{pointer}")
    if stamp is not None:
        (leaf / "provenance.json").write_text(json.dumps(stamp, indent=2) + "\n")
    return leaf


def _vdb_leaf(root: Path, rel: str, *, parameters: dict | None = None, command: str = "run",
              args: dict | None = None, executed_command: str = "", files: dict | None = None,
              pointer: str | None = HASH_A, stamp: dict | None = None, declare: bool = False) -> Path:
    """A vector_database leaf; ``files`` maps sibling file names to JSON bodies."""
    leaf = root / rel
    leaf.mkdir(parents=True, exist_ok=True)
    ts = rel.rsplit("/", 1)[1]
    mode = rel.split("/", 1)[0]
    body = {"benchmark_type": "vector_database", "model": "milvus_DISKANN", "command": command,
            "run_datetime": ts, "accelerator": None, "vdb_engine": "milvus", "vdb_index": "DISKANN",
            "vectordb_config": "default", "parameters": parameters if parameters is not None else {},
            "override_parameters": {}, "executed_command": executed_command,
            "args": {"mode": mode, "benchmark": "vectordb", "command": command, "vdb_engine": "milvus",
                     "vdb_index": "DISKANN", "data_access_protocol": "file", **(args or {})},
            "exit_status": 0}
    if declare:
        body["provenance_file"] = "provenance.json"
    (leaf / f"vector_database_{ts}_metadata.json").write_text(json.dumps(body, indent=2))
    for name, content in (files or {}).items():
        (leaf / name).write_text(json.dumps(content, indent=2))
    if pointer:
        (leaf / ".mlps-code-image").write_text(f"md5-tree-v2:{pointer}")
    if stamp is not None:
        (leaf / "provenance.json").write_text(json.dumps(stamp, indent=2) + "\n")
    return leaf


def _kv_stamp(core_hash: str = KV_CLOSED_HASH) -> dict:
    return _stamp(core_config={"algorithm": CORE_CONFIG_ALGORITHM, "allowlist": "kv_cache@1",
                               "hash": core_hash, "keys": ["options.1.model"]},
                  storage_library={"name": "none"})


def _vdb_stamp(core_hash: str = VDB_RUN_HASH) -> dict:
    return _stamp(core_config={"algorithm": CORE_CONFIG_ALGORITHM, "allowlist": "vector_database@1",
                               "hash": core_hash, "keys": ["database.database"]},
                  storage_library={"name": "none"})


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "sub"
    _org(root)
    _image(root, HASH_A)
    return root


# ---------------------------------------------------------------------------
# The allowlists
# ---------------------------------------------------------------------------

class TestAllowlists:
    def test_four_family_bindings(self):
        table = load_allowlists()
        assert table["families"] == {"training": "training@1", "checkpointing": "checkpointing@1",
                                     "kv_cache": "kv_cache@1", "vector_database": "vector_database@1"}

    def test_kvcache_allowlist_is_the_option_table_seed_config_and_features(self):
        kv = load_allowlists()["allowlists"]["kv_cache@1"]
        assert set(kv["include"]) == {"options.*", "sequence.seed", "sequence.config", "features.*"}
        # measurement protocol and the scaling axis (6.3.3) stay out
        for absent in ("sequence.trials", "sequence.inter_option_delay_s", "cache_dir", "num_users"):
            assert absent not in kv["include"]

    def test_vdb_allowlist_is_the_rules_fixed_subset(self):
        vdb = load_allowlists()["allowlists"]["vector_database@1"]
        assert set(vdb["include"]) == {"database.database", "index.index_type", "dataset.num_vectors",
                                       "dataset.dimension", "benchmark.recall_k", "benchmark.search_limit"}
        # everything in the 5.6.4 tunable table stays out
        for absent in ("index.metric_type", "index.max_degree", "benchmark.runtime", "benchmark.mode",
                       "benchmark.batch_size", "dataset.collection_name", "dataset.num_shards",
                       "storage.storage_root", "index.*", "dataset.*", "benchmark.*"):
            assert absent not in vdb["include"]

    def test_placeholder_blocks_still_hash_unknown(self):
        """The #537 flat keys never reach kv-cache.py in CLOSED; a block that
        carries only them (every v3.0 leaf) is not a workload description."""
        cc = compute_core_config(KV_V30_PLACEHOLDER, "kv_cache")
        assert cc == {"algorithm": CORE_CONFIG_ALGORITHM, "allowlist": "kv_cache@1",
                      "hash": UNKNOWN, "keys": []}
        cc = compute_core_config({"engine": "milvus", "index_type": "DISKANN", "num_vectors": None,
                                  "dimension": None}, "vector_database")
        assert cc["hash"] == UNKNOWN and cc["allowlist"] == "vector_database@1"


# ---------------------------------------------------------------------------
# kv_cache: the hash
# ---------------------------------------------------------------------------

class TestKVCacheHash:
    def test_closed_block_hashes_the_option_table_seed_and_config(self):
        cc = compute_core_config(KV_CLOSED_PARAMETERS, "kv_cache")
        assert cc["allowlist"] == "kv_cache@1"
        assert re.fullmatch(r"[0-9a-f]{16}", cc["hash"])
        assert cc["keys"] == sorted(cc["keys"])
        assert "options.1.num_users" in cc["keys"] and "options.3.model" in cc["keys"]
        assert "sequence.seed" in cc["keys"] and "sequence.config" in cc["keys"]
        assert "sequence.trials" not in cc["keys"] and "sequence.inter_option_delay_s" not in cc["keys"]

    def test_scaling_axis_and_protocol_do_not_change_the_hash(self):
        v = copy.deepcopy(KV_CLOSED_PARAMETERS)
        v.update(KV_V30_PLACEHOLDER)  # the flat placeholders ride along, unhashed
        v["sequence"]["trials"] = 1
        v["sequence"]["inter_option_delay_s"] = 15
        assert compute_core_config(v, "kv_cache")["hash"] == KV_CLOSED_HASH

    @pytest.mark.parametrize("change", [
        ("options", "1", "num_users", 201),
        ("options", "3", "model", "llama3.1-8b"),
        ("options", "2", "cpu_mem_gb", 0),
        ("sequence", "seed", 7),
        ("sequence", "config", "custom"),
        ("features", "enable_rag", True),
    ])
    def test_a_workload_change_changes_the_hash(self, change):
        v = copy.deepcopy(KV_CLOSED_PARAMETERS)
        cur = v
        for key in change[:-2]:
            cur = cur[key]
        cur[change[-2]] = change[-1]
        assert compute_core_config(v, "kv_cache")["hash"] != KV_CLOSED_HASH

    def test_string_values_from_argv_hash_like_the_runtime_ints(self):
        v = copy.deepcopy(KV_CLOSED_PARAMETERS)
        v["options"]["1"].update({"num_users": "200", "gpu_mem_gb": "0.0", "duration": "300"})
        v["sequence"]["seed"] = "42"
        assert compute_core_config(v, "kv_cache")["hash"] == KV_CLOSED_HASH


# ---------------------------------------------------------------------------
# kv_cache: the runtime block
# ---------------------------------------------------------------------------

def _kv_args(tmp_path, mode: str = "closed", **over) -> Namespace:
    ns = Namespace(
        debug=False, verbose=False, what_if=False, stream_log_level="INFO", mode=mode,
        orgname="Acme", systemname="sys-1", results_dir=str(tmp_path), model="llama3.1-8b",
        command="run", num_users=100, duration=60, gpu_mem_gb=16.0, cpu_mem_gb=32.0,
        cache_dir="/mnt/nvme/kvcache", generation_mode="realistic", performance_profile="throughput",
        kvcache_bin_path=None, disable_multi_turn=False, disable_prefix_caching=False,
        enable_rag=True, rag_num_docs=10, enable_autoscaling=True, autoscaler_mode="qos",
        enable_latency_tracing=False, seed=None, trials=None, inter_option_delay=None, config=None,
        max_concurrent_allocs=None, exec_type=None, hosts=None, num_processes=None, npernode=None,
        mpi_bin="mpirun", oversubscribe=False, allow_run_as_root=False, mpi_params=None,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def _kv_benchmark(args, tmp_path):
    from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
    from tests.fixtures.mock_logger import MockLogger
    with patch("mlpstorage_py.benchmarks.base.generate_output_location") as gen, \
         patch("mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information") as cl:
        gen.return_value = str(tmp_path / "output")
        cl.return_value = None
        return KVCacheBenchmark(args, logger=MockLogger(), run_datetime="20260905_130000")


class TestKVCacheRuntimeBlock:
    def test_closed_metadata_records_the_table_the_locks_and_no_features(self, tmp_path):
        params = _kv_benchmark(_kv_args(tmp_path), tmp_path).metadata["parameters"]
        assert params["options"] == KV_CLOSED_PARAMETERS["options"]
        assert params["sequence"] == KV_CLOSED_PARAMETERS["sequence"]
        assert params["features"] == {}
        # the #537 keys the run checker reads are still there
        assert params["model"] == "llama3.1-8b" and params["num_users"] == 100
        assert compute_core_config(params, "kv_cache")["hash"] == KV_CLOSED_HASH

    def test_closed_placeholders_do_not_leak_into_the_options(self, tmp_path):
        """CLOSED CLI defaults (gpu 16 / cpu 32 / rag / autoscaling) never reach
        kv-cache.py; the block must say what ran, not what argparse holds."""
        params = _kv_benchmark(_kv_args(tmp_path, gpu_mem_gb=16.0, num_users=999), tmp_path).metadata["parameters"]
        assert params["options"]["1"]["gpu_mem_gb"] == 0
        assert params["options"]["1"]["num_users"] == 200
        assert params["features"] == {}

    def test_open_supersedes_per_key_and_forwards_features(self, tmp_path):
        args = _kv_args(tmp_path, mode="open", num_users=400, duration=60, gpu_mem_gb=1.0, cpu_mem_gb=1.0,
                        generation_mode="none", enable_rag=False, enable_autoscaling=False,
                        trials=1, inter_option_delay=15)
        params = _kv_benchmark(args, tmp_path).metadata["parameters"]
        for o in ("1", "2", "3"):
            assert params["options"][o]["model"] == "llama3.1-8b"
            assert params["options"][o]["num_users"] == 400
            assert params["options"][o]["duration"] == 60
            assert params["options"][o]["gpu_mem_gb"] == 1.0 and params["options"][o]["cpu_mem_gb"] == 1.0
            assert params["options"][o]["generation_mode"] == "none"
        # max-concurrent-allocs is never exposed: per-option table value
        assert [params["options"][o]["max_concurrent_allocs"] for o in ("1", "2", "3")] == [16, 16, 4]
        assert params["sequence"] == {"seed": 42, "trials": 1, "inter_option_delay_s": 15, "config": "default"}
        # exactly what _build_global_kvcache_args forwards: value flags with a value, true flags
        assert params["features"] == {"rag_num_docs": 10, "autoscaler_mode": "qos",
                                      "performance_profile": "throughput"}
        assert compute_core_config(params, "kv_cache")["hash"] != KV_CLOSED_HASH

    def test_open_true_flags_and_custom_config_are_recorded(self, tmp_path):
        args = _kv_args(tmp_path, mode="open", enable_rag=True, disable_multi_turn=True,
                        config="/etc/my-kv.yaml", seed=7)
        params = _kv_benchmark(args, tmp_path).metadata["parameters"]
        assert params["features"]["enable_rag"] is True
        assert params["features"]["disable_multi_turn"] is True
        assert "disable_prefix_caching" not in params["features"]
        assert params["sequence"]["config"] == "custom"
        assert params["sequence"]["seed"] == 7

    def test_block_mirrors_the_argv_actually_emitted(self, tmp_path):
        """The block and the wrapper argv come from one builder."""
        bm = _kv_benchmark(_kv_args(tmp_path, mode="open", num_users=33), tmp_path)
        argv = bm._build_option_kvcache_args(2, False)
        block = bm.metadata["parameters"]["options"]["2"]
        assert argv[argv.index("--num-users") + 1] == "33" and block["num_users"] == 33
        assert argv[argv.index("--max-concurrent-allocs") + 1] == "16" and block["max_concurrent_allocs"] == 16


# ---------------------------------------------------------------------------
# kv_cache: reconstruction from a v3.0 leaf
# ---------------------------------------------------------------------------

class TestKVCacheReconstruction:
    def test_closed_v30_leaf_derives_the_closed_hash(self, tmp_path):
        root = _root(tmp_path)
        leaf = _kv_leaf(root, LEAF_CLOSED_KV, parameters=KV_V30_PLACEHOLDER,
                        command_output_files=_closed_kv_command_lines())
        stamp = derive_leaf_provenance(leaf, root)
        assert stamp.core_config["allowlist"] == "kv_cache@1"
        assert stamp.core_config["hash"] == KV_CLOSED_HASH
        assert stamp.provenance["core_config"] == "runtime"
        assert stamp.storage_library == {"name": "none"}

    def test_workload_parameters_rebuilds_the_block(self, tmp_path):
        md = json.loads((_kv_leaf(_root(tmp_path), LEAF_CLOSED_KV, parameters=KV_V30_PLACEHOLDER,
                                  command_output_files=_closed_kv_command_lines())
                         / "kv_cache_20260905_130000_metadata.json").read_text())
        block = workload_parameters("kv_cache", md)
        assert block["options"].keys() == {"1", "2", "3"}
        assert block["options"]["3"]["model"] == "llama3.1-70b-instruct"
        assert block["sequence"] == {"seed": 42, "trials": 3, "inter_option_delay_s": 90, "config": "default"}
        assert block["features"] == {}
        assert block["model"] == "llama3.1-8b"  # the flat keys stay

    def test_open_v30_leaf_rebuilds_features_from_the_argv(self, tmp_path):
        root = _root(tmp_path)
        _org(root, mode="open")
        opt = {"model": "llama3.1-8b", "num_users": 400, "duration": 60, "gpu_mem_gb": "1.0",
               "cpu_mem_gb": "1.0", "max_concurrent_allocs": 16, "generation_mode": "none"}
        feats = "--rag-num-docs 10 --autoscaler-mode qos --performance-profile throughput --enable-rag"
        cmds = [{"command": _wrapper_cmd(o, 0, dict(opt, max_concurrent_allocs=16 if o < 3 else 4),
                                         features=feats)} for o in (1, 2, 3)]
        leaf = _kv_leaf(root, LEAF_OPEN_KV, parameters=KV_V30_PLACEHOLDER, command_output_files=cmds,
                        args={"inter_option_delay": 15, "trials": 1})
        block = workload_parameters("kv_cache", json.loads(next(leaf.glob("*_metadata.json")).read_text()))
        assert block["options"]["1"]["num_users"] == 400
        assert block["options"]["3"]["max_concurrent_allocs"] == 4
        assert block["features"] == {"rag_num_docs": 10, "autoscaler_mode": "qos",
                                     "performance_profile": "throughput", "enable_rag": True}
        assert block["sequence"] == {"seed": 42, "trials": 1, "inter_option_delay_s": 15, "config": "default"}
        # and it is the block the new runtime writes for the same invocation
        args = _kv_args(tmp_path, mode="open", num_users=400, duration=60, gpu_mem_gb=1.0, cpu_mem_gb=1.0,
                        generation_mode="none", enable_rag=True, enable_autoscaling=False,
                        trials=1, inter_option_delay=15)
        runtime = _kv_benchmark(args, tmp_path).metadata["parameters"]
        assert compute_core_config(block, "kv_cache")["hash"] == compute_core_config(runtime, "kv_cache")["hash"]

    def test_custom_config_path_is_custom(self, tmp_path):
        cmds = [{"command": _wrapper_cmd(o, 0, p, config="/home/me/my-kv.yaml")}
                for o, p in ((1, KV_OPTION_1), (2, KV_OPTION_2), (3, KV_OPTION_3))]
        leaf = _kv_leaf(_root(tmp_path), LEAF_CLOSED_KV, parameters=KV_V30_PLACEHOLDER, command_output_files=cmds)
        block = workload_parameters("kv_cache", json.loads(next(leaf.glob("*_metadata.json")).read_text()))
        assert block["sequence"]["config"] == "custom"

    def test_no_command_lines_stays_unknown(self, tmp_path):
        root = _root(tmp_path)
        leaf = _kv_leaf(root, LEAF_CLOSED_KV, parameters=KV_V30_PLACEHOLDER, command_output_files=[])
        cc = derive_leaf_provenance(leaf, root).core_config
        assert cc["hash"] == UNKNOWN and cc["allowlist"] == "kv_cache@1" and cc["keys"] == []

    def test_incomplete_sequence_stays_unknown(self, tmp_path):
        """Two of three Options recorded (killed run): not the workload."""
        root = _root(tmp_path)
        cmds = [c for c in _closed_kv_command_lines() if "option_3" not in c["command"]]
        leaf = _kv_leaf(root, LEAF_CLOSED_KV, parameters=KV_V30_PLACEHOLDER, command_output_files=cmds)
        assert derive_leaf_provenance(leaf, root).core_config["hash"] == UNKNOWN

    def test_datasize_leaf_stays_unknown(self, tmp_path):
        root = _root(tmp_path)
        leaf = _kv_leaf(root, LEAF_CLOSED_KV.replace("/run/", "/datasize/"), parameters=KV_V30_PLACEHOLDER,
                        command="datasize")
        assert derive_leaf_provenance(leaf, root).core_config["hash"] == UNKNOWN

    def test_a_structured_block_is_used_as_is(self, tmp_path):
        """A leaf written by this tool already carries the block; the command
        lines are not consulted."""
        root = _root(tmp_path)
        leaf = _kv_leaf(root, LEAF_CLOSED_KV, parameters={**KV_V30_PLACEHOLDER, **KV_CLOSED_PARAMETERS},
                        command_output_files=[])
        assert derive_leaf_provenance(leaf, root).core_config["hash"] == KV_CLOSED_HASH


# ---------------------------------------------------------------------------
# vector_database: the hash
# ---------------------------------------------------------------------------

class TestVDBHash:
    def test_run_block_hashes_the_six_fixed_keys(self):
        cc = compute_core_config(VDB_RUN_PARAMETERS, "vector_database")
        assert cc["allowlist"] == "vector_database@1"
        assert cc["keys"] == ["benchmark.recall_k", "benchmark.search_limit", "database.database",
                              "dataset.dimension", "dataset.num_vectors", "index.index_type"]

    def test_shipped_default_yaml_hashes_to_the_run_block(self):
        import yaml
        y = yaml.safe_load((PROJECT_ROOT / "configs/vectordbbench/default.yaml").read_text())
        assert compute_core_config(y, "vector_database")["hash"] == VDB_RUN_HASH

    def test_tunables_do_not_change_the_hash(self):
        v = copy.deepcopy(VDB_RUN_PARAMETERS)
        v["database"].update({"host": "10.0.0.5", "port": 19531})
        v["index"].update({"metric_type": "L2", "index_params": {"max_degree": 32, "search_list_size": 100}})
        v["dataset"].update({"collection_name": "x", "num_shards": 10, "chunk_size": 5, "batch_size": 7,
                             "vector_dtype": "FLOAT16_VECTOR", "distribution": "uniform"})
        v["benchmark"].update({"mode": "query_count", "runtime": 300, "queries": 5000, "batch_size": 100,
                               "report_count": 5, "num_query_processes": 128, "search_ef": 100,
                               "num_query_vectors": 500})
        v["storage"] = {"storage_root": "/data", "storage_type": "s3"}
        v.update({"engine": "milvus", "index_type": "DISKANN", "num_vectors": 1000000, "dimension": 1536})
        assert compute_core_config(v, "vector_database")["hash"] == VDB_RUN_HASH

    @pytest.mark.parametrize("section,key,value", [
        ("dataset", "num_vectors", 10000000),
        ("dataset", "dimension", 768),
        ("index", "index_type", "HNSW"),
        ("database", "database", "pgvector"),
        ("benchmark", "recall_k", 5),
        ("benchmark", "search_limit", 100),
    ])
    def test_a_workload_change_changes_the_hash(self, section, key, value):
        v = copy.deepcopy(VDB_RUN_PARAMETERS)
        v[section][key] = value
        assert compute_core_config(v, "vector_database")["hash"] != VDB_RUN_HASH

    def test_datagen_block_hashes_the_load_keys_only(self):
        cc = compute_core_config(VDB_DATAGEN_PARAMETERS, "vector_database")
        assert cc["hash"] not in (UNKNOWN, VDB_RUN_HASH)
        assert cc["keys"] == ["database.database", "dataset.dimension", "dataset.num_vectors", "index.index_type"]


# ---------------------------------------------------------------------------
# vector_database: the runtime block
# ---------------------------------------------------------------------------

def _vdb_args(tmp_path, command: str, **over) -> Namespace:
    ns = Namespace(
        debug=False, verbose=False, what_if=False, stream_log_level="INFO", mode="closed",
        orgname="Acme", systemname="sys-1", results_dir=str(tmp_path), command=command,
        config="default", host="127.0.0.1", port=19530, collection=None, category=None,
        vdb_engine="milvus", vdb_index=None, index_type=None, data_access_protocol="file",
        num_query_processes=1, batch_size=1, runtime=60, queries=None, report_count=100,
        benchmark_mode="timed", vector_dim=1536, search_limit=10, search_ef=200, gt_collection=None,
        num_query_vectors=1000, recall_k=None, seed=42, distributed=False, hosts=None, npernode=1,
        storage_root=None, storage_type=None,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


DEFAULT_YAML = {
    "database": {"host": "127.0.0.1", "port": 19530, "database": "milvus"},
    "dataset": {"collection_name": "mlps_1m", "num_vectors": 1000000, "dimension": 1536,
                "distribution": "uniform", "batch_size": 10, "chunk_size": 100000, "num_shards": 1,
                "vector_dtype": "FLOAT_VECTOR"},
    "index": {"index_type": "DISKANN", "metric_type": "COSINE",
              "index_params": {"max_degree": 64, "search_list_size": 200}},
    "workflow": {"compact": True},
    "benchmark": {"mode": "timed", "runtime": 60, "batch_size": 1, "report_count": 100,
                  "recall_k": 10, "search_limit": 10},
    "storage": {"storage_root": None, "storage_type": "local_fs"},
}


def _vdb_benchmark(args, tmp_path, yaml_params=None):
    from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark
    with patch("mlpstorage_py.benchmarks.base.generate_output_location") as gen, \
         patch("mlpstorage_py.benchmarks.vectordbbench.read_config_from_file",
               return_value=copy.deepcopy(yaml_params) if yaml_params is not None else {}), \
         patch("mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark"), \
         patch("mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark._validate_vdb_dependencies"):
        gen.return_value = str(tmp_path / "output")
        return VectorDBBenchmark(args)


class TestVDBRuntimeBlock:
    def test_run_block_comes_from_the_yaml_with_cli_run_knobs(self, tmp_path):
        params = _vdb_benchmark(_vdb_args(tmp_path, "run", vdb_index="DISKANN"), tmp_path, DEFAULT_YAML).metadata["parameters"]
        assert params["database"]["database"] == "milvus"
        assert params["index"]["index_type"] == "DISKANN"
        assert params["dataset"]["num_vectors"] == 1000000 and params["dataset"]["dimension"] == 1536
        assert params["benchmark"]["recall_k"] == 10 and params["benchmark"]["search_limit"] == 10
        assert params["benchmark"]["runtime"] == 60 and params["benchmark"]["mode"] == "timed"
        # the reportgen identity keys stay
        assert params["engine"] == "milvus" and params["index_type"] == "DISKANN"
        assert compute_core_config(params, "vector_database")["hash"] == VDB_RUN_HASH

    def test_cli_overrides_the_yaml(self, tmp_path):
        args = _vdb_args(tmp_path, "run", vdb_index="HNSW", vector_dim=768, search_limit=100, recall_k=5)
        params = _vdb_benchmark(args, tmp_path, DEFAULT_YAML).metadata["parameters"]
        assert params["index"]["index_type"] == "HNSW"
        assert params["dataset"]["dimension"] == 768
        assert params["benchmark"] ["search_limit"] == 100 and params["benchmark"]["recall_k"] == 5

    def test_recall_k_defaults_to_search_limit(self, tmp_path):
        """The tool itself passes ``recall_k or search_limit`` to the bench."""
        args = _vdb_args(tmp_path, "run", vdb_index="DISKANN", search_limit=20, recall_k=None)
        params = _vdb_benchmark(args, tmp_path, {}).metadata["parameters"]
        assert params["benchmark"]["recall_k"] == 20 and params["benchmark"]["search_limit"] == 20

    def test_datagen_block_comes_from_the_cli(self, tmp_path):
        args = _vdb_args(tmp_path, "datagen", vdb_index="HNSW", num_vectors=5000000, dimension=768,
                         distribution="uniform", num_shards=2, vector_dtype="FLOAT_VECTOR",
                         chunk_size=10000, batch_size=1000, metric_type="L2")
        params = _vdb_benchmark(args, tmp_path, DEFAULT_YAML).metadata["parameters"]
        assert params["dataset"]["num_vectors"] == 5000000 and params["dataset"]["dimension"] == 768
        assert params["index"]["index_type"] == "HNSW"
        assert params["database"]["database"] == "milvus"
        assert "benchmark" not in params
        assert params["num_vectors"] == 5000000  # reportgen identity key unchanged

    def test_empty_yaml_still_records_the_cli(self, tmp_path):
        params = _vdb_benchmark(_vdb_args(tmp_path, "run", vdb_index="DISKANN"), tmp_path, {}).metadata["parameters"]
        assert params["database"] == {"database": "milvus"}
        assert params["index"]["index_type"] == "DISKANN"
        assert "num_vectors" not in params["dataset"]  # a bare run does not know the scale
        assert params["dataset"]["dimension"] == 1536
        assert compute_core_config(params, "vector_database")["hash"] != VDB_RUN_HASH

    def test_nulls_are_not_recorded(self, tmp_path):
        params = _vdb_benchmark(_vdb_args(tmp_path, "run", vdb_index="DISKANN"), tmp_path, {}).metadata["parameters"]
        for section in ("database", "index", "dataset", "benchmark"):
            assert None not in params[section].values(), section


# ---------------------------------------------------------------------------
# vector_database: reconstruction from a v3.0 leaf
# ---------------------------------------------------------------------------

TTA_SUMMARY = {"num_vectors": 1000000, "dimension": 1536, "inserted_vectors": 1000000, "index_type": "DISKANN",
               "database": {"database": "milvus"}, "throughput_qps": 57619.79, "total_time_seconds": 62.6,
               "query_count": 3607200, "mean_latency_ms": 6.29, "p95_latency_ms": 19.6, "p99_latency_ms": 107.3,
               "p999_latency_ms": 160.0, "recall": 0.209675}
TTA_CMD = ("mpirun -n 4 -host a:1,b:1 --npernode 1 uv run vdb-mpi-wrapper simple --base-output-dir /x/vectordb/simple "
           "--expected-ranks 4 --seed 42 -- --config /opt/storage/configs/vectordbbench/default.yaml --host 127.0.0.1 "
           "--port 19530 --collection-name mlpsc_1m_diskann --processes 128 --batch-size 100 --report-count 100 "
           "--vector-dim 1536 --search-limit 10 --search-ef 100 --num-query-vectors 1000 --json-output --runtime 60")
SAMSUNG_CONFIG = {"timestamp": "2026-07-21T10:29:53", "processes": 1, "batch_size": 1, "report_count": 100,
                  "vector_dim": 1536, "host": "127.0.0.1", "port": "19530", "collection_name": "mlps_diskann",
                  "runtime_seconds": 60, "total_queries": None, "search_limit": 10, "search_ef": 200,
                  "gt_collection": None, "num_query_vectors": 1000, "no_create_flat": False, "seed": 42,
                  "data_path": None, "recall_k": 10, "metric_type": "COSINE", "index_type": "DISKANN",
                  "search_params": {"metric_type": "COSINE", "params": {"search_list": 200}}}
SAMSUNG_VERDICT = {"result": "valid", "valid": True, "num_queries_evaluated": 1000,
                   "flat_setup": {"ok": True, "coverage": 1.0, "total_vectors": 1000000, "copied_vectors": 1000000,
                                  "had_recoverable_error": False, "reason": "", "reused": True}}


class TestVDBReconstruction:
    def test_mpi_leaf_with_summary_json(self, tmp_path):
        root = _root(tmp_path)
        leaf = _vdb_leaf(root, LEAF_CLOSED_VDB, executed_command=TTA_CMD,
                         args={"search_limit": 10, "recall_k": None, "batch_size": 100, "vector_dim": 1536},
                         files={"summary.json": TTA_SUMMARY})
        stamp = derive_leaf_provenance(leaf, root)
        assert stamp.core_config["allowlist"] == "vector_database@1"
        assert stamp.core_config["hash"] == VDB_RUN_HASH

    def test_single_node_leaf_with_config_json_and_verdict(self, tmp_path):
        root = _root(tmp_path)
        leaf = _vdb_leaf(root, LEAF_CLOSED_VDB, args={"batch_size": 1},
                         files={"config.json": SAMSUNG_CONFIG, "result_verdict.json": SAMSUNG_VERDICT,
                                "recall_stats.json": {"recall_at_k": 0.5}})
        assert derive_leaf_provenance(leaf, root).core_config["hash"] == VDB_RUN_HASH

    def test_workload_parameters_rebuilds_the_sections(self, tmp_path):
        leaf = _vdb_leaf(_root(tmp_path), LEAF_CLOSED_VDB, executed_command=TTA_CMD,
                         files={"summary.json": TTA_SUMMARY})
        md = json.loads(next(leaf.glob("*_metadata.json")).read_text())
        block = workload_parameters("vector_database", md, leaf)
        assert block["database"] == {"database": "milvus"}
        assert block["index"] == {"index_type": "DISKANN"}
        assert block["dataset"] == {"num_vectors": 1000000, "dimension": 1536}
        assert block["benchmark"] == {"recall_k": 10, "search_limit": 10}

    def test_recorded_outputs_win_over_the_args(self, tmp_path):
        """summary.json says what the collection held; args say what was typed."""
        leaf = _vdb_leaf(_root(tmp_path), LEAF_CLOSED_VDB, executed_command=TTA_CMD,
                         args={"vdb_index": "HNSW"}, files={"summary.json": dict(TTA_SUMMARY, num_vectors=10000000)})
        md = json.loads(next(leaf.glob("*_metadata.json")).read_text())
        block = workload_parameters("vector_database", md, leaf)
        assert block["dataset"]["num_vectors"] == 10000000
        assert block["index"]["index_type"] == "DISKANN"

    def test_run_leaf_missing_the_scale_stays_unknown(self, tmp_path):
        root = _root(tmp_path)
        leaf = _vdb_leaf(root, LEAF_CLOSED_VDB, executed_command=TTA_CMD)  # no summary, no config.json
        cc = derive_leaf_provenance(leaf, root).core_config
        assert cc["hash"] == UNKNOWN and cc["allowlist"] == "vector_database@1"

    def test_run_leaf_missing_the_recall_target_stays_unknown(self, tmp_path):
        root = _root(tmp_path)
        leaf = _vdb_leaf(root, LEAF_CLOSED_VDB, files={"summary.json": TTA_SUMMARY})  # no argv, no config.json
        assert derive_leaf_provenance(leaf, root).core_config["hash"] == UNKNOWN

    def test_datagen_leaf_hashes_the_load_keys(self, tmp_path):
        root = _root(tmp_path)
        leaf = _vdb_leaf(root, LEAF_CLOSED_VDB_DATAGEN, command="datagen",
                         args={"num_vectors": 1000000, "dimension": 1536, "index_type": "DISKANN",
                               "distribution": "uniform"})
        cc = derive_leaf_provenance(leaf, root).core_config
        assert cc["hash"] == VDB_DATAGEN_HASH

    def test_a_structured_block_is_used_as_is(self, tmp_path):
        root = _root(tmp_path)
        leaf = _vdb_leaf(root, LEAF_CLOSED_VDB, parameters={"engine": "milvus", **VDB_RUN_PARAMETERS})
        assert derive_leaf_provenance(leaf, root).core_config["hash"] == VDB_RUN_HASH


# ---------------------------------------------------------------------------
# The classes
# ---------------------------------------------------------------------------

class TestClasses:
    def test_kvcache_closed_class(self):
        t = _table()
        c = next(c for c in t.classes if c.id == "llama3.1-8b-A")
        assert (c.division, c.family, c.model, c.accelerator, c.allowlist) == \
               ("closed", "kv_cache", "llama3.1-8b", "any", "kv_cache@1")
        assert c.core_configs == (KV_CLOSED_HASH,) and c.editions == ("3.0",)
        for acc in (None, "unknown", "b200"):
            assert t.classify(division="closed", family="kv_cache", model="llama3.1-8b", accelerator=acc,
                              core_config=KV_CLOSED_HASH, edition="3.0") is c

    def test_kvcache_class_equals_the_runtime_table(self, tmp_path):
        """Table KVCache-1 lives in kvcache.py WORKLOAD_PARAMS; the class must
        follow it (the kv_cache analogue of the shipped-template check)."""
        params = _kv_benchmark(_kv_args(tmp_path), tmp_path).metadata["parameters"]
        c = _table().classify(division="closed", family="kv_cache", model="llama3.1-8b", accelerator=None,
                              core_config=compute_core_config(params, "kv_cache")["hash"], edition="3.0")
        assert c is not None and c.id == "llama3.1-8b-A"

    def test_vdb_closed_class(self):
        t = _table()
        c = next(c for c in t.classes if c.id == "milvus-diskann-A")
        assert (c.division, c.family, c.model, c.accelerator, c.allowlist) == \
               ("closed", "vector_database", "milvus/DISKANN", "any", "vector_database@1")
        assert c.core_configs == (VDB_RUN_HASH,) and c.editions == ("3.0",)
        assert c.reference == "configs/vectordbbench/default.yaml"
        assert t.classify(division="closed", family="vector_database", model="milvus/DISKANN", accelerator=None,
                          core_config=VDB_RUN_HASH, edition="3.0") is c

    def test_open_runs_have_no_class(self):
        t = _table()
        assert t.classify(division="open", family="kv_cache", model="llama3.1-8b", accelerator=None,
                          core_config=KV_CLOSED_HASH, edition="3.0") is None
        assert t.classify(division="open", family="vector_database", model="milvus/DISKANN", accelerator=None,
                          core_config=VDB_RUN_HASH, edition="3.0") is None

    def test_class_families_cover_every_family(self):
        from mlpstorage_py.editions import CLASS_FAMILIES
        assert set(CLASS_FAMILIES) == {"training", "checkpointing", "vector_database", "kv_cache"}


# ---------------------------------------------------------------------------
# EDN-02 and runs show
# ---------------------------------------------------------------------------

class TestEnforcement:
    def test_closed_kv_run_in_the_class_is_silent(self, tmp_path):
        root = _root(tmp_path)
        _kv_leaf(root, LEAF_CLOSED_KV, parameters=KV_CLOSED_PARAMETERS, stamp=_kv_stamp(), declare=True)
        check, log = _check(root)
        assert check() is True
        assert _rule_lines(log.lines) == []

    def test_closed_kv_variant_is_an_error(self, tmp_path):
        root = _root(tmp_path)
        _kv_leaf(root, LEAF_CLOSED_KV, parameters=KV_CLOSED_PARAMETERS, stamp=_kv_stamp("f" * 16), declare=True)
        check, log = _check(root)
        assert check() is False
        assert len(log.errors) == 1
        assert log.errors[0].startswith("[EDN-02 comparabilityClass] ")
        assert "kv_cache/llama3.1-8b" in log.errors[0] and "kv_cache@1" in log.errors[0]

    def test_closed_vdb_run_in_the_class_is_silent_and_a_variant_errors(self, tmp_path):
        root = _root(tmp_path)
        _vdb_leaf(root, LEAF_CLOSED_VDB, parameters=VDB_RUN_PARAMETERS, stamp=_vdb_stamp(), declare=True)
        check, log = _check(root)
        assert check() is True and _rule_lines(log.lines) == []
        _vdb_leaf(root, LEAF_CLOSED_VDB, parameters=VDB_RUN_PARAMETERS, stamp=_vdb_stamp("e" * 16), declare=True)
        check, log = _check(root)
        assert check() is False
        assert len(log.errors) == 1 and "vector_database/milvus/DISKANN" in log.errors[0]

    def test_unknown_hash_is_still_skipped(self, tmp_path):
        root = _root(tmp_path)
        _kv_leaf(root, LEAF_CLOSED_KV, parameters=KV_V30_PLACEHOLDER, stamp=_kv_stamp(UNKNOWN), declare=True)
        check, log = _check(root)
        assert check() is True and _rule_lines(log.lines) == []

    def test_runs_show_prints_the_vdb_class_for_a_derived_stamp(self, tree, capsys):
        leaf = Path(tree) / LEAF_VDB
        md = next(leaf.glob("*_metadata.json"))
        body = json.loads(md.read_text())
        body["parameters"] = {"engine": "milvus", **VDB_RUN_PARAMETERS}
        md.write_text(json.dumps(body, indent=2))
        assert _main(["runs", "show", "5"]) == EXIT_CODE.SUCCESS
        out = capsys.readouterr().out
        assert "provenance:  derived" in out
        assert re.search(r"core config:\s+" + VDB_RUN_HASH + r" \(vector_database@1\)", out)
        assert re.search(r"class:\s+milvus-diskann-A \(matched by hash", out)


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

class TestDocs:
    def test_rules_md_names_the_allowlists(self):
        text = (PROJECT_ROOT / "Rules.md").read_text()
        assert "kv_cache@1" in text and "vector_database@1" in text
        assert "Families without an allowlist (kv_cache, vector_database)" not in text
        assert "Families without a core-config allowlist (kv_cache, vector_database) have no classes yet" not in text

    def test_manpage_names_the_allowlists(self):
        text = (PROJECT_ROOT / "ManPage.md").read_text()
        assert "kv_cache@1" in text and "vector_database@1" in text
        assert "`kv_cache` / `vector_database` stamp `unknown`" not in text

    def test_core_config_keys_yaml_explains_the_two_allowlists(self):
        text = (PROJECT_ROOT / "mlpstorage_py/rules/core_config_keys.yaml").read_text()
        assert "Table KVCache-1" in text and "5.6.4" in text
        assert "have no allowlist" not in text
