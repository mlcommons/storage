"""Column-parity: results.{csv,json} is a FIXED webpage-parity schema.

The v3.0 user-visible results web page is 8 tables (2 training + 4
checkpointing + 1 kvcache + 1 vdb) sharing a System-Under-Test block.
``reports reportgen`` emits ONE flat ``results.csv`` / ``results.json``
that a MLCommons staff member opens in Excel and reduces — by deleting
the workload blocks + discriminator columns that don't apply — to any one
of those 8 tables.

That requires the emitted file to be a FIXED schema (the exact reference
columns + the 3 agreed discriminators), NOT a data-driven set of columns.
This file pins that contract at the emitted-file boundary:

- The header is EXACTLY ``FINAL_SCHEMA`` (order + membership), regardless
  of which metric keys a given run's summary.json happened to contain.
- Every JSON row dict carries EXACTLY those keys, in order.
- The internal machine columns (``category`` / ``orgname`` /
  ``systemname`` / ``benchmark_type`` prefix, ``*_mean_of_*`` dynamic
  metric-mean columns, trailing ``issues``) do NOT appear.
- Values are cherry-picked from the in-memory row into the right fixed
  column: Division = CLOSED/OPEN, Model = display label, kvcache option
  N routed to its fixed group, etc.

This is the write-layer analog of the internal-structure tests in
tests/unit/test_aggregation.py — those pin the ``_aggregate_*`` machine
keys (UNCHANGED by this task); this pins the projected output file.
"""

from __future__ import annotations

import csv
import json
import pathlib
from typing import Any, Dict, List
from unittest.mock import patch

from mlpstorage_py.report_generator import ReportGenerator, Hyperlink


# --------------------------------------------------------------------------- #
# The authoritative fixed schema (Results Table Structure.xlsx, v3.0 +        #
# agreed Division/Benchmark Type/Model discriminators). 54 columns.           #
# --------------------------------------------------------------------------- #

FINAL_SCHEMA: List[str] = [
    # Left edge + shared SUT block (13)
    "Public ID",
    "Organization",
    "Division",
    "Benchmark Type",
    "Model",
    "Name",
    "Description",
    "Type",
    "Access Protocol",
    "Availability",
    "RU's",
    "Integrated Client Storage (TiB)",
    "Usable Capacity (TiB)",
    # Training block (6)
    "Training - Accelerator Type",
    "Training - # Client Nodes",
    "Training - Code",
    "Training - Logs",
    "Training - # Simulated Accelerators",
    "Training - Read B/W (GiB/s)",
    # Checkpointing block (9)
    "Checkpointing - Checkpoint Mode",
    "Checkpointing - # Client Nodes",
    "Checkpointing - DP Instances",
    "Checkpointing - Code",
    "Checkpointing - Logs",
    "Checkpointing - Write B/W (GiB/s)",
    "Checkpointing - Write Duration (secs)",
    "Checkpointing - Read B/W (GiB/s)",
    "Checkpointing - Read Duration (secs)",
    # VDB block (11)
    "VDB - # Client Nodes",
    "VDB - Code",
    "VDB - Logs",
    "VDB - Vector Count",
    "VDB - Vector Dimension",
    "VDB - Index Type",
    "VDB - Queries per Sec",
    "VDB - Query Latency (ms)",
    "VDB - Recall Percentage",
    "VDB - Storage IOPs",
    "VDB - Read B/W (GiB/s)",
    # KVCache block (3 shared + 3 groups x 4 = 15)
    "KVCache - # Client Nodes",
    "KVCache - Code",
    "KVCache - Logs",
    "KVCache llama3.1-8b Storage Only - Throughput (tok/s)",
    "KVCache llama3.1-8b Storage Only - Read B/W (GiB/s)",
    "KVCache llama3.1-8b Storage Only - Write B/W (GiB/s)",
    "KVCache llama3.1-8b Storage Only - P95 Read Latency (ms)",
    "KVCache llama3.1-8b Storage + Mem - Throughput (tok/s)",
    "KVCache llama3.1-8b Storage + Mem - Read B/W (GiB/s)",
    "KVCache llama3.1-8b Storage + Mem - Write B/W (GiB/s)",
    "KVCache llama3.1-8b Storage + Mem - P95 Read Latency (ms)",
    "KVCache llama3.1-70b Storage Only - Throughput (tok/s)",
    "KVCache llama3.1-70b Storage Only - Read B/W (GiB/s)",
    "KVCache llama3.1-70b Storage Only - Write B/W (GiB/s)",
    "KVCache llama3.1-70b Storage Only - P95 Read Latency (ms)",
]


def _bare_generator(tmp_path: pathlib.Path) -> ReportGenerator:
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)
    with patch.object(ReportGenerator, "accumulate_results"):
        with patch.object(ReportGenerator, "print_results"):
            return ReportGenerator(str(results_dir), validate_structure=False)


def _internal_rows() -> List[Dict[str, Any]]:
    """Internal (machine-key) rows exactly as ``_workload_result_to_row``
    produces them — one per workload type. The projection under test must
    map these onto ``FINAL_SCHEMA`` and drop everything else.
    """
    sut = {
        "sut_public_id": "",
        "sut_organization": "acme",
        "sut_name": Hyperlink("system-a", "closed/acme/systems/system-a.yaml"),
        "sut_description": Hyperlink("PDF", "closed/acme/systems/system-a.pdf"),
        "sut_type": "",
        "sut_access_protocol": "",
        "sut_availability": "",
        "sut_rus": 14,
        "sut_integrated_client_storage": "",
        "sut_usable_capacity_tib": "",
        "sut_code": Hyperlink("code", "acme/code-abc12345/"),
        "sut_logs": Hyperlink("logs", "acme/code-abc12345/"),
    }
    common = {
        "category": "closed",
        "orgname": "acme",
        "systemname": "system-a",
    }
    training = {
        **common,
        "benchmark_type": "training",
        "model": "unet3d",
        "accelerator": "h100",
        **sut,
        # dynamic metric-mean columns that MUST be dropped:
        "train_mean_of_au_percentage": 95.0,
        "train_mean_of_throughput_samples_per_second": 1250.0,
        # final-table metric keys:
        "train_num_client_nodes": 8,
        "train_num_simulated_accelerators": 16,
        "train_read_bw_gibps": 12.5,
        "issues": "",
    }
    checkpointing = {
        **common,
        "benchmark_type": "checkpointing",
        "model": "llama3-1t",
        "accelerator": "",
        **sut,
        "checkpoint_mean_of_read_throughput_GB_per_second": 12.4,
        "checkpoint_num_client_nodes": 4,
        "checkpoint_mode": "Full",
        "checkpoint_dp_instances": 2,
        "checkpoint_write_bw_gibps": 8.8,
        "checkpoint_write_duration_secs": 30.0,
        "checkpoint_read_bw_gibps": 9.1,
        "checkpoint_read_duration_secs": 27.5,
        "issues": "",
    }
    vdb = {
        **common,
        "benchmark_type": "vector_database",
        "model": "",
        "accelerator": "",
        **sut,
        "vdb_num_client_nodes": 3,
        "vdb_num_vectors": 1000000,
        "vdb_dimension": 768,
        "vdb_index_type": "hnsw",
        "vdb_throughput_qps": 4200.0,
        "vdb_p99_latency_ms": 1.7,
        "vdb_recall": 98.5,
        "vdb_storage_iops": None,
        "vdb_read_bw_gibps": 5.5,
        "issues": "",
    }
    kvcache = {
        **common,
        "benchmark_type": "kv_cache",
        "model": "llama3.1-8b",
        "accelerator": "",
        **sut,
        "kvcache_num_client_nodes": 2,
        "kvcache_option_1_aggregated_avg_throughput_tokens_per_sec": 111.0,
        "kvcache_option_1_aggregated_read_bandwidth_gbps": 11.1,
        "kvcache_option_1_aggregated_write_bandwidth_gbps": 1.11,
        "kvcache_option_1_aggregated_p95_latency_ms": 0.11,
        "kvcache_option_1_aggregated_storage_read_p95_ms": 45.6,
        "kvcache_option_2_aggregated_avg_throughput_tokens_per_sec": 222.0,
        "kvcache_option_2_aggregated_read_bandwidth_gbps": 22.2,
        "kvcache_option_2_aggregated_write_bandwidth_gbps": 2.22,
        "kvcache_option_2_aggregated_p95_latency_ms": 0.22,
        "kvcache_option_2_aggregated_storage_read_p95_ms": 7.8,
        "kvcache_option_3_aggregated_avg_throughput_tokens_per_sec": 333.0,
        "kvcache_option_3_aggregated_read_bandwidth_gbps": 33.3,
        "kvcache_option_3_aggregated_write_bandwidth_gbps": 3.33,
        "kvcache_option_3_aggregated_p95_latency_ms": 0.33,
        "kvcache_option_3_aggregated_storage_read_p95_ms": 91.2,
        "issues": "",
    }
    return [training, checkpointing, vdb, kvcache]


class TestFixedSchemaHeader:
    def test_csv_header_is_exactly_final_schema(self, tmp_path):
        gen = _bare_generator(tmp_path)
        out = tmp_path / "csv_out"
        out.mkdir()
        gen.write_csv_file(_internal_rows(), target_dir=str(out))
        with open(out / "results.csv", newline="") as fh:
            header = next(csv.reader(fh))
        assert header == FINAL_SCHEMA, (
            "results.csv header must be EXACTLY the fixed webpage-parity "
            f"schema.\nExpected: {FINAL_SCHEMA}\nGot:      {header}"
        )

    def test_empty_input_still_emits_full_fixed_header(self, tmp_path):
        gen = _bare_generator(tmp_path)
        out = tmp_path / "empty_out"
        out.mkdir()
        gen.write_csv_file([], target_dir=str(out))
        with open(out / "results.csv", newline="") as fh:
            header = next(csv.reader(fh))
        assert header == FINAL_SCHEMA

    def test_json_keys_are_exactly_final_schema_in_order(self, tmp_path):
        gen = _bare_generator(tmp_path)
        out = tmp_path / "json_out"
        out.mkdir()
        gen.write_json_file(_internal_rows(), target_dir=str(out))
        rows = json.loads((out / "results.json").read_text())
        assert len(rows) == 4
        for row in rows:
            assert list(row.keys()) == FINAL_SCHEMA


class TestInternalColumnsRemoved:
    def test_machine_and_dynamic_columns_absent(self, tmp_path):
        gen = _bare_generator(tmp_path)
        out = tmp_path / "absent_out"
        out.mkdir()
        gen.write_csv_file(_internal_rows(), target_dir=str(out))
        with open(out / "results.csv", newline="") as fh:
            header = set(next(csv.reader(fh)))
        for banned in (
            "category",
            "orgname",
            "systemname",
            "benchmark_type",
            "model",
            "accelerator",
            "issues",
            "sut_organization",
            "train_mean_of_au_percentage",
            "checkpoint_mean_of_read_throughput_GB_per_second",
        ):
            assert banned not in header, f"{banned!r} must not appear in results.csv"


class TestValueProjection:
    def _json_rows(self, tmp_path):
        gen = _bare_generator(tmp_path)
        out = tmp_path / "vp_out"
        out.mkdir()
        gen.write_json_file(_internal_rows(), target_dir=str(out))
        return json.loads((out / "results.json").read_text())

    def test_training_row_values(self, tmp_path):
        row = self._json_rows(tmp_path)[0]
        assert row["Division"] == "CLOSED"
        assert row["Benchmark Type"] == "training"
        assert row["Model"] == "Unet3D"
        assert row["Organization"] == "acme"
        assert row["RU's"] == 14
        assert row["Training - Accelerator Type"] == "h100"
        assert row["Training - # Client Nodes"] == 8
        assert row["Training - # Simulated Accelerators"] == 16
        assert row["Training - Read B/W (GiB/s)"] == 12.5
        # Code/Logs live in the workload block, populated for this row.
        assert row["Training - Code"] == {
            "text": "code", "href": "acme/code-abc12345/"}
        # Other workloads' blocks are blank for a training row.
        assert row["Checkpointing - Write B/W (GiB/s)"] == ""
        assert row["VDB - Vector Count"] == ""

    def test_checkpointing_row_values(self, tmp_path):
        row = self._json_rows(tmp_path)[1]
        assert row["Benchmark Type"] == "checkpointing"
        assert row["Model"] == "1250B"  # llama3-1t display label
        assert row["Checkpointing - Checkpoint Mode"] == "Full"
        assert row["Checkpointing - # Client Nodes"] == 4
        assert row["Checkpointing - DP Instances"] == 2
        assert row["Checkpointing - Write B/W (GiB/s)"] == 8.8
        assert row["Checkpointing - Read Duration (secs)"] == 27.5

    def test_vdb_row_values(self, tmp_path):
        row = self._json_rows(tmp_path)[2]
        assert row["Benchmark Type"] == "vector_database"
        assert row["Model"] == ""  # blank for vdb
        assert row["VDB - Vector Count"] == 1000000
        assert row["VDB - Index Type"] == "hnsw"
        assert row["VDB - Query Latency (ms)"] == 1.7
        assert row["VDB - Recall Percentage"] == 98.5
        assert row["VDB - Storage IOPs"] in ("", None)
        assert row["VDB - Read B/W (GiB/s)"] == 5.5

    def test_kvcache_group_option_routing(self, tmp_path):
        row = self._json_rows(tmp_path)[3]
        assert row["Benchmark Type"] == "kv_cache"
        assert row["Model"] == ""  # blank for kvcache
        assert row["KVCache - # Client Nodes"] == 2
        # option 1 -> Storage Only, option 2 -> Storage + Mem, option 3 -> 70b
        assert row["KVCache llama3.1-8b Storage Only - Throughput (tok/s)"] == 111.0
        assert row["KVCache llama3.1-8b Storage Only - Read B/W (GiB/s)"] == 11.1
        assert row["KVCache llama3.1-8b Storage + Mem - Throughput (tok/s)"] == 222.0
        assert row["KVCache llama3.1-70b Storage Only - Throughput (tok/s)"] == 333.0
        # R5: the P95 Read Latency column sources the per-IO read
        # percentile (aggregated_storage_read_p95_ms), NOT the
        # per-request cumulative storage-I/O total that
        # aggregated_p95_latency_ms actually measures. Both keys are
        # planted with different values to pin the precedence.
        assert row["KVCache llama3.1-8b Storage Only - P95 Read Latency (ms)"] == 45.6
        assert row["KVCache llama3.1-8b Storage + Mem - P95 Read Latency (ms)"] == 7.8
        assert row["KVCache llama3.1-70b Storage Only - P95 Read Latency (ms)"] == 91.2
