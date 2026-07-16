"""
Regression tests for issue #801:

    Storage throughput double-counted for a single NVMe drive.

``/proc/diskstats`` reports I/O counters for both a whole block device
(e.g. ``nvme0n1``) and each of its partitions (``nvme0n1p1``) with
identical values. ``build_disk_io_stats`` — and the sibling aggregation
paths in ``enhanced_bench`` — naively summed every entry, so a single
physical drive whose only active partition holds the data was counted
twice, inflating ``total_bytes_read`` / ``total_bytes_read_per_sec`` by
~2x for single-drive configs (worse with more partitions).

The fix collapses whole-disk/partition duplicates before summing: when a
partition's parent whole-disk device is present in the same snapshot the
parent is kept (its counters already cover all partition I/O) and the
partition is dropped; an orphan partition with no present parent is kept.

These tests need neither a live Milvus nor pymilvus — ``disk_stats``
depends only on the standard library.
"""
import os
import sys

# Make the vdbbench package importable regardless of pytest's invocation dir.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vdbbench.disk_stats import (  # noqa: E402
    build_disk_io_stats,
    classify_storage_target,
    collapse_partition_duplicates,
    parent_disk,
)

MOUNTS_LOCAL = [
    {"source": "/dev/nvme0n1p2", "mountpoint": "/", "fstype": "ext4"},
    {"source": "/dev/nvme1n1", "mountpoint": "/var/lib/milvus", "fstype": "xfs"},
]


def _fmt(b):
    return f"{b} B"


def _local_target():
    return classify_storage_target("/var/lib/milvus", mounts=MOUNTS_LOCAL)


# --- parent_disk --------------------------------------------------------------

def test_parent_disk_nvme_partition():
    assert parent_disk("nvme0n1p1") == "nvme0n1"
    assert parent_disk("nvme2n1p2") == "nvme2n1"


def test_parent_disk_scsi_and_mmc_partition():
    assert parent_disk("sda1") == "sda"
    assert parent_disk("vdb2") == "vdb"
    assert parent_disk("xvda1") == "xvda"
    assert parent_disk("mmcblk0p1") == "mmcblk0"


def test_parent_disk_whole_disks_and_virtual_return_none():
    for dev in ("nvme0n1", "sda", "vdb", "mmcblk0", "dm-0", "md0"):
        assert parent_disk(dev) is None, dev


# --- collapse_partition_duplicates -------------------------------------------

def test_collapse_drops_partition_keeps_parent():
    diff = {
        "nvme0n1":   {"bytes_read": 100, "bytes_written": 0},
        "nvme0n1p1": {"bytes_read": 100, "bytes_written": 0},
    }
    out = collapse_partition_duplicates(diff)
    assert set(out) == {"nvme0n1"}


def test_collapse_retains_orphan_partition():
    # Parent not present in the snapshot -> partition must be kept.
    diff = {"nvme0n1p1": {"bytes_read": 100, "bytes_written": 0}}
    out = collapse_partition_duplicates(diff)
    assert set(out) == {"nvme0n1p1"}


def test_collapse_keeps_independent_whole_disks():
    diff = {
        "nvme0n1": {"bytes_read": 1, "bytes_written": 0},
        "nvme1n1": {"bytes_read": 2, "bytes_written": 0},
        "sda":     {"bytes_read": 3, "bytes_written": 0},
    }
    assert collapse_partition_duplicates(diff) == diff


# --- build_disk_io_stats (end-to-end payload) --------------------------------

def test_single_drive_not_double_counted():
    # Exact numbers from the issue's DISKANN run 20260714_124702.
    diff = {
        "nvme0n1":   {"bytes_read": 560280633344, "bytes_written": 0},
        "nvme0n1p1": {"bytes_read": 560280633344, "bytes_written": 0},
    }
    p = build_disk_io_stats(diff, 300.0, _local_target(), _fmt)
    assert p["total_bytes_read"] == 560280633344          # not 2x
    assert p["total_bytes_read_per_sec"] == 560280633344 / 300.0
    assert "nvme0n1" in p["devices"]
    assert "nvme0n1p1" not in p["devices"]


def test_multiple_drives_each_deduped():
    diff = {
        "nvme2n1":   {"bytes_read": 1351680, "bytes_written": 0},
        "nvme2n1p2": {"bytes_read": 1351680, "bytes_written": 0},
        "nvme0n1":   {"bytes_read": 560280633344, "bytes_written": 0},
        "nvme0n1p1": {"bytes_read": 560280633344, "bytes_written": 0},
    }
    p = build_disk_io_stats(diff, 300.0, _local_target(), _fmt)
    assert p["total_bytes_read"] == 1351680 + 560280633344


def test_multiple_partitions_on_one_disk():
    diff = {
        "sda":  {"bytes_read": 4096, "bytes_written": 0},
        "sda1": {"bytes_read": 4096, "bytes_written": 0},
        "sda2": {"bytes_read": 4096, "bytes_written": 0},
    }
    p = build_disk_io_stats(diff, 10.0, _local_target(), _fmt)
    assert p["total_bytes_read"] == 4096                  # whole disk only


def test_orphan_partition_still_counted():
    diff = {"nvme0n1p1": {"bytes_read": 4096, "bytes_written": 0}}
    p = build_disk_io_stats(diff, 10.0, _local_target(), _fmt)
    assert p["total_bytes_read"] == 4096
    assert "nvme0n1p1" in p["devices"]
