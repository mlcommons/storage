"""Unit tests for vdbbench.disk_stats (issue #591)."""
import sys
sys.path.insert(0, ".")
from vdbbench.disk_stats import (
    build_disk_io_stats, classify_storage_target, find_mount_for_path,
    is_network_fs, list_network_mounts,
)

MOUNTS_NFS = [
    {"source": "/dev/nvme0n1p2", "mountpoint": "/", "fstype": "ext4"},
    {"source": "tmpfs", "mountpoint": "/run", "fstype": "tmpfs"},
    {"source": "filer:/export/vdb", "mountpoint": "/mnt/vdb", "fstype": "nfs4"},
]
MOUNTS_LOCAL = [
    {"source": "/dev/nvme0n1p2", "mountpoint": "/", "fstype": "ext4"},
    {"source": "/dev/nvme1n1", "mountpoint": "/var/lib/milvus", "fstype": "xfs"},
]

def fmt(b):
    return f"{b} B"

def test_is_network_fs():
    assert is_network_fs("nfs4")
    assert is_network_fs("cephfs")
    assert is_network_fs("lustre")
    assert is_network_fs("fuse.sshfs", "user@host:/data")
    assert not is_network_fs("ext4", "/dev/nvme0n1p2")
    assert not is_network_fs("xfs", "/dev/sda1")

def test_longest_prefix_match():
    m = find_mount_for_path("/mnt/vdb/milvus/data", MOUNTS_NFS)
    assert m and m["fstype"] == "nfs4"
    m = find_mount_for_path("/home/user", MOUNTS_NFS)
    assert m and m["mountpoint"] == "/"

def test_classify_nfs_target_not_applicable():
    r = classify_storage_target("/mnt/vdb/milvus", mounts=MOUNTS_NFS)
    assert r["applicable"] is False
    assert r["confidence"] == "exact"
    assert r["target_fstype"] == "nfs4"
    assert "network" in r["reason"]

def test_classify_local_target_applicable():
    r = classify_storage_target("/var/lib/milvus/data", mounts=MOUNTS_LOCAL)
    assert r["applicable"] is True
    assert r["confidence"] == "exact"
    assert r["target_fstype"] == "xfs"

def test_classify_heuristic_no_data_path():
    r = classify_storage_target(None, mounts=MOUNTS_NFS)
    assert r["applicable"] is True          # numbers still reported...
    assert r["confidence"] == "heuristic"   # ...but flagged as heuristic
    assert len(r["network_mounts"]) == 1

def test_payload_na_preserves_client_local_io():
    diff = {"nvme0n1": {"bytes_read": 4096, "bytes_written": 8192}}
    tgt = classify_storage_target("/mnt/vdb", mounts=MOUNTS_NFS)
    p = build_disk_io_stats(diff, 10.0, tgt, fmt)
    assert p["applicable"] is False and p["status"] == "N/A"
    assert "total_bytes_read" not in p            # no top-level counters
    assert p["client_local_io"]["total_bytes_read"] == 4096
    assert p["not_applicable_reason"]

def test_payload_applicable_keeps_legacy_fields():
    diff = {"nvme1n1": {"bytes_read": 1024, "bytes_written": 2048}}
    tgt = classify_storage_target("/var/lib/milvus", mounts=MOUNTS_LOCAL)
    p = build_disk_io_stats(diff, 10.0, tgt, fmt)
    assert p["applicable"] is True and p["status"] == "OK"
    assert p["total_bytes_read"] == 1024          # backward compatible
    assert p["total_bytes_read_per_sec"] == 102.4
    assert "nvme1n1" in p["devices"]

def test_payload_empty_diff():
    tgt = classify_storage_target(None, mounts=MOUNTS_LOCAL)
    p = build_disk_io_stats({}, 10.0, tgt, fmt)
    assert p["applicable"] is False and "error" in p

# --- issue #801: /proc/diskstats double-counts a whole disk and its partitions ---

def test_partition_not_double_counted():
    # /proc/diskstats reports a whole NVMe device and its data partition with
    # identical read counters; summing both inflates throughput ~2x.
    diff = {
        "nvme0n1":   {"bytes_read": 560280633344, "bytes_written": 0},
        "nvme0n1p1": {"bytes_read": 560280633344, "bytes_written": 0},
    }
    tgt = classify_storage_target("/var/lib/milvus", mounts=MOUNTS_LOCAL)
    p = build_disk_io_stats(diff, 300.0, tgt, fmt)
    assert p["total_bytes_read"] == 560280633344   # not 2x
    assert "nvme0n1" in p["devices"]               # parent kept
    assert "nvme0n1p1" not in p["devices"]         # partition collapsed

def test_multiple_disks_each_deduped():
    diff = {
        "nvme2n1":   {"bytes_read": 1351680, "bytes_written": 0},
        "nvme2n1p2": {"bytes_read": 1351680, "bytes_written": 0},
        "nvme0n1":   {"bytes_read": 560280633344, "bytes_written": 0},
        "nvme0n1p1": {"bytes_read": 560280633344, "bytes_written": 0},
    }
    tgt = classify_storage_target("/var/lib/milvus", mounts=MOUNTS_LOCAL)
    p = build_disk_io_stats(diff, 300.0, tgt, fmt)
    assert p["total_bytes_read"] == 1351680 + 560280633344

def test_scsi_partition_deduped():
    diff = {
        "sda":  {"bytes_read": 4096, "bytes_written": 0},
        "sda1": {"bytes_read": 4096, "bytes_written": 0},
        "sda2": {"bytes_read": 4096, "bytes_written": 0},
    }
    tgt = classify_storage_target("/var/lib/milvus", mounts=MOUNTS_LOCAL)
    p = build_disk_io_stats(diff, 10.0, tgt, fmt)
    assert p["total_bytes_read"] == 4096           # only the whole disk

def test_orphan_partition_retained():
    # A partition whose parent whole-disk is absent must still be counted.
    diff = {"nvme0n1p1": {"bytes_read": 4096, "bytes_written": 0}}
    tgt = classify_storage_target("/var/lib/milvus", mounts=MOUNTS_LOCAL)
    p = build_disk_io_stats(diff, 10.0, tgt, fmt)
    assert p["total_bytes_read"] == 4096
    assert "nvme0n1p1" in p["devices"]

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print(f"PASS {f.__name__}")
    print(f"\n{len(fns)} tests passed")

