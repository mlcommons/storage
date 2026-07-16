"""
Storage-target classification for disk_io reporting (issue #591).

The disk_io figures in statistics.json are derived from /proc/diskstats,
which only accounts for local block devices on the benchmark client node.
When the storage under test is a network / remote filesystem (e.g. NFS,
CephFS, Lustre, GPFS, PanFS), there is no corresponding local block device
and the diskstats deltas do not describe the storage under test.

This module implements option (b) from issue #591: detect network/remote
mounts, classify the storage target, and let the benchmarks mark disk_io
as N/A in statistics.json instead of emitting empty or misleading values.

The QPS score is unaffected: this is purely a reporting-clarity change.
"""

import re
from typing import Any, Dict, List, Optional

MOUNTS_FILE = "/proc/self/mounts"

# Partition-name patterns used to collapse /proc/diskstats double-counting
# (issue #801). Linux names a partition by appending its number to the whole
# disk, inserting a 'p' separator when the disk name already ends in a digit:
#   - digit-ending disks use 'p': nvme0n1 -> nvme0n1p1, mmcblk0 -> mmcblk0p1
#   - traditional letter-named disks append directly: sda -> sda1, vdb -> vdb2
# The direct-suffix form is anchored to the sd/vd/hd/xvd families so whole
# disks whose own name ends in a digit (mmcblk0, loop0, md0) are not mistaken
# for partitions.
_PSEP_PARTITION_RE = re.compile(r"^(.+\d)p\d+$")
_DIRECT_PARTITION_RE = re.compile(r"^((?:sd|vd|hd|xvd)[a-z]+)\d+$")

# Filesystem types that indicate the mount is network / remote-attached.
# /proc/diskstats has no local block device for these targets.
NETWORK_FS_TYPES = {
    "nfs",
    "nfs4",
    "cifs",
    "smb3",
    "smbfs",
    "ceph",
    "cephfs",
    "glusterfs",
    "lustre",
    "gpfs",
    "beegfs",
    "panfs",
    "ocfs2",
    "afs",
    "9p",
    "virtiofs",
    "davfs",
    "fuse.sshfs",
    "fuse.glusterfs",
    "fuse.ceph",
    "fuse.juicefs",
    "fuse.s3fs",
    "fuse.gcsfuse",
    "fuse.weka",
    "fuse.beeond",
}

# Pseudo / virtual filesystems that are neither local block storage nor
# the storage under test; skipped during mount scans.
PSEUDO_FS_TYPES = {
    "proc", "sysfs", "devtmpfs", "devpts", "tmpfs", "cgroup", "cgroup2",
    "securityfs", "pstore", "efivarfs", "bpf", "debugfs", "tracefs",
    "configfs", "fusectl", "mqueue", "hugetlbfs", "autofs", "binfmt_misc",
    "rpc_pipefs", "nsfs", "ramfs", "squashfs", "overlay",
}


def read_mounts(mounts_file: str = MOUNTS_FILE) -> List[Dict[str, str]]:
    """Parse the mount table into a list of {source, mountpoint, fstype}."""
    mounts: List[Dict[str, str]] = []

    try:
        with open(mounts_file, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                parts = line.split()
                if len(parts) < 3:
                    continue

                mounts.append(
                    {
                        # Octal escapes (e.g. \040 for space) per fstab(5).
                        "source": parts[0],
                        "mountpoint": parts[1]
                        .replace("\\040", " ")
                        .replace("\\011", "\t"),
                        "fstype": parts[2],
                    }
                )
    except OSError:
        return []

    return mounts


def is_network_fs(fstype: str, source: str = "") -> bool:
    """Return True if a mount's fstype/source indicates remote storage."""
    fstype_lower = (fstype or "").lower()

    if fstype_lower in NETWORK_FS_TYPES:
        return True

    # Any fuse.* client whose source looks remote (host:/export, //server,
    # or a URI scheme) is treated as network-attached.
    if fstype_lower.startswith("fuse"):
        if ":" in source or source.startswith("//") or "://" in source:
            return True

    return False


def find_mount_for_path(
    path: str,
    mounts: Optional[List[Dict[str, str]]] = None,
    resolve_symlinks: Optional[bool] = None,
) -> Optional[Dict[str, str]]:
    """Longest-prefix match of `path` against the mount table.

    Symlinks should only be resolved against the same filesystem the
    mount table came from. By default (resolve_symlinks=None), symlinks
    are resolved iff the live mount table is read here (mounts is None);
    callers holding a live-derived table can pass resolve_symlinks=True,
    while synthetic tables (e.g. unit tests) must not trigger resolution
    — macOS symlinks /var to /private/var, which would break matching
    against injected fixtures.
    """
    import os

    if resolve_symlinks is None:
        resolve_symlinks = mounts is None

    if mounts is None:
        mounts = read_mounts()

    if resolve_symlinks:
        try:
            real_path = os.path.realpath(path)
        except OSError:
            real_path = os.path.normpath(path)
    else:
        real_path = os.path.normpath(path)

    best: Optional[Dict[str, str]] = None
    best_len = -1

    for mount in mounts:
        mnt = mount["mountpoint"]
        if real_path == mnt or real_path.startswith(mnt.rstrip("/") + "/") or mnt == "/":
            if len(mnt) > best_len:
                best = mount
                best_len = len(mnt)

    return best


def list_network_mounts(
    mounts: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Return all network/remote mounts, excluding pseudo filesystems."""
    if mounts is None:
        mounts = read_mounts()

    return [
        mount
        for mount in mounts
        if mount["fstype"] not in PSEUDO_FS_TYPES
        and is_network_fs(mount["fstype"], mount["source"])
    ]


def classify_storage_target(
    data_path: Optional[str] = None,
    mounts: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Classify the storage under test for disk_io applicability.

    Args:
        data_path: Optional path to the storage under test as mounted on
            this node (e.g. the Milvus data directory or the NFS mount
            backing it). When provided, classification is authoritative
            for that path. When omitted, classification is best-effort
            based on whether any network mounts are present on the node.

    Returns:
        {
          "applicable":        bool  — diskstats describe the target,
          "confidence":        "exact" | "heuristic",
          "target_path":       str | None,
          "target_fstype":     str | None,
          "target_mountpoint": str | None,
          "target_source":     str | None,
          "network_mounts":    [ {source, mountpoint, fstype}, ... ],
          "reason":            str,
        }
    """
    live_mounts = mounts is None
    if live_mounts:
        mounts = read_mounts()

    network_mounts = list_network_mounts(mounts)

    result: Dict[str, Any] = {
        "applicable": True,
        "confidence": "heuristic",
        "target_path": data_path,
        "target_fstype": None,
        "target_mountpoint": None,
        "target_source": None,
        "network_mounts": network_mounts,
        "reason": "",
    }

    if data_path:
        mount = find_mount_for_path(
            data_path, mounts, resolve_symlinks=live_mounts
        )

        if mount is None:
            result["applicable"] = False
            result["confidence"] = "exact"
            result["reason"] = (
                f"--data-path {data_path!r} does not resolve to any mount "
                "on this node; disk_io from /proc/diskstats cannot be "
                "attributed to the storage under test."
            )
            return result

        result["confidence"] = "exact"
        result["target_fstype"] = mount["fstype"]
        result["target_mountpoint"] = mount["mountpoint"]
        result["target_source"] = mount["source"]

        if is_network_fs(mount["fstype"], mount["source"]):
            result["applicable"] = False
            result["reason"] = (
                f"storage under test is a network/remote filesystem "
                f"({mount['fstype']} mounted at {mount['mountpoint']} "
                f"from {mount['source']}); /proc/diskstats only accounts "
                "for local block devices, so disk_io is not applicable."
            )
        else:
            result["reason"] = (
                f"storage under test resolves to a local mount "
                f"({mount['fstype']} at {mount['mountpoint']}); "
                "disk_io from /proc/diskstats is applicable."
            )

        return result

    # No data path supplied: best-effort heuristic.
    if network_mounts:
        result["reason"] = (
            "no --data-path was supplied and network/remote mounts are "
            "present on this node; disk_io reflects client-local block "
            "devices only and may not describe the storage under test. "
            "Pass --data-path to classify the target exactly."
        )
    else:
        result["reason"] = (
            "no network/remote mounts detected on this node; disk_io "
            "reflects client-local block devices."
        )

    return result


def parent_disk(device: str) -> Optional[str]:
    """Return the whole-disk device a partition belongs to, else ``None``.

    ``/proc/diskstats`` lists both whole block devices (``nvme0n1``, ``sda``)
    and their partitions (``nvme0n1p1``, ``sda1``). This maps a partition name
    back to its parent so duplicate counters can be collapsed (issue #801).
    Whole disks and non-partition entries (``dm-0``, ``loop0``) return ``None``.
    """
    m = _PSEP_PARTITION_RE.match(device)
    if m:
        return m.group(1)
    m = _DIRECT_PARTITION_RE.match(device)
    if m:
        return m.group(1)
    return None


def collapse_partition_duplicates(
    disk_io_diff: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, int]]:
    """Drop partition entries whose parent whole-disk device is also present.

    ``/proc/diskstats`` reports I/O counters for a whole block device *and*
    separately for each of its partitions, so the same bytes are counted twice
    when every entry is summed (issue #801). When a partition's parent device
    appears in the same snapshot we keep the parent — whose counters already
    cover all partition I/O — and drop the partition. An orphan partition whose
    parent is absent is retained so its I/O is not lost.
    """
    present = set(disk_io_diff)
    return {
        device: stats
        for device, stats in disk_io_diff.items()
        if parent_disk(device) not in present
    }


def build_disk_io_stats(
    disk_io_diff: Dict[str, Dict[str, int]],
    duration_seconds: float,
    storage_target: Dict[str, Any],
    format_bytes_fn,
) -> Dict[str, Any]:
    """
    Assemble the statistics.json `disk_io` payload with applicability
    marking (issue #591, options a+b).

    When the storage target is a network/remote filesystem the payload is
    marked N/A; raw client-local counters are preserved under
    `client_local_io` so they are still auditable but cannot be mistaken
    for storage-under-test I/O.
    """
    applicable = bool(storage_target.get("applicable", True))

    scope_note = (
        "Derived from /proc/diskstats on the benchmark client node; "
        "covers local block devices only. Not meaningful for "
        "network-attached storage targets."
    )

    target_summary = {
        "applicable": applicable,
        "confidence": storage_target.get("confidence"),
        "data_path": storage_target.get("target_path"),
        "fstype": storage_target.get("target_fstype"),
        "mountpoint": storage_target.get("target_mountpoint"),
        "source": storage_target.get("target_source"),
        "network_mounts_detected": [
            f"{m['fstype']}:{m['mountpoint']}"
            for m in storage_target.get("network_mounts", [])
        ],
        "reason": storage_target.get("reason", ""),
    }

    if not disk_io_diff:
        return {
            "applicable": False,
            "status": "N/A",
            "scope": scope_note,
            "storage_target": target_summary,
            "error": "Disk I/O statistics not available",
        }

    # Issue #801: collapse whole-disk/partition duplicates before summing so a
    # single physical drive is not counted once per partition.
    disk_io_diff = collapse_partition_duplicates(disk_io_diff)

    total_bytes_read = sum(d["bytes_read"] for d in disk_io_diff.values())
    total_bytes_written = sum(d["bytes_written"] for d in disk_io_diff.values())

    duration = duration_seconds if duration_seconds and duration_seconds > 0 else 0

    devices: Dict[str, Any] = {}
    for device, io_stats in disk_io_diff.items():
        bytes_read = io_stats["bytes_read"]
        bytes_written = io_stats["bytes_written"]

        if bytes_read > 0 or bytes_written > 0:
            devices[device] = {
                "bytes_read": bytes_read,
                "bytes_written": bytes_written,
                "read_formatted": format_bytes_fn(bytes_read),
                "write_formatted": format_bytes_fn(bytes_written),
            }

    counters = {
        "total_bytes_read": total_bytes_read,
        "total_bytes_read_per_sec": (
            total_bytes_read / duration if duration > 0 else 0.0
        ),
        "total_bytes_written": total_bytes_written,
        "total_bytes_written_per_sec": (
            total_bytes_written / duration if duration > 0 else 0.0
        ),
        "total_read_formatted": format_bytes_fn(total_bytes_read),
        "total_write_formatted": format_bytes_fn(total_bytes_written),
        "devices": devices,
    }

    if applicable:
        payload: Dict[str, Any] = {
            "applicable": True,
            "status": "OK",
            "scope": scope_note,
            "storage_target": target_summary,
        }
        payload.update(counters)
        return payload

    return {
        "applicable": False,
        "status": "N/A",
        "not_applicable_reason": storage_target.get("reason", ""),
        "scope": scope_note,
        "storage_target": target_summary,
        # Preserved for debugging/audit only; NOT storage-under-test I/O.
        "client_local_io": counters,
    }

