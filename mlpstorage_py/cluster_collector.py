"""
MPI-based Cluster Information Collector for MLPerf Storage.

This module provides functionality to collect system information from all nodes
in a distributed cluster using MPI. It collects data from /proc filesystem
including meminfo, cpuinfo, diskstats, and network statistics.
"""

import fnmatch
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Pattern, Tuple

from mlpstorage_py.config import MPIRUN, MPIEXEC, MPI_RUN_BIN, MPI_EXEC_BIN
from mlpstorage_py.errors import ErrorCode, FileSystemError
from mlpstorage_py.interfaces.collector import ClusterCollectorInterface, CollectionResult
from mlpstorage_py.storage_config import _mask_credential_id, _redact_secret


# =============================================================================
# Localhost Detection
# =============================================================================

LOCALHOST_IDENTIFIERS = ('localhost', '127.0.0.1', '::1')


def _is_localhost(hostname: str) -> bool:
    """Check if hostname refers to local machine.

    Args:
        hostname: The hostname to check.

    Returns:
        True if hostname refers to localhost, False otherwise.
    """
    hostname_lower = hostname.lower()
    if hostname_lower in LOCALHOST_IDENTIFIERS:
        return True
    try:
        local_hostname = socket.gethostname()
        if hostname_lower == local_hostname.lower():
            return True
        local_fqdn = socket.getfqdn()
        if hostname_lower == local_fqdn.lower():
            return True
    except Exception:
        pass
    return False


# =============================================================================
# Data Classes for System Information
# =============================================================================

@dataclass
class HostDiskInfo:
    """
    Disk statistics for a host from /proc/diskstats.

    Fields correspond to the columns in /proc/diskstats as documented in
    the Linux kernel documentation (Documentation/admin-guide/iostats.rst).
    """
    device_name: str
    reads_completed: int = 0
    reads_merged: int = 0
    sectors_read: int = 0
    time_reading_ms: int = 0
    writes_completed: int = 0
    writes_merged: int = 0
    sectors_written: int = 0
    time_writing_ms: int = 0
    ios_in_progress: int = 0
    time_doing_ios_ms: int = 0
    weighted_time_doing_ios_ms: int = 0
    # Optional newer fields (kernel 4.18+)
    discards_completed: Optional[int] = None
    discards_merged: Optional[int] = None
    sectors_discarded: Optional[int] = None
    time_discarding_ms: Optional[int] = None
    # Flush fields (kernel 5.5+)
    flush_requests_completed: Optional[int] = None
    time_flushing_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HostDiskInfo':
        """Create instance from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class HostNetworkInfo:
    """
    Network interface statistics from /proc/net/dev.

    Contains receive (rx) and transmit (tx) statistics for a network interface.
    """
    interface_name: str
    rx_bytes: int = 0
    rx_packets: int = 0
    rx_errors: int = 0
    rx_dropped: int = 0
    rx_fifo: int = 0
    rx_frame: int = 0
    rx_compressed: int = 0
    rx_multicast: int = 0
    tx_bytes: int = 0
    tx_packets: int = 0
    tx_errors: int = 0
    tx_dropped: int = 0
    tx_fifo: int = 0
    tx_collisions: int = 0
    tx_carrier: int = 0
    tx_compressed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HostNetworkInfo':
        """Create instance from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class HostSystemInfo:
    """
    Extended system information for a host.

    Contains kernel version, OS release info, uptime, and load averages.
    """
    hostname: str
    kernel_version: str = ""
    os_release: Dict[str, str] = field(default_factory=dict)
    uptime_seconds: float = 0.0
    load_average_1min: float = 0.0
    load_average_5min: float = 0.0
    load_average_15min: float = 0.0
    running_processes: int = 0
    total_processes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HostSystemInfo':
        """Create instance from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MountInfo:
    """Mount point information from /proc/mounts."""
    device: str
    mount_point: str
    fs_type: str
    options: str
    dump_freq: int = 0
    pass_num: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MountInfo':
        """Create instance from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CgroupInfo:
    """Cgroup subsystem information from /proc/cgroups."""
    subsys_name: str
    hierarchy: int
    num_cgroups: int
    enabled: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CgroupInfo':
        """Create instance from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# /proc File Parsers
# =============================================================================

def parse_proc_meminfo(content: str) -> Dict[str, int]:
    """
    Parse /proc/meminfo content into a dictionary.

    Args:
        content: Raw content of /proc/meminfo file.

    Returns:
        Dictionary mapping field names to values in kB.

    Example:
        >>> content = "MemTotal:       16384000 kB\\nMemFree:         8192000 kB\\n"
        >>> parse_proc_meminfo(content)
        {'MemTotal': 16384000, 'MemFree': 8192000}
    """
    result = {}
    for line in content.strip().split('\n'):
        if not line or ':' not in line:
            continue
        parts = line.split(':')
        if len(parts) != 2:
            continue
        key = parts[0].strip()
        value_parts = parts[1].strip().split()
        if value_parts:
            try:
                # Value is typically in kB, extract just the number
                result[key] = int(value_parts[0])
            except ValueError:
                continue
    return result


def parse_proc_cpuinfo(content: str) -> List[Dict[str, Any]]:
    """
    Parse /proc/cpuinfo content into a list of CPU dictionaries.

    Args:
        content: Raw content of /proc/cpuinfo file.

    Returns:
        List of dictionaries, one per CPU/core, with fields like
        'processor', 'model name', 'cpu cores', etc.

    Example:
        >>> content = "processor\\t: 0\\nmodel name\\t: Intel...\\n\\nprocessor\\t: 1\\n"
        >>> cpus = parse_proc_cpuinfo(content)
        >>> len(cpus)
        2
    """
    cpus = []
    current_cpu = {}

    for line in content.strip().split('\n'):
        line = line.strip()
        if not line:
            # Empty line indicates end of CPU block
            if current_cpu:
                cpus.append(current_cpu)
                current_cpu = {}
            continue

        if ':' not in line:
            continue

        parts = line.split(':', 1)
        if len(parts) != 2:
            continue

        key = parts[0].strip()
        value = parts[1].strip()

        # Try to convert numeric values
        try:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass

        current_cpu[key] = value

    # Don't forget the last CPU if file doesn't end with empty line
    if current_cpu:
        cpus.append(current_cpu)

    return cpus


def parse_proc_diskstats(content: str) -> List[HostDiskInfo]:
    """
    Parse /proc/diskstats content into a list of HostDiskInfo objects.

    Args:
        content: Raw content of /proc/diskstats file.

    Returns:
        List of HostDiskInfo objects, one per disk device.

    Note:
        Only includes block devices (excludes partitions by checking
        if the device name ends with a digit after letters).
    """
    disks = []

    for line in content.strip().split('\n'):
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) < 14:
            continue

        # Fields: major minor name reads_completed reads_merged sectors_read
        #         time_reading writes_completed writes_merged sectors_written
        #         time_writing ios_in_progress time_doing_ios weighted_time
        #         [discards_completed discards_merged sectors_discarded time_discarding]
        #         [flush_requests time_flushing]

        device_name = parts[2]

        try:
            disk_info = HostDiskInfo(
                device_name=device_name,
                reads_completed=int(parts[3]),
                reads_merged=int(parts[4]),
                sectors_read=int(parts[5]),
                time_reading_ms=int(parts[6]),
                writes_completed=int(parts[7]),
                writes_merged=int(parts[8]),
                sectors_written=int(parts[9]),
                time_writing_ms=int(parts[10]),
                ios_in_progress=int(parts[11]),
                time_doing_ios_ms=int(parts[12]),
                weighted_time_doing_ios_ms=int(parts[13]),
            )

            # Parse optional discard fields (kernel 4.18+)
            if len(parts) >= 18:
                disk_info.discards_completed = int(parts[14])
                disk_info.discards_merged = int(parts[15])
                disk_info.sectors_discarded = int(parts[16])
                disk_info.time_discarding_ms = int(parts[17])

            # Parse optional flush fields (kernel 5.5+)
            if len(parts) >= 20:
                disk_info.flush_requests_completed = int(parts[18])
                disk_info.time_flushing_ms = int(parts[19])

            disks.append(disk_info)

        except (ValueError, IndexError):
            continue

    return disks


def parse_proc_net_dev(content: str) -> List[HostNetworkInfo]:
    """
    Parse /proc/net/dev content into a list of HostNetworkInfo objects.

    Args:
        content: Raw content of /proc/net/dev file.

    Returns:
        List of HostNetworkInfo objects, one per network interface.
    """
    interfaces = []
    lines = content.strip().split('\n')

    # Skip header lines (first two lines)
    for line in lines[2:]:
        if not line.strip() or ':' not in line:
            continue

        # Format: "interface: rx_bytes rx_packets ... tx_bytes tx_packets ..."
        parts = line.split(':')
        if len(parts) != 2:
            continue

        interface_name = parts[0].strip()
        stats = parts[1].split()

        if len(stats) < 16:
            continue

        try:
            net_info = HostNetworkInfo(
                interface_name=interface_name,
                rx_bytes=int(stats[0]),
                rx_packets=int(stats[1]),
                rx_errors=int(stats[2]),
                rx_dropped=int(stats[3]),
                rx_fifo=int(stats[4]),
                rx_frame=int(stats[5]),
                rx_compressed=int(stats[6]),
                rx_multicast=int(stats[7]),
                tx_bytes=int(stats[8]),
                tx_packets=int(stats[9]),
                tx_errors=int(stats[10]),
                tx_dropped=int(stats[11]),
                tx_fifo=int(stats[12]),
                tx_collisions=int(stats[13]),
                tx_carrier=int(stats[14]),
                tx_compressed=int(stats[15]),
            )
            interfaces.append(net_info)
        except (ValueError, IndexError):
            continue

    return interfaces


def parse_proc_version(content: str) -> str:
    """
    Parse /proc/version to extract the kernel version string.

    Args:
        content: Raw content of /proc/version file.

    Returns:
        The full kernel version string.

    Example:
        >>> content = "Linux version 5.4.0-42-generic (buildd@lgw01-amd64-038) ..."
        >>> parse_proc_version(content)
        'Linux version 5.4.0-42-generic (buildd@lgw01-amd64-038) ...'
    """
    return content.strip()


def parse_proc_loadavg(content: str) -> Tuple[float, float, float, int, int]:
    """
    Parse /proc/loadavg to extract load averages and process counts.

    Args:
        content: Raw content of /proc/loadavg file.

    Returns:
        Tuple of (1min_avg, 5min_avg, 15min_avg, running_procs, total_procs)

    Example:
        >>> content = "0.50 0.75 0.80 2/500 12345"
        >>> parse_proc_loadavg(content)
        (0.5, 0.75, 0.8, 2, 500)
    """
    parts = content.strip().split()
    if len(parts) < 4:
        return (0.0, 0.0, 0.0, 0, 0)

    try:
        load_1 = float(parts[0])
        load_5 = float(parts[1])
        load_15 = float(parts[2])

        # Parse running/total processes (format: "running/total")
        proc_parts = parts[3].split('/')
        running = int(proc_parts[0]) if len(proc_parts) >= 1 else 0
        total = int(proc_parts[1]) if len(proc_parts) >= 2 else 0

        return (load_1, load_5, load_15, running, total)
    except (ValueError, IndexError):
        return (0.0, 0.0, 0.0, 0, 0)


def parse_proc_uptime(content: str) -> float:
    """
    Parse /proc/uptime to extract system uptime in seconds.

    Args:
        content: Raw content of /proc/uptime file.

    Returns:
        System uptime in seconds.

    Example:
        >>> content = "12345.67 98765.43"
        >>> parse_proc_uptime(content)
        12345.67
    """
    parts = content.strip().split()
    if parts:
        try:
            return float(parts[0])
        except ValueError:
            pass
    return 0.0


def parse_os_release(content: str) -> Dict[str, str]:
    """
    Parse /etc/os-release content into a dictionary.

    Args:
        content: Raw content of /etc/os-release file.

    Returns:
        Dictionary of OS release information.

    Example:
        >>> content = 'NAME="Ubuntu"\\nVERSION="20.04"\\n'
        >>> parse_os_release(content)
        {'NAME': 'Ubuntu', 'VERSION': '20.04'}
    """
    result = {}
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line or '=' not in line:
            continue

        key, _, value = line.partition('=')
        # Remove quotes if present
        value = value.strip('"\'')
        result[key] = value

    return result


def parse_proc_vmstat(content: str) -> Dict[str, int]:
    """
    Parse /proc/vmstat content into a dictionary.

    Args:
        content: Raw content of /proc/vmstat file.

    Returns:
        Dictionary mapping field names to integer values.

    Example:
        >>> content = "nr_free_pages 12345\\nnr_zone_inactive_anon 6789\\n"
        >>> parse_proc_vmstat(content)
        {'nr_free_pages': 12345, 'nr_zone_inactive_anon': 6789}
    """
    result = {}
    for line in content.strip().split('\n'):
        parts = line.split()
        if len(parts) == 2:
            try:
                result[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return result


def parse_proc_mounts(content: str) -> List[MountInfo]:
    """
    Parse /proc/mounts content into a list of MountInfo objects.

    Args:
        content: Raw content of /proc/mounts file.

    Returns:
        List of MountInfo objects, one per mount point.

    Example:
        >>> content = "/dev/sda1 / ext4 rw,relatime 0 1"
        >>> mounts = parse_proc_mounts(content)
        >>> mounts[0].mount_point
        '/'
    """
    mounts = []
    for line in content.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                mount = MountInfo(
                    device=parts[0],
                    mount_point=parts[1],
                    fs_type=parts[2],
                    options=parts[3],
                    dump_freq=int(parts[4]) if len(parts) > 4 else 0,
                    pass_num=int(parts[5]) if len(parts) > 5 else 0,
                )
                mounts.append(mount)
            except (ValueError, IndexError):
                continue
    return mounts


def parse_proc_cgroups(content: str) -> List[CgroupInfo]:
    """
    Parse /proc/cgroups content into a list of CgroupInfo objects.

    Args:
        content: Raw content of /proc/cgroups file.

    Returns:
        List of CgroupInfo objects, one per cgroup subsystem.

    Example:
        >>> content = "#subsys_name\\thierarchy\\tnum_cgroups\\tenabled\\ncpu\\t0\\t1\\t1\\n"
        >>> cgroups = parse_proc_cgroups(content)
        >>> cgroups[0].subsys_name
        'cpu'
    """
    cgroups = []
    lines = content.strip().split('\n')
    for line in lines:
        # Skip header line and comments
        if line.startswith('#') or 'subsys_name' in line:
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                cgroup = CgroupInfo(
                    subsys_name=parts[0],
                    hierarchy=int(parts[1]),
                    num_cgroups=int(parts[2]),
                    enabled=parts[3] == '1',
                )
                cgroups.append(cgroup)
            except (ValueError, IndexError):
                continue
    return cgroups


# =============================================================================
# Chassis Model Collection (DMI / SMBIOS) — Phase 3 Plan 02 (D-21, COLL-03)
# =============================================================================
#
# Path indirection lets tests point the reader at a tmp_path fixture without
# patching builtins.open (which corrupts PyYAML and other concurrent I/O —
# see RESEARCH 738-792 / Anti-Patterns to Avoid).
_DMI_PRODUCT_NAME_PATH: str = '/sys/class/dmi/id/product_name'


# D-21 verbatim placeholder list, normalized to lower-case. The collector
# treats any of these (case-insensitive, post-strip) as "BIOS junk, treat
# as blank" — the blank then surfaces in submission validation as a
# visible to-do for the submitter (SER-02 pattern). The empty string is
# included so an empty product_name file collapses naturally through the
# same set-membership branch.
_DMI_PLACEHOLDERS: Final[frozenset[str]] = frozenset({
    "",
    "to be filled by o.e.m.",
    "default string",
    "system product name",
    "system manufacturer",
    "none",
    "not specified",
    "not applicable",
    "oem",
    "unknown",
})


def _normalize_dmi(s: str) -> str:
    """Normalize a DMI product_name string per D-21.

    Returns the empty string when ``s.strip().lower()`` matches any entry in
    ``_DMI_PLACEHOLDERS``; otherwise returns ``s.strip()`` (case preserved
    for real product names like "PowerEdge R760").

    Both branches are pure str → str. No exception path.
    """
    stripped = s.strip()
    if stripped.lower() in _DMI_PLACEHOLDERS:
        return ""
    return stripped


def collect_chassis_model(dmi_path: str = _DMI_PRODUCT_NAME_PATH) -> str:
    """Read the system's chassis/product model from DMI/SMBIOS sysfs.

    Per the universal D-2 collection-failure rule: any exception (file
    missing on a container without DMI passthrough, PermissionError on
    a hardened image, OSError EINVAL on an exotic kernel, decoding
    failure on a corrupt SMBIOS table) yields the empty string. The
    collector NEVER raises for a sysfs read failure.

    T-3-05 mitigation: explicit 8KB read cap (sysfs files are kernel-buffered
    to PAGE_SIZE on Linux, typically 4KB; the cap is defense-in-depth
    against any future kernel exposing an unbounded blob here).

    Args:
        dmi_path: Path to the DMI product_name file. Production callers
            use the module default; tests pass a tmp_path fixture.

    Returns:
        The normalized product name (e.g., "PowerEdge R760"), or the empty
        string on any read failure or D-21 placeholder match.
    """
    try:
        with open(dmi_path, 'r') as f:
            raw = f.read(8192)
        return _normalize_dmi(raw)
    except Exception:
        return ""


# =============================================================================
# Networking Collection (sysfs + InfiniBand) — Phase 3 Plan 03
#                     (D-18 filter scope, D-19 IB-first, D-20 operstate +
#                      effective-state demotion, COLL-04)
# =============================================================================
#
# Per-host enumeration of real NICs and IB ports. Output is a flat list of
# per-iface/per-port {type, speed, state} dicts, ungrouped — the per-host
# grouping into stanzas with unit_count happens in auto_generator's
# node_dict_from_host via group_by_fingerprint (Plan 03-04 / 03-05).
#
# Path indirection (net_root, ib_root parameters with module-default
# production constants) lets tests point the reader at a tmp_path fixture
# without patching builtins.open — RESEARCH 738-792 / Pattern D.
_SYSFS_NET_ROOT: str = '/sys/class/net'
_SYSFS_INFINIBAND_ROOT: str = '/sys/class/infiniband'

# D-18 virtual-interface name-prefix list (belt-and-suspenders against
# D-19 IPoIB shadow double-counting via ib*/iboeth*/ib_eth* entries).
# Compiled as a single regex at module load. The literal prefix tuple is
# preserved alongside so D-18 grep-finds the source-of-truth list.
_VIRTUAL_NAME_PREFIXES: Tuple[str, ...] = (
    'lo', 'docker', 'virbr', 'veth', 'tun', 'tap', 'gre', 'wg',
    'ib', 'iboeth', 'ib_eth',
)

# Matches: exact 'lo'; docker0/docker123; virbr0; veth* with any suffix;
# tun/tap/gre/wg with optional numeric suffix; ib0; iboeth0; ib_eth0.
# Anchored ^...$ for whole-name match.
_VIRTUAL_NAME_RE = re.compile(
    r'^(lo|docker[0-9]*|virbr[0-9]*|veth.*|tun[0-9]*|tap[0-9]*|'
    r'gre[0-9]*|wg[0-9]*|ib[0-9]*|iboeth[0-9]*|ib_eth[0-9]*)$'
)

# D-20 permissive operstate mapping: 'up' and 'unknown' both map to up.
# 'unknown' is included because many drivers (virtio, several wifi) don't
# update operstate when carrier comes online; treating unknown as down
# would systematically misreport those NICs.
_OPERSTATE_UP_VALUES: frozenset = frozenset({'up', 'unknown'})

# T-3-07 belt-and-suspenders: iface names are kernel-side basenames but a
# defensive whitelist regex blocks any name with path separators or shell
# metacharacters before we construct a sysfs path with it. POSIX device
# names are [A-Za-z0-9._-]+; nothing legitimate violates this.
_SAFE_IFACE_NAME_RE = re.compile(r'^[A-Za-z0-9._-]+$')


def _read_sysfs_text(path: str) -> str:
    """Read a single-line sysfs file; return its stripped content.

    Returns '' on any exception (FileNotFoundError on hot-unplug,
    PermissionError on hardened images, OSError on exotic kernels).
    T-3-09: explicit 8KB read cap defense-in-depth against an unbounded
    blob (sysfs is normally PAGE_SIZE-buffered to 4KB).
    """
    try:
        with open(path, 'r') as f:
            return f.read(8192).strip()
    except Exception:
        return ''


def _read_sysfs_int(path: str, default: int = -1) -> int:
    """Read a sysfs file and int() its first token; return default on any failure.

    T-3-08: int() failure on tampered/garbled content returns default
    rather than raising; the -1 sentinel then participates in the D-20
    effective-state demotion path so the iface emits as down with no
    speed key.
    """
    try:
        with open(path, 'r') as f:
            raw = f.read(8192).strip()
        return int(raw.split()[0])
    except Exception:
        return default


def _is_virtual_by_name(iface: str) -> bool:
    """D-18 name-prefix shortcut: True if the iface name matches any
    excluded prefix. Fast O(1) regex test before any I/O.

    D-18 source-of-truth prefix list (also exposed via
    _VIRTUAL_NAME_PREFIXES for grep): lo, docker*, virbr*, veth*, tun*,
    tap*, gre*, wg*. D-19 belt-and-suspenders prefixes (skip IPoIB
    shadows from net walk): ib*, iboeth*, ib_eth*.
    """
    return _VIRTUAL_NAME_RE.match(iface) is not None


def _is_bridge_master(iface_dir: str) -> bool:
    """D-18: a Linux bridge (br0 etc.) carries /sys/class/net/<iface>/bridge/.
    Its speed is meaningless (kernel returns a default value regardless
    of real port speeds). Skip."""
    return os.path.isdir(os.path.join(iface_dir, 'bridge'))


def _is_vlan_subif(iface_dir: str) -> bool:
    """D-18: detect VLAN sub-interfaces (eth0.100) and MACVLAN/IPVLAN
    children. Both have iflink pointing at the parent interface's
    ifindex, distinct from their own ifindex.

    Returns False if either sysfs file is unreadable — we err on the
    side of inclusion (a stale read of '0' for both would otherwise
    falsely match) rather than silently dropping real NICs.
    """
    iflink_path = os.path.join(iface_dir, 'iflink')
    ifindex_path = os.path.join(iface_dir, 'ifindex')
    if not (os.path.exists(iflink_path) and os.path.exists(ifindex_path)):
        return False
    iflink = _read_sysfs_int(iflink_path, default=-1)
    ifindex = _read_sysfs_int(ifindex_path, default=-1)
    if iflink == -1 or ifindex == -1:
        return False
    return iflink != ifindex


def _is_bond_slave(iface_dir: str) -> bool:
    """D-18: bond slaves leak operstate=up + a real speed; the bond's own
    aggregate speed is reported on the master. We must filter slaves so
    they do not double-count alongside the master. Detection: <iface>/
    master symlink exists AND basename of the target starts with 'bond'.
    """
    master_path = os.path.join(iface_dir, 'master')
    if not os.path.exists(master_path):
        return False
    try:
        target = os.readlink(master_path)
        return os.path.basename(target).startswith('bond')
    except OSError:
        return False


def _is_bond_master(iface_dir: str) -> bool:
    """D-18: bond masters carry /sys/class/net/<iface>/bonding/ subdir."""
    return os.path.isdir(os.path.join(iface_dir, 'bonding'))


def _bond_aggregate_speed_mbps(iface_dir: str, net_root: str) -> int:
    """Per RESEARCH 484-519 bond master aggregation: sum the speed_mbps of
    every active slave (positive speed only; slaves with speed=-1 or 0
    contribute zero). Returns total Mbps; 0 means the bond has no active
    legs.

    The bond's own /sys/class/net/<bond>/speed is unreliable (Pitfall 4)
    — many drivers report a sentinel rather than the aggregate. We always
    walk slaves and re-derive.
    """
    slaves_path = os.path.join(iface_dir, 'bonding', 'slaves')
    raw = _read_sysfs_text(slaves_path)
    if not raw:
        return 0
    total = 0
    for name in raw.split():
        if not _SAFE_IFACE_NAME_RE.match(name):
            # T-3-09: skip slave names with shell metacharacters / path
            # separators rather than constructing a sysfs path with them.
            continue
        speed = _read_sysfs_int(os.path.join(net_root, name, 'speed'),
                                default=-1)
        if speed > 0:
            total += speed
    return total


def _map_operstate(operstate: str) -> str:
    """D-20 permissive operstate mapping: 'up' | 'unknown' → 'up';
    everything else (down, dormant, notpresent, lowerlayerdown, testing)
    → 'down'."""
    return 'up' if operstate.strip().lower() in _OPERSTATE_UP_VALUES else 'down'


def _parse_ib_state(state_file_contents: str) -> str:
    """D-19; Pitfall 8 robust '4:' prefix match. The IB state file format
    is '<num>: <text>' (e.g. '4: ACTIVE'). Only '4: ACTIVE' counts as up;
    everything else (1: DOWN, 2: INIT, 3: ARMED, 5: ACTIVE_DEFER) is down.
    """
    return 'up' if state_file_contents.strip().startswith('4:') else 'down'


def _parse_ib_rate(rate_file_contents: str) -> Optional[int]:
    """D-19: parse 'NN Gb/sec (...)' → int(NN). Returns None on parse
    failure (empty file, garbled content); the caller demotes such ports
    to state=down so the downstream Pydantic NetworkPort never sees an
    'up' state without a speed.
    """
    try:
        return int(rate_file_contents.split()[0])
    except (ValueError, IndexError):
        return None


def collect_networking(net_root: str = _SYSFS_NET_ROOT,
                       ib_root: str = _SYSFS_INFINIBAND_ROOT) -> List[Dict[str, Any]]:
    """Enumerate real per-host network interfaces and IB ports.

    Returns a list of {type, speed?, state} dicts — one entry per
    surviving ethernet iface (D-18 filters virtuals/bridges/VLANs/bond
    slaves; bond masters emit ONE aggregated entry per LAG) and one
    entry per IB port (D-19 IB-first). The list is ungrouped — per-host
    stanza collapsing happens in node_dict_from_host (Plan 03-05) via
    group_by_fingerprint(...).

    Universal D-2 rule applied at per-iface scope: a single bad iface
    (hot-unplug FileNotFoundError, PermissionError, parse error)
    silently skips that one entry; the function returns whatever
    entries did succeed.

    Args:
        net_root: Path to /sys/class/net (production default); tests
            pass a tmp_path fixture root.
        ib_root: Path to /sys/class/infiniband (production default);
            tests pass a tmp_path fixture root or a non-existent path
            to exercise the missing-IB-hardware code path.

    Returns:
        list of dicts. For up ethernet/IB: {type, speed (Gbps), state:'up'}.
        For down: {type, state:'down'} (no speed key, consistent with
        Pydantic model_dump(exclude_none=True) on a NetworkPort with
        speed=None).
    """
    out: List[Dict[str, Any]] = []

    # --- Ethernet walk: /sys/class/net/* per RESEARCH 484-519 decision tree.
    try:
        net_entries = os.listdir(net_root)
    except Exception:
        net_entries = []

    for iface in net_entries:
        try:
            # T-3-07: defensive name validation before constructing any path.
            if not _SAFE_IFACE_NAME_RE.match(iface):
                continue

            # 1. Name-prefix shortcut (D-18 + D-19): cheap O(1) reject.
            if _is_virtual_by_name(iface):
                continue

            iface_dir = os.path.join(net_root, iface)

            # 2. Bridge master?
            if _is_bridge_master(iface_dir):
                continue

            # 3. VLAN / MACVLAN / IPVLAN sub-interface?
            if _is_vlan_subif(iface_dir):
                continue

            # 4. Bond slave?
            if _is_bond_slave(iface_dir):
                continue

            # 5. Bond master? Emit one aggregated entry per LAG.
            if _is_bond_master(iface_dir):
                aggregate_mbps = _bond_aggregate_speed_mbps(iface_dir, net_root)
                if aggregate_mbps > 0:
                    out.append({
                        'type': 'ethernet',
                        'speed': aggregate_mbps // 1000,
                        'state': 'up',
                    })
                else:
                    out.append({'type': 'ethernet', 'state': 'down'})
                continue

            # 6. Plain physical ethernet.
            iface_type = _read_sysfs_text(os.path.join(iface_dir, 'type'))
            if iface_type != '1':
                # Not ARPHRD_ETHER (e.g., sit0=776, ip6tnl0=769). Skip.
                continue

            operstate_raw = _read_sysfs_text(os.path.join(iface_dir, 'operstate'))
            mapped_state = _map_operstate(operstate_raw)
            speed_mbps = _read_sysfs_int(os.path.join(iface_dir, 'speed'),
                                         default=-1)

            # D-20 effective-state demotion: an iface that says 'up' but
            # reports no negotiated speed is operationally down (Pitfall 2
            # — virtio NICs report speed=-1 despite operstate=up).
            if mapped_state == 'up' and speed_mbps in (-1, 0):
                mapped_state = 'down'

            if mapped_state == 'up':
                out.append({
                    'type': 'ethernet',
                    'speed': speed_mbps // 1000,
                    'state': 'up',
                })
            else:
                out.append({'type': 'ethernet', 'state': 'down'})
        except Exception:
            # Per-iface defense (D-2 at iface scope): any unexpected
            # failure skips this one iface, never aborts the walk.
            continue

    # --- InfiniBand walk: /sys/class/infiniband/<dev>/ports/<port>/ per D-19.
    if os.path.isdir(ib_root):
        try:
            ib_devs = os.listdir(ib_root)
        except Exception:
            ib_devs = []
        for dev in ib_devs:
            if not _SAFE_IFACE_NAME_RE.match(dev):
                continue
            ports_dir = os.path.join(ib_root, dev, 'ports')
            if not os.path.isdir(ports_dir):
                continue
            try:
                ports = os.listdir(ports_dir)
            except Exception:
                continue
            for port in ports:
                try:
                    if not _SAFE_IFACE_NAME_RE.match(port):
                        continue
                    port_dir = os.path.join(ports_dir, port)
                    state_raw = _read_sysfs_text(os.path.join(port_dir, 'state'))
                    ib_state = _parse_ib_state(state_raw)
                    if ib_state == 'up':
                        rate_raw = _read_sysfs_text(os.path.join(port_dir, 'rate'))
                        speed = _parse_ib_rate(rate_raw)
                        if speed is None:
                            # D-19 blank-splice: state ACTIVE but rate
                            # unparseable → cannot truthfully claim 'up'
                            # without a speed. Emit as down.
                            out.append({'type': 'infiniband', 'state': 'down'})
                        else:
                            out.append({
                                'type': 'infiniband',
                                'speed': speed,
                                'state': 'up',
                            })
                    else:
                        out.append({'type': 'infiniband', 'state': 'down'})
                except Exception:
                    continue

    return out


# =============================================================================
# Sysctl Collection (/proc/sys walk + shipped allowlist file) — Phase 4 Plan
#                     04-01 (D-27 allowlist file, D-28 walk semantics,
#                      D-29 multi-value verbatim emit, COLL-05)
# =============================================================================
#
# Data-driven allowlist: editing
# mlpstorage_py/system_description/sysctl_allowlist.txt adds keys to the next
# run's output with no code change. The walk reads each leaf with an 8 KiB
# cap (defense-in-depth; /proc/sys is PAGE_SIZE-buffered to ~4 KiB) and emits
# {name, value} dicts in dotted form. Per-leaf failures (write-only leaves
# like vm.drop_caches, PermissionError on hardened kernels, OSError on
# disappearing dynamic entries) are isolated per the universal D-2 rule —
# the offending key is skipped, the walk continues. RESEARCH Q2 documents
# the write-only-leaf set.
_PROC_SYS_ROOT: str = '/proc/sys'

# Shipped allowlist file lives alongside the schema in system_description/.
# Submitters edit it in-place in their editable install; no CLI override is
# offered in Phase 4 (deferred to a later phase per CONTEXT.md).
_SYSCTL_ALLOWLIST_PATH: str = str(
    Path(__file__).parent / 'system_description' / 'sysctl_allowlist.txt'
)


def _load_sysctl_allowlist(
    path: str = _SYSCTL_ALLOWLIST_PATH,
) -> Tuple[Pattern, ...]:
    """Load the on-disk allowlist and return a tuple of compiled regex objects.

    One regex per glob line, via ``re.compile(fnmatch.translate(glob))``.
    Blank lines and lines whose ``lstrip()`` starts with ``#`` are skipped;
    each glob is ``.strip()``-ed before translation. On any read failure
    (FileNotFoundError, OSError, PermissionError) returns ``tuple()`` per
    the universal D-2 collection-failure rule — ``collect_sysctl`` then
    matches nothing and emits ``[]``.

    RESEARCH Q3 "deep-match" gotcha: ``fnmatch`` is NOT path-separator aware.
    A glob ``net.core.*`` matches ``net.core.rmem_max`` AND
    ``net.core.bpf_jit_harden`` AND any future deeper-nested key the kernel
    might introduce (``net.core.foo.bar``). The current shipped patterns
    intentionally use narrow prefixes that sidestep this; a future editor
    adding ``net.ipv4.*`` would also pick up ``net.ipv4.conf.eth0.forwarding``
    (interface-parameterized leaves do appear in the /proc/sys walk per
    RESEARCH Q2). If shallow-only matching is ever desired, use
    ``re.compile(r'^net\\.ipv4\\.[^.]+$')`` directly instead of fnmatch.
    """
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
    except OSError:
        return tuple()
    patterns: List[Pattern] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.lstrip().startswith('#'):
            continue
        patterns.append(re.compile(fnmatch.translate(line)))
    return tuple(patterns)


def collect_sysctl(
    proc_sys_root: str = _PROC_SYS_ROOT,
    allowlist: Optional[Tuple[Pattern, ...]] = None,
) -> List[Dict[str, str]]:
    """Walk /proc/sys, emit one {name, value} dict per matching leaf.

    Per D-27 / D-28: each leaf path is converted to its dotted form (e.g.,
    ``/proc/sys/vm/dirty_ratio`` → ``vm.dirty_ratio``) and matched against
    every pattern in ``allowlist``. Leaves matching at least one pattern
    are read (8 KiB cap per D-28 / RESEARCH Q2), trailing newline stripped
    (D-29: internal whitespace preserved verbatim so multi-value leaves
    like ``net.ipv4.tcp_rmem`` round-trip cleanly), and emitted.

    Universal D-2 rule applies at two scopes:
      - Outer ``os.walk`` envelope: any exception (missing root, OSError)
        yields the empty list. The collector never raises.
      - Per-leaf inner try/except: a single PermissionError / OSError /
        UnicodeDecodeError on a write-only leaf (RESEARCH Q2: drop_caches,
        compact_memory, route/flush, sysrq) skips that leaf; the walk
        continues. The reduces P(one weird key kills the whole walk) → 0.

    Args:
        proc_sys_root: path to the /proc/sys walk root. Production callers
            use the module default; tests pass a tmp_path fixture.
        allowlist: tuple of compiled regex objects. ``None`` triggers a
            fresh ``_load_sysctl_allowlist()`` read (the production path);
            tests pass a constructed tuple.

    Returns:
        List of ``{"name": "<dotted>", "value": "<verbatim>"}`` dicts.
        Empty list on any catastrophic failure.
    """
    if allowlist is None:
        allowlist = _load_sysctl_allowlist()
    out: List[Dict[str, str]] = []
    try:
        walker = os.walk(proc_sys_root)
        for dirpath, _dirnames, filenames in walker:
            for filename in filenames:
                leaf = os.path.join(dirpath, filename)
                rel = os.path.relpath(leaf, proc_sys_root)
                # Dotted form: '/proc/sys/net/ipv4/tcp_rmem' →
                # 'net.ipv4.tcp_rmem'. os.sep is '/' on Linux but this
                # generalizes via os.sep.
                name = rel.replace(os.sep, '.')
                if not any(p.match(name) for p in allowlist):
                    continue
                try:
                    with open(leaf, 'r') as f:
                        raw = f.read(8192)
                except Exception:
                    # D-2 / RESEARCH Q2: write-only or protected leaf;
                    # skip and continue, never abort the walk.
                    continue
                # D-29: strip only the trailing newline; preserve internal
                # tabs / whitespace so multi-value emit is verbatim.
                out.append({"name": name, "value": raw.rstrip('\n')})
    except Exception:
        # D-2 outer envelope: catastrophic failure (no /proc/sys, exotic
        # kernel surfacing OSError EINVAL on os.walk) → empty list.
        return []
    return out


# =============================================================================
# Phase 4 Plan 04-02 — Environment collector (D-23, D-24, D-25, D-26, COLL-06)
# =============================================================================
#
# `collect_environment()` filters `os.environ` to a prefix-or-literal allowlist
# (D-26: BUCKET literal + AWS_*, STORAGE_*, OMPI_*, UCX_*, NCCL_* prefixes) and
# dispatches AWS credential vars through the unified storage_config redactors
# (D-23 `_mask_credential_id` for KEY_ID; D-24 `_redact_secret` for SECRET).
# All other matching vars are emitted verbatim.
#
# The returned list is sorted by `name` for deterministic emit (D-34 fingerprint
# stability — different host environments must produce the same byte order
# when the var set is identical).
#
# Pattern B duplicate (D-36): the four symbols (_ENV_LITERALS, _ENV_PREFIXES,
# _env_allowlist_match, collect_environment) plus the two redactors (untyped
# form) live inline in MPI_COLLECTOR_SCRIPT below. The parity test exec's the
# script in a controlled namespace and asserts behavioral equivalence on a
# monkeypatched os.environ snapshot.
# =============================================================================

_ENV_LITERALS: Final[frozenset] = frozenset({"BUCKET"})

_ENV_PREFIXES: Final[Tuple[str, ...]] = (
    "AWS_",
    "STORAGE_",
    "OMPI_",
    "UCX_",
    "NCCL_",
)

# Runtime-volatile launcher metadata that matches the OMPI_ allowlist prefix
# but changes on every mpirun invocation (PIDs, TCP sockets, jobids, crypto
# tokens, session dirs, command-line). Must be excluded from the fingerprint
# or SystemDriftError fires on every legitimate re-run (Phase 5 UAT Test 3,
# LIFE-04). Phase 5.1 will broaden this to other launchers
# (.planning/todos/pending/phase-5.1-env-sysctl-fingerprint-audit.md).
_ENV_RUNTIME_DENYLIST: Final[frozenset] = frozenset({
    "OMPI_ARGV",
    "OMPI_FILE_LOCATION",
    "OMPI_MCA_ess_base_jobid",
    "OMPI_MCA_orte_hnp_uri",
    "OMPI_MCA_orte_jobfam_session_dir",
    "OMPI_MCA_orte_local_daemon_uri",
    "OMPI_MCA_orte_precondition_transports",
})


def _env_allowlist_match(name: str) -> bool:
    """Return True iff `name` is in `_ENV_LITERALS` or starts with any
    element of `_ENV_PREFIXES`, AND `name` is not in
    `_ENV_RUNTIME_DENYLIST` (D-26 + runtime-volatile guard).

    Case-sensitive (matches POSIX env-var convention; `bucket` is NOT a
    BUCKET match).
    """
    if name in _ENV_RUNTIME_DENYLIST:
        return False
    return name in _ENV_LITERALS or name.startswith(_ENV_PREFIXES)


def collect_environment() -> List[Dict[str, str]]:
    """Return allowlisted environment variables, with AWS credentials redacted.

    Per D-26: only vars whose name matches `_env_allowlist_match` are emitted.
    Per D-23: AWS_ACCESS_KEY_ID flows through `_mask_credential_id`.
    Per D-24: AWS_SECRET_ACCESS_KEY flows through `_redact_secret`.
    All other matching vars are emitted verbatim.
    Per D-34: output is sorted by `name` for deterministic fingerprinting.

    D-2 envelope: any exception (e.g., the allowlist match function blowing
    up, a hostile os.environ surface) yields `[]` rather than raising. The
    collector never fails the benchmark.

    Returns:
        Sorted list of ``{"name": str, "value": str}`` dicts.
    """
    try:
        out: List[Dict[str, str]] = []
        for name, value in sorted(os.environ.items()):
            if not _env_allowlist_match(name):
                continue
            if name == "AWS_ACCESS_KEY_ID":
                value = _mask_credential_id(value)
            elif name == "AWS_SECRET_ACCESS_KEY":
                value = _redact_secret(value)
            out.append({"name": name, "value": value})
        return out
    except Exception:
        return []


# =============================================================================
# Phase 4 / Plan 04-03 — Drives collector (COLL-07, D-30/31/32/33/36)
# =============================================================================
#
# Invokes `lsblk -J -b -d -o NAME,MODEL,VENDOR,SIZE,ROTA,TRAN,RM` via
# subprocess.run, JSON-parses the output, applies the D-31 four-rule filter
# chain, and emits one `{vendor_name, model_name, interface, capacity_in_GB}`
# dict per surviving row per D-30.
#
# D-2 envelope at two scopes:
#   - Outer try/except: any subprocess failure (FileNotFoundError on busybox
#     Alpine, TimeoutExpired on stuck I/O, SubprocessError, JSONDecodeError on
#     stdout corruption, non-zero returncode) → return [].
#   - Per-row try/except: a single malformed row (non-int size, missing keys)
#     skips itself, never aborts the whole walk.
#
# D-33: empty output (lsblk absent OR all rows filtered) yields []; the
# auto_generator transform layer (Plan 04-04) is responsible for omitting
# the `drives` key from the emitted client stanza entirely.
#
# Pattern B (D-36): collect_drives + _LSBLK_ARGS + the three filter constants
# are duplicated inline in MPI_COLLECTOR_SCRIPT (untyped form per Phase 3
# convention). The parity test asserts behavioral equivalence under the same
# monkeypatched subprocess.run.
#
# RESEARCH Q1 quirks honored:
#   (a) Empty TRAN with NAME starting `nvme` is rescued to TRAN='nvme'
#       (older kernels on some NVMe drives reported TRAN='' rather than
#       'nvme' before kernel 5.4).
#   (b) RM is sometimes string ('0'/'1') in util-linux <2.37 and int (0/1)
#       in util-linux >=2.37; `str(row.get('rm', '0')) == '1'` handles both.
#   (c) SIZE is bytes because of the -b flag; capacity_in_GB is decimal GB
#       (// 10**9) per the nameplate-capacity convention drive specs use.
# =============================================================================

_LSBLK_ARGS: Final[Tuple[str, ...]] = (
    'lsblk', '-J', '-b', '-d', '-o', 'NAME,MODEL,VENDOR,SIZE,ROTA,TRAN,RM',
)

_DRIVE_VIRTUAL_NAME_PREFIXES: Final[Tuple[str, ...]] = (
    'loop', 'dm-', 'zram', 'ram', 'sr', 'fd',
)

_DRIVE_VIRTUAL_TRANS: Final[frozenset] = frozenset({'loop', 'zram'})

_DRIVE_ACCEPTED_TRANS: Final[frozenset] = frozenset({'nvme', 'sata', 'sas'})


def collect_drives() -> List[Dict[str, Any]]:
    """Return one dict per accepted block device, with D-31 filter chain
    applied to ``lsblk -J -b -d -o NAME,MODEL,VENDOR,SIZE,ROTA,TRAN,RM`` output.

    Each surviving row emits the D-30 grouping-key shape:
        ``{vendor_name, model_name, interface, capacity_in_GB}``

    D-31 four-rule filter chain (applied per row, in order, with early continue):
      1. RM=1 reject — string-or-int coercion via `str(row.get('rm','0'))=='1'`
         (RESEARCH Q1 util-linux variance).
      2. Virtual NAME prefix reject — `{loop, dm-, zram, ram, sr, fd}`.
         Virtual TRAN reject — `{loop, zram}`.
      3. Unknown TRAN drop — only `{nvme, sata, sas}` accepted. Exception:
         empty TRAN with NAME starting `nvme` is rescued to `nvme` (RESEARCH
         Q1 quirk a — older kernels report TRAN='' on some NVMe drives).
         Note: rows with unknown TRAN (`usb`, `virtio`, `ata`, `mmc`, etc.) are
         DROPPED, NOT mapped to `'other'`. The DriveInstance.other schema enum
         value remains for submitter hand-fills (D-31, 04-CONTEXT §specifics).
      4. Per-row try/except on the emit step isolates a single bad row.

    D-2 envelope: lsblk binary absent (FileNotFoundError on busybox / minimal
    Alpine), TimeoutExpired, SubprocessError, JSONDecodeError, non-zero
    returncode → `[]`. The collector never raises.

    Per D-33, an empty return (absent lsblk or all rows filtered) is the
    universal "absent or invalid" signal the auto_generator splice layer uses
    to omit the `drives` key from the emitted client stanza entirely.

    Per D-32, no schema change; `media_type`, `form_factor`, and
    `performance` are deliberately NOT emitted (SER-02 submitter
    responsibility — spec-sheet facts not derivable from lsblk).
    """
    try:
        cp = subprocess.run(
            list(_LSBLK_ARGS),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if cp.returncode != 0:
            return []
        payload = json.loads(cp.stdout)
        rows = payload.get('blockdevices', []) or []
        out: List[Dict[str, Any]] = []
        for row in rows:
            try:
                # D-31 rule 1: removable skip. Handle all three observed
                # util-linux variants: string '1' (<2.37), int 1 (>=2.37
                # without --bytes), and bool True (>=2.37 with newer JSON
                # output, including WSL2 dev shell). bool is a subclass of
                # int in Python; isinstance(True, int) is True.
                rm = row.get('rm', 0)
                if rm in (1, True, '1'):
                    continue
                # D-31 rule 2: virtual NAME prefix or TRAN.
                name = row.get('name', '') or ''
                if name.startswith(_DRIVE_VIRTUAL_NAME_PREFIXES):
                    continue
                tran = (row.get('tran') or '').lower()
                if tran in _DRIVE_VIRTUAL_TRANS:
                    continue
                # D-31 rule 3: unknown TRAN drop, with empty-TRAN-nvme-name
                # rescue per RESEARCH Q1 quirk (a).
                if tran not in _DRIVE_ACCEPTED_TRANS:
                    if tran == '' and name.startswith('nvme'):
                        tran = 'nvme'  # rescue
                    else:
                        continue
                # D-30 emit. Per-row try around int() catches non-int size /
                # missing size key — isolated row failure, not whole walk.
                capacity_in_GB = int(row['size']) // 10**9
                out.append({
                    'vendor_name':    (row.get('vendor') or '').strip(),
                    'model_name':     (row.get('model') or '').strip(),
                    'interface':      tran,
                    'capacity_in_GB': capacity_in_GB,
                })
            except Exception:
                # Per-row D-2 — single malformed row skips itself.
                continue
        return out
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        subprocess.SubprocessError,
        json.JSONDecodeError,
        Exception,
    ):
        return []


# =============================================================================
# Local System Information Collection
# =============================================================================

def collect_local_system_info() -> Dict[str, Any]:
    """
    Collect system information from the local node.

    Reads various /proc files and /etc/os-release to gather comprehensive
    system information about the local host.

    Returns:
        Dictionary containing:
        - hostname: str
        - meminfo: Dict from /proc/meminfo
        - cpuinfo: List[Dict] from /proc/cpuinfo
        - diskstats: List[Dict] from /proc/diskstats
        - netdev: List[Dict] from /proc/net/dev
        - version: str from /proc/version
        - loadavg: Dict with load average info from /proc/loadavg
        - uptime: float from /proc/uptime
        - os_release: Dict from /etc/os-release
        - vmstat: Dict from /proc/vmstat
        - mounts: List[Dict] from /proc/mounts
        - cgroups: List[Dict] from /proc/cgroups
        - collection_timestamp: ISO format timestamp
        - errors: Dict of any errors encountered during collection
    """
    result = {
        'hostname': socket.gethostname(),
        'collection_timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'errors': {}
    }

    # Collect /proc/meminfo
    try:
        with open('/proc/meminfo', 'r') as f:
            result['meminfo'] = parse_proc_meminfo(f.read())
    except Exception as e:
        result['errors']['meminfo'] = str(e)
        result['meminfo'] = {}

    # Collect /proc/cpuinfo
    try:
        with open('/proc/cpuinfo', 'r') as f:
            result['cpuinfo'] = parse_proc_cpuinfo(f.read())
    except Exception as e:
        result['errors']['cpuinfo'] = str(e)
        result['cpuinfo'] = []

    # Collect /proc/diskstats
    try:
        with open('/proc/diskstats', 'r') as f:
            disks = parse_proc_diskstats(f.read())
            result['diskstats'] = [d.to_dict() for d in disks]
    except Exception as e:
        result['errors']['diskstats'] = str(e)
        result['diskstats'] = []

    # Collect /proc/net/dev
    try:
        with open('/proc/net/dev', 'r') as f:
            interfaces = parse_proc_net_dev(f.read())
            result['netdev'] = [n.to_dict() for n in interfaces]
    except Exception as e:
        result['errors']['netdev'] = str(e)
        result['netdev'] = []

    # Collect /proc/version
    try:
        with open('/proc/version', 'r') as f:
            result['version'] = parse_proc_version(f.read())
    except Exception as e:
        result['errors']['version'] = str(e)
        result['version'] = ''

    # Collect /proc/loadavg
    try:
        with open('/proc/loadavg', 'r') as f:
            load_1, load_5, load_15, running, total = parse_proc_loadavg(f.read())
            result['loadavg'] = {
                'load_1min': load_1,
                'load_5min': load_5,
                'load_15min': load_15,
                'running_processes': running,
                'total_processes': total
            }
    except Exception as e:
        result['errors']['loadavg'] = str(e)
        result['loadavg'] = {}

    # Collect /proc/uptime
    try:
        with open('/proc/uptime', 'r') as f:
            result['uptime_seconds'] = parse_proc_uptime(f.read())
    except Exception as e:
        result['errors']['uptime'] = str(e)
        result['uptime_seconds'] = 0.0

    # Collect /etc/os-release
    try:
        with open('/etc/os-release', 'r') as f:
            result['os_release'] = parse_os_release(f.read())
    except Exception as e:
        result['errors']['os_release'] = str(e)
        result['os_release'] = {}

    # Collect /sys/class/dmi/id/product_name → chassis_model (D-21, COLL-03).
    # collect_chassis_model swallows its own exceptions per D-2, so the
    # wrapper try/except here is defense-in-depth against an unexpected
    # surface (e.g., the path-indirection argument default being mutated
    # to a non-string by a future bug) — the universal-rule contract is
    # "the key is always present, the value is always a string".
    try:
        result['chassis_model'] = collect_chassis_model()
    except Exception as e:
        result['errors']['chassis_model'] = str(e)
        result['chassis_model'] = ''

    # Collect networking inventory via /sys/class/net + /sys/class/infiniband
    # (D-18 filter scope, D-19 IB-first, D-20 operstate + effective-state
    # demotion, COLL-04). collect_networking applies per-iface D-2 internally
    # so individual hot-unplug / permission failures don't surface here; the
    # outer try/except is defense-in-depth against an unexpected listdir
    # failure surfacing through the top-level walk.
    try:
        result['networking'] = collect_networking()
    except Exception as e:
        result['errors']['networking'] = str(e)
        result['networking'] = []

    # Collect /proc/sys walk → sysctl[] (D-27 allowlist, D-28 walk semantics,
    # D-29 multi-value verbatim emit, COLL-05). collect_sysctl applies per-leaf
    # D-2 internally so individual write-only leaves (vm.drop_caches, sysrq)
    # don't surface here; the outer try/except mirrors the chassis_model /
    # networking shape as defense-in-depth.
    try:
        result['sysctl'] = collect_sysctl()
    except Exception as e:
        result['errors']['sysctl'] = str(e)
        result['sysctl'] = []

    # Collect os.environ → environment[] (D-23 KEY_ID mask, D-24 SECRET length-only,
    # D-25 unified redactors, D-26 prefix-or-literal allowlist, COLL-06).
    # collect_environment applies the D-2 envelope internally and never raises;
    # the outer try/except mirrors the chassis_model / networking / sysctl shape
    # as defense-in-depth.
    try:
        result['environment'] = collect_environment()
    except Exception as e:
        result['errors']['environment'] = str(e)
        result['environment'] = []

    # Collect lsblk -J -b -d → drives[] (D-30 emit shape, D-31 four-rule filter
    # chain, D-33 absent/empty → [] universal-failure rule, COLL-07).
    # collect_drives applies the D-2 envelope internally (subprocess failure +
    # JSON parse + per-row); the outer try/except mirrors chassis_model /
    # networking / sysctl / environment shape as defense-in-depth.
    try:
        result['drives'] = collect_drives()
    except Exception as e:
        result['errors']['drives'] = str(e)
        result['drives'] = []

    # Collect /proc/vmstat
    try:
        with open('/proc/vmstat', 'r') as f:
            result['vmstat'] = parse_proc_vmstat(f.read())
    except Exception as e:
        result['errors']['vmstat'] = str(e)
        result['vmstat'] = {}

    # Collect /proc/mounts (filesystems)
    try:
        with open('/proc/mounts', 'r') as f:
            mounts = parse_proc_mounts(f.read())
            result['mounts'] = [m.to_dict() for m in mounts]
    except Exception as e:
        result['errors']['mounts'] = str(e)
        result['mounts'] = []

    # Collect /proc/cgroups
    try:
        with open('/proc/cgroups', 'r') as f:
            cgroups = parse_proc_cgroups(f.read())
            result['cgroups'] = [c.to_dict() for c in cgroups]
    except Exception as e:
        result['errors']['cgroups'] = str(e)
        result['cgroups'] = []

    # Remove errors dict if empty
    if not result['errors']:
        del result['errors']

    return result


def summarize_cpuinfo(cpuinfo_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize CPU information from parsed cpuinfo.

    Args:
        cpuinfo_list: List of CPU dictionaries from parse_proc_cpuinfo.

    Returns:
        Dictionary with:
        - num_logical_cores: Total number of logical CPUs
        - num_physical_cores: Number of physical cores (if available)
        - model: CPU model name
        - architecture: CPU architecture (from flags or model)
        - cpu_mhz: CPU frequency in MHz (if available)
        - physical_ids: Set of unique physical CPU IDs
    """
    if not cpuinfo_list:
        return {
            'num_logical_cores': 0,
            'num_physical_cores': 0,
            'model': '',
            'architecture': '',
        }

    num_logical = len(cpuinfo_list)
    model = cpuinfo_list[0].get('model name', '')

    # Count physical cores by unique (physical id, core id) pairs
    physical_ids = set()
    core_pairs = set()
    cpu_mhz = None

    for cpu in cpuinfo_list:
        phys_id = cpu.get('physical id')
        core_id = cpu.get('core id')
        if phys_id is not None:
            physical_ids.add(phys_id)
        if phys_id is not None and core_id is not None:
            core_pairs.add((phys_id, core_id))
        if cpu_mhz is None and 'cpu MHz' in cpu:
            cpu_mhz = cpu['cpu MHz']

    # If we couldn't determine physical cores, assume 1 core per logical
    num_physical = len(core_pairs) if core_pairs else num_logical

    # Try to determine architecture from flags or model name
    architecture = ''
    flags = cpuinfo_list[0].get('flags', '')
    if isinstance(flags, str):
        if 'lm' in flags.split():  # long mode = x86_64
            architecture = 'x86_64'
        elif 'tm' in flags.split():
            architecture = 'i686'

    if not architecture and 'aarch64' in model.lower():
        architecture = 'aarch64'
    elif not architecture and ('x86' in model.lower() or 'intel' in model.lower() or 'amd' in model.lower()):
        architecture = 'x86_64'

    result = {
        'num_logical_cores': num_logical,
        'num_physical_cores': num_physical,
        'model': model,
        'architecture': architecture,
        'num_sockets': len(physical_ids) if physical_ids else 1,
    }

    if cpu_mhz is not None:
        result['cpu_mhz'] = cpu_mhz

    return result


# =============================================================================
# MPI Collection Script Generator
# =============================================================================

# The MPI collection script is embedded as a string template to avoid
# dependency issues when running on remote nodes
MPI_COLLECTOR_SCRIPT = '''#!/usr/bin/env python3
"""
MPI System Information Collector - Generated by MLPerf Storage.

This script is executed via MPI on all nodes to collect system information.
It gathers data from /proc files and aggregates results on rank 0.
"""

import fnmatch
import json
import os
import re
import socket
import subprocess
import sys
import time


def parse_proc_meminfo(content):
    """Parse /proc/meminfo content into a dictionary."""
    result = {}
    for line in content.strip().split('\\n'):
        if not line or ':' not in line:
            continue
        parts = line.split(':')
        if len(parts) != 2:
            continue
        key = parts[0].strip()
        value_parts = parts[1].strip().split()
        if value_parts:
            try:
                result[key] = int(value_parts[0])
            except ValueError:
                continue
    return result


def parse_proc_cpuinfo(content):
    """Parse /proc/cpuinfo content into a list of CPU dictionaries."""
    cpus = []
    current_cpu = {}

    for line in content.strip().split('\\n'):
        line = line.strip()
        if not line:
            if current_cpu:
                cpus.append(current_cpu)
                current_cpu = {}
            continue

        if ':' not in line:
            continue

        parts = line.split(':', 1)
        if len(parts) != 2:
            continue

        key = parts[0].strip()
        value = parts[1].strip()

        try:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass

        current_cpu[key] = value

    if current_cpu:
        cpus.append(current_cpu)

    return cpus


def parse_proc_diskstats(content):
    """Parse /proc/diskstats content into a list of disk info dicts."""
    disks = []

    for line in content.strip().split('\\n'):
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) < 14:
            continue

        device_name = parts[2]

        try:
            disk_info = {
                'device_name': device_name,
                'reads_completed': int(parts[3]),
                'reads_merged': int(parts[4]),
                'sectors_read': int(parts[5]),
                'time_reading_ms': int(parts[6]),
                'writes_completed': int(parts[7]),
                'writes_merged': int(parts[8]),
                'sectors_written': int(parts[9]),
                'time_writing_ms': int(parts[10]),
                'ios_in_progress': int(parts[11]),
                'time_doing_ios_ms': int(parts[12]),
                'weighted_time_doing_ios_ms': int(parts[13]),
            }

            if len(parts) >= 18:
                disk_info['discards_completed'] = int(parts[14])
                disk_info['discards_merged'] = int(parts[15])
                disk_info['sectors_discarded'] = int(parts[16])
                disk_info['time_discarding_ms'] = int(parts[17])

            if len(parts) >= 20:
                disk_info['flush_requests_completed'] = int(parts[18])
                disk_info['time_flushing_ms'] = int(parts[19])

            disks.append(disk_info)

        except (ValueError, IndexError):
            continue

    return disks


def parse_proc_net_dev(content):
    """Parse /proc/net/dev content into a list of network info dicts."""
    interfaces = []
    lines = content.strip().split('\\n')

    for line in lines[2:]:
        if not line.strip() or ':' not in line:
            continue

        parts = line.split(':')
        if len(parts) != 2:
            continue

        interface_name = parts[0].strip()
        stats = parts[1].split()

        if len(stats) < 16:
            continue

        try:
            net_info = {
                'interface_name': interface_name,
                'rx_bytes': int(stats[0]),
                'rx_packets': int(stats[1]),
                'rx_errors': int(stats[2]),
                'rx_dropped': int(stats[3]),
                'rx_fifo': int(stats[4]),
                'rx_frame': int(stats[5]),
                'rx_compressed': int(stats[6]),
                'rx_multicast': int(stats[7]),
                'tx_bytes': int(stats[8]),
                'tx_packets': int(stats[9]),
                'tx_errors': int(stats[10]),
                'tx_dropped': int(stats[11]),
                'tx_fifo': int(stats[12]),
                'tx_collisions': int(stats[13]),
                'tx_carrier': int(stats[14]),
                'tx_compressed': int(stats[15]),
            }
            interfaces.append(net_info)
        except (ValueError, IndexError):
            continue

    return interfaces


def parse_os_release(content):
    """Parse /etc/os-release content into a dictionary."""
    result = {}
    for line in content.strip().split('\\n'):
        line = line.strip()
        if not line or '=' not in line:
            continue
        key, _, value = line.partition('=')
        value = value.strip('"\\\'')
        result[key] = value
    return result


# Phase 3 Plan 02 (D-21, COLL-03) — Pattern B (RESEARCH 675-679) duplication
# of the chassis-model collector into the MPI worker script. Must stay in
# behavioral parity with the module-level versions in this same file; the
# parity test in tests/unit/test_cluster_collector.py::TestMPIScriptParity
# fails loudly on any drift between the two copies. Untyped form (no Final[],
# no frozenset[str] subscript) so the script survives on Python 3.8 hosts
# in heterogeneous SSH-fan-out fleets — see PLAN.md Task 2 Step 3 note.
_DMI_PRODUCT_NAME_PATH = '/sys/class/dmi/id/product_name'

_DMI_PLACEHOLDERS = frozenset({
    "",
    "to be filled by o.e.m.",
    "default string",
    "system product name",
    "system manufacturer",
    "none",
    "not specified",
    "not applicable",
    "oem",
    "unknown",
})


def _normalize_dmi(s):
    """Normalize a DMI product_name string per D-21 (mirror of module copy)."""
    stripped = s.strip()
    if stripped.lower() in _DMI_PLACEHOLDERS:
        return ""
    return stripped


def collect_chassis_model(dmi_path=_DMI_PRODUCT_NAME_PATH):
    """Read DMI product_name with universal-failure rule (mirror of module copy)."""
    try:
        with open(dmi_path, 'r') as f:
            raw = f.read(8192)
        return _normalize_dmi(raw)
    except Exception:
        return ""


# Phase 3 Plan 03 (D-18, D-19, D-20, COLL-04) — Pattern B (RESEARCH 675-679)
# duplication of the networking sysfs + IB walk. Must stay in behavioral
# parity with the module-level versions; the parity test in
# tests/unit/test_cluster_collector.py::TestNetworkingMPIScriptParity
# fails on any drift. Untyped form so the script survives on Python 3.8
# hosts in heterogeneous SSH-fan-out fleets.
_SYSFS_NET_ROOT = '/sys/class/net'
_SYSFS_INFINIBAND_ROOT = '/sys/class/infiniband'

_VIRTUAL_NAME_PREFIXES = (
    'lo', 'docker', 'virbr', 'veth', 'tun', 'tap', 'gre', 'wg',
    'ib', 'iboeth', 'ib_eth',
)

_VIRTUAL_NAME_RE = re.compile(
    r'^(lo|docker[0-9]*|virbr[0-9]*|veth.*|tun[0-9]*|tap[0-9]*|'
    r'gre[0-9]*|wg[0-9]*|ib[0-9]*|iboeth[0-9]*|ib_eth[0-9]*)$'
)

_OPERSTATE_UP_VALUES = frozenset(['up', 'unknown'])

_SAFE_IFACE_NAME_RE = re.compile(r'^[A-Za-z0-9._-]+$')


def _read_sysfs_text(path):
    try:
        with open(path, 'r') as f:
            return f.read(8192).strip()
    except Exception:
        return ''


def _read_sysfs_int(path, default=-1):
    try:
        with open(path, 'r') as f:
            raw = f.read(8192).strip()
        return int(raw.split()[0])
    except Exception:
        return default


def _is_virtual_by_name(iface):
    return _VIRTUAL_NAME_RE.match(iface) is not None


def _is_bridge_master(iface_dir):
    return os.path.isdir(os.path.join(iface_dir, 'bridge'))


def _is_vlan_subif(iface_dir):
    iflink_path = os.path.join(iface_dir, 'iflink')
    ifindex_path = os.path.join(iface_dir, 'ifindex')
    if not (os.path.exists(iflink_path) and os.path.exists(ifindex_path)):
        return False
    iflink = _read_sysfs_int(iflink_path, default=-1)
    ifindex = _read_sysfs_int(ifindex_path, default=-1)
    if iflink == -1 or ifindex == -1:
        return False
    return iflink != ifindex


def _is_bond_slave(iface_dir):
    master_path = os.path.join(iface_dir, 'master')
    if not os.path.exists(master_path):
        return False
    try:
        target = os.readlink(master_path)
        return os.path.basename(target).startswith('bond')
    except OSError:
        return False


def _is_bond_master(iface_dir):
    return os.path.isdir(os.path.join(iface_dir, 'bonding'))


def _bond_aggregate_speed_mbps(iface_dir, net_root):
    slaves_path = os.path.join(iface_dir, 'bonding', 'slaves')
    raw = _read_sysfs_text(slaves_path)
    if not raw:
        return 0
    total = 0
    for name in raw.split():
        if not _SAFE_IFACE_NAME_RE.match(name):
            continue
        speed = _read_sysfs_int(os.path.join(net_root, name, 'speed'),
                                default=-1)
        if speed > 0:
            total += speed
    return total


def _map_operstate(operstate):
    return 'up' if operstate.strip().lower() in _OPERSTATE_UP_VALUES else 'down'


def _parse_ib_state(state_file_contents):
    return 'up' if state_file_contents.strip().startswith('4:') else 'down'


def _parse_ib_rate(rate_file_contents):
    try:
        return int(rate_file_contents.split()[0])
    except (ValueError, IndexError):
        return None


def collect_networking(net_root=_SYSFS_NET_ROOT, ib_root=_SYSFS_INFINIBAND_ROOT):
    """Enumerate real per-host NICs and IB ports (mirror of module copy)."""
    out = []

    try:
        net_entries = os.listdir(net_root)
    except Exception:
        net_entries = []

    for iface in net_entries:
        try:
            if not _SAFE_IFACE_NAME_RE.match(iface):
                continue
            if _is_virtual_by_name(iface):
                continue
            iface_dir = os.path.join(net_root, iface)
            if _is_bridge_master(iface_dir):
                continue
            if _is_vlan_subif(iface_dir):
                continue
            if _is_bond_slave(iface_dir):
                continue
            if _is_bond_master(iface_dir):
                aggregate_mbps = _bond_aggregate_speed_mbps(iface_dir, net_root)
                if aggregate_mbps > 0:
                    out.append({
                        'type': 'ethernet',
                        'speed': aggregate_mbps // 1000,
                        'state': 'up',
                    })
                else:
                    out.append({'type': 'ethernet', 'state': 'down'})
                continue

            iface_type = _read_sysfs_text(os.path.join(iface_dir, 'type'))
            if iface_type != '1':
                continue

            operstate_raw = _read_sysfs_text(os.path.join(iface_dir, 'operstate'))
            mapped_state = _map_operstate(operstate_raw)
            speed_mbps = _read_sysfs_int(os.path.join(iface_dir, 'speed'),
                                         default=-1)
            if mapped_state == 'up' and speed_mbps in (-1, 0):
                mapped_state = 'down'

            if mapped_state == 'up':
                out.append({
                    'type': 'ethernet',
                    'speed': speed_mbps // 1000,
                    'state': 'up',
                })
            else:
                out.append({'type': 'ethernet', 'state': 'down'})
        except Exception:
            continue

    if os.path.isdir(ib_root):
        try:
            ib_devs = os.listdir(ib_root)
        except Exception:
            ib_devs = []
        for dev in ib_devs:
            if not _SAFE_IFACE_NAME_RE.match(dev):
                continue
            ports_dir = os.path.join(ib_root, dev, 'ports')
            if not os.path.isdir(ports_dir):
                continue
            try:
                ports = os.listdir(ports_dir)
            except Exception:
                continue
            for port in ports:
                try:
                    if not _SAFE_IFACE_NAME_RE.match(port):
                        continue
                    port_dir = os.path.join(ports_dir, port)
                    state_raw = _read_sysfs_text(os.path.join(port_dir, 'state'))
                    ib_state = _parse_ib_state(state_raw)
                    if ib_state == 'up':
                        rate_raw = _read_sysfs_text(os.path.join(port_dir, 'rate'))
                        speed = _parse_ib_rate(rate_raw)
                        if speed is None:
                            out.append({'type': 'infiniband', 'state': 'down'})
                        else:
                            out.append({
                                'type': 'infiniband',
                                'speed': speed,
                                'state': 'up',
                            })
                    else:
                        out.append({'type': 'infiniband', 'state': 'down'})
                except Exception:
                    continue

    return out


# Phase 4 Plan 04-01 (D-27, D-28, D-29, COLL-05) — Pattern B (RESEARCH 675-679)
# duplication of the sysctl collector. Must stay in behavioral parity with the
# module-level versions; the parity test in
# tests/unit/test_cluster_collector.py::TestSysctlMPIScriptParity fails on any
# drift. Untyped form so the script survives on Python 3.8 hosts in
# heterogeneous SSH-fan-out fleets.
#
# Pattern B forbids file I/O for package-data lookups inside the script (the
# script ships as a string and is exec'd over SSH; there's no installed
# package on every host). The allowlist is therefore baked in as a tuple
# literal here. SOURCE OF TRUTH for the four globs is
# mlpstorage_py/system_description/sysctl_allowlist.txt; keep this tuple in
# sync with that file — the parity test asserts behavioral equivalence, not
# allowlist-content equivalence, so a manual sync between the two copies is
# the load-bearing discipline. (Future editor: if you add a glob to the
# shipped file, also add it here.)
_PROC_SYS_ROOT = '/proc/sys'

_SYSCTL_ALLOWLIST_LINES = (
    'vm.dirty_*',
    'net.core.*',
    'net.ipv4.tcp_*',
    'kernel.numa_balancing',
)


def _load_sysctl_allowlist():
    """Compile the baked-in allowlist tuple into a tuple of regex objects
    via fnmatch.translate (mirror of module copy)."""
    return tuple(re.compile(fnmatch.translate(g)) for g in _SYSCTL_ALLOWLIST_LINES)


def collect_sysctl(proc_sys_root=_PROC_SYS_ROOT, allowlist=None):
    """Walk /proc/sys with per-leaf D-2 isolation (mirror of module copy).

    D-29: trailing newline stripped only; internal tabs preserved verbatim
    so multi-value leaves like net.ipv4.tcp_rmem round-trip cleanly.
    D-28: 8 KiB read cap on each leaf.
    """
    if allowlist is None:
        allowlist = _load_sysctl_allowlist()
    out = []
    try:
        for dirpath, _dirnames, filenames in os.walk(proc_sys_root):
            for filename in filenames:
                leaf = os.path.join(dirpath, filename)
                rel = os.path.relpath(leaf, proc_sys_root)
                name = rel.replace(os.sep, '.')
                if not any(p.match(name) for p in allowlist):
                    continue
                try:
                    with open(leaf, 'r') as f:
                        raw = f.read(8192)
                except Exception:
                    continue
                out.append({"name": name, "value": raw.rstrip('\\n')})
    except Exception:
        return []
    return out


# Phase 4 Plan 04-02 (D-23, D-24, D-25, D-26, COLL-06) — Pattern B (RESEARCH
# 675-679) duplication of the environment collector + the two unified
# redactors. Must stay in behavioral parity with the module-level versions in
# storage_config.py and the module copy in this file; the parity test in
# tests/unit/test_cluster_collector.py::TestEnvironmentMPIScriptParity fails
# on any drift. Untyped form so the script survives on Python 3.8 hosts in
# heterogeneous SSH-fan-out fleets.
#
# The script cannot import storage_config (it runs as a generated string
# over SSH on hosts that may not have the mlpstorage_py package installed),
# so both redactors are inlined here.
_ENV_LITERALS = ("BUCKET",)

_ENV_PREFIXES = ("AWS_", "STORAGE_", "OMPI_", "UCX_", "NCCL_")

# Runtime-volatile launcher metadata (mirror of module copy, see UAT Test 3
# / LIFE-04). Must match the module-level _ENV_RUNTIME_DENYLIST byte-for-byte
# or TestEnvironmentMPIScriptParity will trip.
_ENV_RUNTIME_DENYLIST = (
    "OMPI_ARGV",
    "OMPI_FILE_LOCATION",
    "OMPI_MCA_ess_base_jobid",
    "OMPI_MCA_orte_hnp_uri",
    "OMPI_MCA_orte_jobfam_session_dir",
    "OMPI_MCA_orte_local_daemon_uri",
    "OMPI_MCA_orte_precondition_transports",
)


def _redact_secret(val):
    """Length-only credential redactor (mirror of storage_config copy, D-24)."""
    if val is None:
        return "[not set]"
    if val == "":
        return "[SET — empty]"
    return "[SET — " + str(len(val)) + " chars]"


def _mask_credential_id(val):
    """First-4/last-4 mask (mirror of storage_config copy, D-23)."""
    if val is None:
        return "[not set]"
    if val == "":
        return "[SET — empty]"
    if len(val) < 8:
        return "****"
    return val[:4] + "****" + val[-4:]


def _env_allowlist_match(name):
    """Prefix-or-literal allowlist match + runtime-volatile denylist (mirror
    of module copy, D-26 + Phase 5 UAT Test 3 fix)."""
    if name in _ENV_RUNTIME_DENYLIST:
        return False
    return name in _ENV_LITERALS or name.startswith(_ENV_PREFIXES)


def collect_environment():
    """Filter os.environ to allowlist with credential dispatch (mirror of
    module copy, D-23 / D-24 / D-26). Sorted by name for D-34 stability.
    D-2 envelope: any exception yields []."""
    try:
        out = []
        for name, value in sorted(os.environ.items()):
            if not _env_allowlist_match(name):
                continue
            if name == "AWS_ACCESS_KEY_ID":
                value = _mask_credential_id(value)
            elif name == "AWS_SECRET_ACCESS_KEY":
                value = _redact_secret(value)
            out.append({"name": name, "value": value})
        return out
    except Exception:
        return []


# ----- Phase 4 / Plan 04-03 drives collector (Pattern B, D-36) -----
#
# Untyped form to survive on Python 3.8 hosts (no Final[], no PEP-585
# subscripted generics, no Optional[] annotation in signatures). The module
# copy in `collect_drives()` carries the typed form. Drift between the two
# copies is caught by TestDrivesMPIScriptParity.
_LSBLK_ARGS = (
    'lsblk', '-J', '-b', '-d', '-o', 'NAME,MODEL,VENDOR,SIZE,ROTA,TRAN,RM',
)

_DRIVE_VIRTUAL_NAME_PREFIXES = (
    'loop', 'dm-', 'zram', 'ram', 'sr', 'fd',
)

_DRIVE_VIRTUAL_TRANS = frozenset(['loop', 'zram'])

_DRIVE_ACCEPTED_TRANS = frozenset(['nvme', 'sata', 'sas'])


def collect_drives():
    """Return one dict per accepted block device (mirror of module copy,
    D-30 emit, D-31 four-rule filter, D-33 absent/empty → []).
    D-2 envelope: subprocess / JSON / per-row failure → []."""
    try:
        cp = subprocess.run(
            list(_LSBLK_ARGS),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if cp.returncode != 0:
            return []
        payload = json.loads(cp.stdout)
        rows = payload.get('blockdevices', []) or []
        out = []
        for row in rows:
            try:
                # D-31 rule 1: removable skip — string/int/bool variants.
                rm = row.get('rm', 0)
                if rm in (1, True, '1'):
                    continue
                name = row.get('name', '') or ''
                if name.startswith(_DRIVE_VIRTUAL_NAME_PREFIXES):
                    continue
                tran = (row.get('tran') or '').lower()
                if tran in _DRIVE_VIRTUAL_TRANS:
                    continue
                if tran not in _DRIVE_ACCEPTED_TRANS:
                    if tran == '' and name.startswith('nvme'):
                        tran = 'nvme'
                    else:
                        continue
                capacity_in_GB = int(row['size']) // 10**9
                out.append({
                    'vendor_name':    (row.get('vendor') or '').strip(),
                    'model_name':     (row.get('model') or '').strip(),
                    'interface':      tran,
                    'capacity_in_GB': capacity_in_GB,
                })
            except Exception:
                continue
        return out
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        subprocess.SubprocessError,
        json.JSONDecodeError,
        Exception,
    ):
        return []


def collect_local_info():
    """Collect system information from the local node."""
    result = {
        'hostname': socket.gethostname(),
        'collection_timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'errors': {}
    }

    # Collect /proc/meminfo
    try:
        with open('/proc/meminfo', 'r') as f:
            result['meminfo'] = parse_proc_meminfo(f.read())
    except Exception as e:
        result['errors']['meminfo'] = str(e)
        result['meminfo'] = {}

    # Collect /proc/cpuinfo
    try:
        with open('/proc/cpuinfo', 'r') as f:
            result['cpuinfo'] = parse_proc_cpuinfo(f.read())
    except Exception as e:
        result['errors']['cpuinfo'] = str(e)
        result['cpuinfo'] = []

    # Collect /proc/diskstats
    try:
        with open('/proc/diskstats', 'r') as f:
            result['diskstats'] = parse_proc_diskstats(f.read())
    except Exception as e:
        result['errors']['diskstats'] = str(e)
        result['diskstats'] = []

    # Collect /proc/net/dev
    try:
        with open('/proc/net/dev', 'r') as f:
            result['netdev'] = parse_proc_net_dev(f.read())
    except Exception as e:
        result['errors']['netdev'] = str(e)
        result['netdev'] = []

    # Collect /proc/version
    try:
        with open('/proc/version', 'r') as f:
            result['version'] = f.read().strip()
    except Exception as e:
        result['errors']['version'] = str(e)
        result['version'] = ''

    # Collect /proc/loadavg
    try:
        with open('/proc/loadavg', 'r') as f:
            parts = f.read().strip().split()
            proc_parts = parts[3].split('/') if len(parts) >= 4 else ['0', '0']
            result['loadavg'] = {
                'load_1min': float(parts[0]) if parts else 0.0,
                'load_5min': float(parts[1]) if len(parts) > 1 else 0.0,
                'load_15min': float(parts[2]) if len(parts) > 2 else 0.0,
                'running_processes': int(proc_parts[0]) if proc_parts else 0,
                'total_processes': int(proc_parts[1]) if len(proc_parts) > 1 else 0
            }
    except Exception as e:
        result['errors']['loadavg'] = str(e)
        result['loadavg'] = {}

    # Collect /proc/uptime
    try:
        with open('/proc/uptime', 'r') as f:
            parts = f.read().strip().split()
            result['uptime_seconds'] = float(parts[0]) if parts else 0.0
    except Exception as e:
        result['errors']['uptime'] = str(e)
        result['uptime_seconds'] = 0.0

    # Collect /etc/os-release
    try:
        with open('/etc/os-release', 'r') as f:
            result['os_release'] = parse_os_release(f.read())
    except Exception as e:
        result['errors']['os_release'] = str(e)
        result['os_release'] = {}

    # Collect /sys/class/dmi/id/product_name → chassis_model (D-21, COLL-03).
    # Parallel to the module-side wiring in collect_local_system_info;
    # Pattern B duplication discipline (RESEARCH 675-679) keeps the MPI
    # fan-out path producing the same per-host data shape as the local
    # fallback. collect_chassis_model swallows its own exceptions per D-2;
    # the wrapper try/except is defense-in-depth.
    try:
        result['chassis_model'] = collect_chassis_model()
    except Exception as e:
        result['errors']['chassis_model'] = str(e)
        result['chassis_model'] = ''

    # Collect networking inventory (D-18 / D-19 / D-20, COLL-04) — Pattern B
    # parallel to the module-side wiring in collect_local_system_info. The
    # collect_networking function applies per-iface D-2 internally; this
    # outer try/except is defense-in-depth.
    try:
        result['networking'] = collect_networking()
    except Exception as e:
        result['errors']['networking'] = str(e)
        result['networking'] = []

    # Collect /proc/sys walk → sysctl[] (D-27, D-28, D-29, COLL-05) — Pattern B
    # parallel to the module-side wiring in collect_local_system_info. The
    # collect_sysctl function applies per-leaf D-2 internally; this outer
    # try/except is defense-in-depth.
    try:
        result['sysctl'] = collect_sysctl()
    except Exception as e:
        result['errors']['sysctl'] = str(e)
        result['sysctl'] = []

    # Collect os.environ → environment[] (D-23 / D-24 / D-26, COLL-06) —
    # Pattern B parallel to the module-side wiring in collect_local_system_info.
    # collect_environment applies D-2 internally; this outer try/except is
    # defense-in-depth.
    try:
        result['environment'] = collect_environment()
    except Exception as e:
        result['errors']['environment'] = str(e)
        result['environment'] = []

    # Collect lsblk -J -b -d → drives[] (D-30 / D-31 / D-33, COLL-07) —
    # Pattern B parallel to the module-side wiring. collect_drives applies
    # the D-2 envelope internally (subprocess + JSON + per-row); this outer
    # try/except is defense-in-depth.
    try:
        result['drives'] = collect_drives()
    except Exception as e:
        result['errors']['drives'] = str(e)
        result['drives'] = []

    if not result['errors']:
        del result['errors']

    return result


def main():
    """Main entry point for MPI collection."""
    output_file = sys.argv[1] if len(sys.argv) > 1 else '/tmp/mlps_cluster_info.json'

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    except ImportError as e:
        # mpi4py not available - this is a critical error when running under MPI
        # because each rank would write to the same file, corrupting the output.
        # Write an error marker and exit with non-zero code so the launcher
        # knows MPI collection failed and can fall back to local-only collection.
        error_output = {
            '_mpi_import_error': True,
            '_error_message': f'mpi4py not available: {e}',
            '_hostname': socket.gethostname(),
        }
        with open(output_file, 'w') as f:
            json.dump(error_output, f, indent=2)
        sys.exit(1)

    # Collect local info — wrap in try/except so every rank always reaches
    # comm.gather(); an early exit from any rank would deadlock all others.
    try:
        local_info = collect_local_info()
        local_info['mpi_rank'] = rank
    except Exception as e:
        local_info = {
            'hostname': socket.gethostname(),
            'mpi_rank': rank,
            '_collection_error': str(e),
        }

    # Gather all info to rank 0 — every rank must reach this call
    all_info = comm.gather(local_info, root=0)

    if rank == 0:
        # Combine results by hostname
        output = {}
        for info in all_info:
            hostname = info.get('hostname', f'unknown_rank_{info.get("mpi_rank", "?")}')
            # If we have duplicate hostnames (multiple ranks per host),
            # just keep the first one
            if hostname not in output:
                output[hostname] = info

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)


if __name__ == '__main__':
    main()
'''


# =============================================================================
# CAP-02 Shared-Filesystem Probe Script (Phase 5 / Plan 05-04)
# =============================================================================
#
# A SEPARATE MPI heredoc from MPI_COLLECTOR_SCRIPT (per RESEARCH §A3 / D-36
# Pattern B): different lifecycle stage (pre-execution gate vs. start-of-run
# cluster snapshot) so the two scripts are NOT merged. The probe runs once
# per benchmark instance from `_pre_execution_gate`, after the CAP-01
# capacity check; rank 0 creates a per-run-uuid-suffixed sentinel file in the
# dataset destination, every rank `os.stat`s it, the MPI gather collects
# (st_dev, st_ino) tuples to rank 0, rank 0 enforces cardinality exactly 1
# (or fails fast with each hostname's reported tuple), unlinks in a finally
# block, sleeps 5.0s for storage quiesce (D-49), and all ranks reach a final
# MPI_Barrier so the measured workload starts simultaneously.
#
# References:
#   - D-36 Pattern B: script-side helpers are inlined in untyped form (no
#     Final[], no subscript generics, no `from typing import`) so the script
#     runs on remote hosts without depending on the module-level typing
#     imports. Mirrors MPI_COLLECTOR_SCRIPT's convention.
#   - D-43: per-instance sentinel suffix (uuid.uuid4().hex) — see
#     Benchmark.__init__ in mlpstorage_py/benchmarks/base.py for the
#     generation site.
#   - D-44: unlink failure in the finally block is a logger.warning, NOT a
#     raise — leftover sentinels are cosmetic.
#   - D-45: any per-rank failure (EACCES / ENOSPC / ENOENT / NFS-stale) or
#     cardinality > 1 raises FileSystemError BEFORE the workload begins.
#   - D-49: rank 0 sleeps 5.0s INSIDE the finally block, BEFORE the final
#     comm.Barrier(); the sleep is rank-0-only so non-rank-0 ranks don't
#     block the whole fleet for 5s.
#   - Pitfall 4 / A5 (LOAD-BEARING): rank 0 broadcasts status='fail' via
#     comm.bcast(status, root=0) BEFORE the final barrier when it is about
#     to raise; non-rank-0 ranks read the broadcast and raise FileSystemError
#     themselves so the gather/barrier protocol completes cleanly and no
#     rank silently proceeds into the workload.
#   - Pitfall 6: untyped script body only (no Final[], no `dict[str, int]`,
#     no `from typing import`).
#   - Pitfall 7: per-instance UUID lock — every Benchmark instance generates
#     its own self._run_uuid; the launcher passes it through to argv[2]
#     unchanged.
#   - Pitfall 8: mpi4py ImportError → write error-marker JSON + sys.exit(1)
#     (mirrors MPI_COLLECTOR_SCRIPT).
#
# argv contract (HARDEN-02 / D-54: 2 positionals; stdout-marker transport):
#   argv[1] = data_dir     (the destination to probe — typically args.data_dir
#                           or the resolved per-benchmark _capacity_gate_destination)
#   argv[2] = run_uuid     (the uuid.uuid4().hex suffix for the sentinel filename;
#                           supplied by the caller — the launcher MUST pass this
#                           through unchanged; the script MUST NOT generate its own)
#
# Result transport (HARDEN-02 / D-54 / D-55 stdout markers):
#   rank 0 emits three lines to stdout (flushed):
#     __CAP02_RESULT_BEGIN__
#     <single-line compact JSON payload>
#     __CAP02_RESULT_END__
#   Non-rank-0 ranks emit NOTHING to stdout (enforced by
#   tests/unit/test_cluster_collector.py::TestSharedFsProbeNonRank0Silence).
#   The launcher parses subprocess.run(...).stdout via the marker regex.
#
# JSON output schema (rank 0 only, between BEGIN/END markers):
#   {
#     "status": "ok" | "fail",
#     "ranks": [
#       {"hostname": str, "rank": int, "failure": None | dict, "st_dev": int|None, "st_ino": int|None},
#       ...
#     ],
#     "failure_summary": None | {
#       "kind": "cardinality" | "per_rank",
#       "message": str   # human-readable, used by the launcher verbatim
#     },
#     "unlink_warning": None | str    # set if rank-0 unlink failed (D-44 cosmetic)
#   }
#
# Exit codes:
#   0 on status='ok'
#   1 on status='fail' OR on any setup error (e.g., mpi4py ImportError)
#
SHARED_FS_PROBE_SCRIPT = '''#!/usr/bin/env python3
"""
CAP-02 Shared-Filesystem Probe Script — Generated by MLPerf Storage.

Pattern B per D-36: script-side body is untyped (no Final[], no subscript
generics, no `from typing import`). Runs once per benchmark instance from
`_pre_execution_gate` after CAP-01. See cluster_collector.py module docs
for the full argv / JSON / D-43 / D-44 / D-45 / D-49 / Pitfall 4 contract.
"""

import json
import os
import socket
import stat as stat_mod
import sys
import time


def _build_cardinality_message(payloads):
    """Build the verbatim multi-line error body for a cardinality > 1 fault.

    Format (D-45 + REQUIREMENTS.md CAP-02):

      CAP-02: shared-FS probe detected the data-dir is NOT the same filesystem
      on every participating host.
        host=<h1> rank=<r1> st_dev=<d1> st_ino=<i1>
        host=<h2> rank=<r2> st_dev=<d2> st_ino=<i2>
        ...
      This typically means one or more hosts have a local-disk path where a
      shared mount was expected.
    """
    lines = []
    lines.append(
        "CAP-02: shared-FS probe detected the data-dir is NOT the same "
        "filesystem on every participating host."
    )
    for p in payloads:
        lines.append(
            "  host={h} rank={r} st_dev={d} st_ino={i}".format(
                h=p.get("hostname", "?"),
                r=p.get("rank", "?"),
                d=p.get("st_dev"),
                i=p.get("st_ino"),
            )
        )
    # REQUIREMENTS.md CAP-02 verbatim hint lock (single-line literal so the
    # plan's `grep -c '...'` acceptance criterion matches exactly):
    lines.append("this typically means one or more hosts have a local-disk path where a shared mount was expected.")
    return "\\n".join(lines)


def _build_per_rank_message(payloads):
    """Build the verbatim multi-line error body for any per-rank failure.

    Mentions every failing rank's hostname + mode + errno + message so the
    operator can identify the failing node(s) in a heterogeneous fleet.
    """
    failed = [p for p in payloads if p.get("failure") is not None]
    lines = []
    lines.append(
        "CAP-02: shared-FS probe failed on one or more participating hosts."
    )
    for p in failed:
        f = p["failure"]
        lines.append(
            "  host={h} rank={r} mode={m} errno={e} message={msg}".format(
                h=p.get("hostname", "?"),
                r=p.get("rank", "?"),
                m=f.get("mode", "?"),
                e=f.get("errno", "?"),
                msg=f.get("message", ""),
            )
        )
    lines.append(
        "Verify the data-dir path is accessible and has correct "
        "permissions on every participating host."
    )
    return "\\n".join(lines)


def main():
    """Probe entry point. argv[1]=data_dir, argv[2]=run_uuid (HARDEN-02 D-54: 2 positionals)."""
    if len(sys.argv) < 3:
        # Pre-MPI error — emit error marker via stdout (rank-0 only convention
        # does not apply here: this fires BEFORE the mpi4py import succeeds,
        # so we cannot determine rank. Emitting on all ranks is acceptable
        # because every rank fails the same way and the launcher's regex
        # picks up the FIRST marker pair via non-greedy .*?).
        # CAP-02 stdout transport (D-54/D-55).
        try:
            print("__CAP02_RESULT_BEGIN__", flush=True)
            print(json.dumps({
                "_argv_error": True,
                "_error_message": "expected 2 argv positions: data_dir run_uuid",
                "_hostname": socket.gethostname(),
            }, separators=(",", ":")), flush=True)
            print("__CAP02_RESULT_END__", flush=True)
        except Exception:
            pass
        sys.exit(1)

    data_dir = sys.argv[1]
    run_uuid = sys.argv[2]

    # mpi4py import (deferred per Pitfall 8 / MPI_COLLECTOR_SCRIPT analog).
    try:
        from mpi4py import MPI
    except ImportError as e:
        # CAP-02 stdout transport (D-54/D-55): emitted on every rank that
        # cannot import mpi4py. Acceptable because the launcher's regex picks
        # up the FIRST marker pair (non-greedy .*?), and all ranks failing
        # mpi4py-import yield identical error_output dicts.
        error_output = {
            "_mpi_import_error": True,
            "_error_message": "mpi4py not available: {0}".format(e),
            "_hostname": socket.gethostname(),
        }
        try:
            print("__CAP02_RESULT_BEGIN__", flush=True)
            print(json.dumps(error_output, separators=(",", ":")), flush=True)
            print("__CAP02_RESULT_END__", flush=True)
        except Exception:
            pass
        sys.exit(1)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    sentinel = os.path.join(
        data_dir,
        ".mlpstorage-shared-fs-probe-" + run_uuid,
    )

    # Per-rank failure tracker. None = healthy; dict = failure to report.
    failure = None
    st_dev = None
    st_ino = None
    unlink_warning = None
    status = None  # rank 0 sets to 'ok' or 'fail'; non-rank-0 reads via bcast.
    rank0_failure_summary = None

    try:
        # ---- Step A: rank 0 atomically creates the sentinel (O_CREAT|O_EXCL).
        if rank == 0:
            try:
                fd = os.open(
                    sentinel,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
                os.close(fd)
            except OSError as e:
                failure = {
                    "mode": "sentinel_create",
                    "host": socket.gethostname(),
                    "errno": e.errno,
                    "message": str(e),
                }

        # ---- Step B: synchronize so non-rank-0 ranks don't stat too early.
        comm.Barrier()

        # ---- Step C: every rank stats the sentinel; report per-rank failure.
        if failure is None:
            try:
                st = os.stat(sentinel)
                st_dev = st.st_dev
                st_ino = st.st_ino
            except OSError as e:
                failure = {
                    "mode": "sentinel_stat",
                    "host": socket.gethostname(),
                    "errno": e.errno,
                    "message": str(e),
                }

        # ---- Step D: every rank packs its local payload + gather to rank 0.
        local_payload = {
            "hostname": socket.gethostname(),
            "rank": rank,
            "failure": failure,
            "st_dev": st_dev,
            "st_ino": st_ino,
        }
        all_payloads = comm.gather(local_payload, root=0)

        # ---- Step E: rank 0 analyzes the gather.
        if rank == 0:
            any_failure = any(p.get("failure") is not None for p in all_payloads)
            if any_failure:
                status = "fail"
                rank0_failure_summary = {
                    "kind": "per_rank",
                    "message": _build_per_rank_message(all_payloads),
                }
            else:
                ids = set()
                for p in all_payloads:
                    ids.add((p.get("st_dev"), p.get("st_ino")))
                if len(ids) != 1:
                    status = "fail"
                    rank0_failure_summary = {
                        "kind": "cardinality",
                        "message": _build_cardinality_message(all_payloads),
                    }
                else:
                    status = "ok"

        # ---- Step F: Pitfall 4 / A5 LOAD-BEARING — rank 0 broadcasts the
        # final status BEFORE the post-quiesce barrier so non-rank-0 ranks
        # know whether the run is healthy. Without this bcast, a rank-0
        # failure followed by a non-rank-0 success would let the fleet
        # silently proceed into the workload on N-1 nodes.
        status = comm.bcast(status, root=0)

    finally:
        # ---- Step G: rank 0 unlinks the sentinel (D-44 cosmetic; warns, not raises).
        if rank == 0:
            try:
                os.unlink(sentinel)
            except OSError as e:
                # D-44: leftover sentinels are cosmetic. Warn via stderr; the
                # launcher captures stderr and surfaces the warning at INFO
                # level. Do NOT raise.
                unlink_warning = "rank-0 unlink failed: {0}".format(e)
                try:
                    sys.stderr.write("WARNING: " + unlink_warning + "\\n")
                except Exception:
                    pass

        # ---- Step H: rank-0 D-49 quiesce sleep (5.0s) BEFORE the final barrier.
        # rank-0-only so non-rank-0 ranks don't block the whole fleet.
        if rank == 0:
            time.sleep(5.0)

        # ---- Step I: final fleet-wide barrier so the measured workload
        # starts simultaneously on every rank.
        comm.Barrier()

    # ---- Step J: rank 0 emits the JSON result via stdout markers; all ranks exit per status.
    # CAP-02 stdout transport (D-54/D-55): rank-0 only; non-rank-0 ranks MUST NOT
    # print to stdout — enforced by TestSharedFsProbeNonRank0Silence.
    if rank == 0:
        try:
            _result = {
                "status": status if status is not None else "fail",
                "ranks": all_payloads,
                "failure_summary": rank0_failure_summary,
                "unlink_warning": unlink_warning,
            }
            # Compact single-line JSON to stay under PIPE_BUF and avoid framing ambiguity.
            print("__CAP02_RESULT_BEGIN__", flush=True)
            print(json.dumps(_result, separators=(",", ":")), flush=True)
            print("__CAP02_RESULT_END__", flush=True)
        except Exception as e:
            try:
                sys.stderr.write(
                    "WARNING: rank-0 failed to emit probe output: {0}\\n".format(e)
                )
            except Exception:
                pass

    # Every rank exits based on the broadcast status.
    if status == "ok":
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
'''


# =============================================================================
# MPI Cluster Collector Class
# =============================================================================

class MPIClusterCollector:
    """
    Collects system information from all nodes in a cluster using MPI.

    This class generates a Python script that is executed via MPI on all nodes
    to collect and aggregate system information.

    Attributes:
        hosts: List of hostnames or IP addresses to collect from.
        mpi_bin: MPI command to use (mpirun or mpiexec).
        logger: Logger instance for output.
        allow_run_as_root: Whether to allow running MPI as root.
        timeout: Timeout in seconds for the collection.
    """

    def __init__(
        self,
        hosts: List[str],
        mpi_bin: str,
        logger,
        results_dir: str,
        allow_run_as_root: bool = False,
        timeout_seconds: int = 60,
        shared_staging_dir: Optional[str] = None,
        shared_tmp_dir: Optional[str] = None,  # deprecated, see note below
        ssh_username: Optional[str] = None,
    ):
        """
        Initialize the MPI cluster collector.

        Args:
            hosts: List of hostnames/IPs, optionally with slot counts (e.g., "host1:4").
            mpi_bin: MPI binary to use (MPIRUN or MPIEXEC constant).
            logger: Logger instance for messages.
            results_dir: Absolute or relative path to the benchmark results
                directory. The collector stages its helper script under
                ``<results_dir>/collector-staging/``; the staged script
                persists after the run as a debuggable artifact. This
                replaces the previous per-invocation ``tempfile`` staging
                directory so no programmatic ``rm -rf`` is ever issued over
                SSH (see PR #347 review).
            allow_run_as_root: If True, adds --allow-run-as-root flag.
            timeout_seconds: Maximum time to wait for collection.
            shared_staging_dir: Optional path that is visible on every node.
                When set, the collector writes the helper script under this
                path and skips SSH-based staging. Typically used on clusters
                with a shared scratch filesystem (NFS/Lustre/GPFS).
            shared_tmp_dir: Deprecated alias for ``shared_staging_dir``.
                Kept for one release for backward compatibility; emits a
                DeprecationWarning.
            ssh_username: Optional SSH username used when staging the script on
                remote hosts. Defaults to the current user. Ignored when
                ``shared_staging_dir`` is set or when all hosts are localhost.

        Raises:
            ValueError: if ``results_dir`` is empty or None. Multi-host
                collection without a results directory has no defensible
                staging location now that tempdir-based staging is gone.
        """
        if not results_dir:
            raise ValueError(
                "MPIClusterCollector requires results_dir for script staging"
            )

        # Backward compatibility for the old kwarg name. Drop in a future
        # release.
        if shared_tmp_dir is not None:
            warnings.warn(
                "shared_tmp_dir is deprecated; use shared_staging_dir instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if shared_staging_dir is None:
                shared_staging_dir = shared_tmp_dir

        self.hosts = hosts
        self.mpi_bin = mpi_bin
        self.logger = logger
        self.results_dir = os.path.abspath(results_dir)
        self.allow_run_as_root = allow_run_as_root
        self.timeout = timeout_seconds
        self.shared_staging_dir = (
            os.path.abspath(shared_staging_dir) if shared_staging_dir else None
        )
        self.ssh_username = ssh_username

    def _get_unique_hosts(self) -> List[str]:
        """Extract unique hostnames from the hosts list (removing slot counts)."""
        unique = []
        seen = set()
        for host in self.hosts:
            hostname = host.split(':')[0] if ':' in host else host
            if hostname not in seen:
                seen.add(hostname)
                unique.append(hostname)
        return unique

    def _generate_mpi_command(self, script_path: str, output_path: str) -> str:
        """
        Generate the MPI command to run the collection script.

        Args:
            script_path: Path to the generated collector script.
            output_path: Path where the JSON output should be written.

        Returns:
            Full MPI command string.
        """
        unique_hosts = self._get_unique_hosts()
        num_hosts = len(unique_hosts)

        # Build host string with 1 slot per host (we only need one process per node)
        host_slots = [f"{host}:1" for host in unique_hosts]

        # Select MPI binary
        if self.mpi_bin == MPIRUN:
            mpi_executable = MPI_RUN_BIN
        elif self.mpi_bin == MPIEXEC:
            mpi_executable = MPI_EXEC_BIN
        else:
            mpi_executable = self.mpi_bin

        cmd = f"{mpi_executable} -n {num_hosts} -host {','.join(host_slots)}"

        # Add common flags
        cmd += " --bind-to none --map-by node"

        if self.allow_run_as_root:
            cmd += " --allow-run-as-root"

        # Add the Python script and output path
        cmd += f" python3 {script_path} {output_path}"

        return cmd

    def _write_collector_script(self, script_path: str) -> None:
        """Write the collector script to the specified path."""
        with open(script_path, 'w') as f:
            f.write(MPI_COLLECTOR_SCRIPT)
        os.chmod(script_path, 0o755)

    def _ssh_target(self, host: str) -> str:
        """Return '[user@]host' for SSH/SCP invocations."""
        return f"{self.ssh_username}@{host}" if self.ssh_username else host

    def _ssh_common_opts(self) -> List[str]:
        """SSH/SCP options used for all staging operations.

        * ``BatchMode=yes`` — never prompt for a password; fail fast if
          passwordless SSH is not configured.
        * ``StrictHostKeyChecking=accept-new`` — accept new host keys on first
          contact but reject changed keys; matches the behavior users already
          have configured for ``mpirun``.
        * ``ForwardX11=no`` — suppress the ``Authorization required, but no
          authorization protocol specified`` noise seen in issue #303.
        * ``ConnectTimeout`` — bound per-host handshake time so a single bad
          host cannot consume the whole collection timeout budget.
        """
        return [
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ForwardX11=no",
            "-o", f"ConnectTimeout={max(5, self.timeout // 6)}",
        ]

    def _remote_hosts_needing_staging(self) -> List[str]:
        """Return remote (non-localhost) unique hosts that need the script."""
        return [h for h in self._get_unique_hosts() if not _is_localhost(h)]

    def _stage_script_on_remote_hosts(
        self,
        script_local_path: str,
        remote_dir: str,
        hosts: List[str],
    ) -> Dict[str, Optional[str]]:
        """SCP the collector script to ``remote_dir`` on each remote host.

        The per-host work is parallelised with a thread pool; each call is
        independent and almost entirely I/O-bound.

        Args:
            script_local_path: Path to the collector script on the launch host.
            remote_dir: Absolute directory to create on each remote host; the
                script will be placed at ``remote_dir/mlps_collector.py``. The
                same absolute path is used on every node so the ``mpirun``
                command line is identical everywhere.
            hosts: Remote hostnames to stage to. Callers should pass the result
                of :meth:`_remote_hosts_needing_staging` to avoid SSHing to the
                launch host.

        Returns:
            Mapping ``{host: None on success, error_message_str on failure}``.
        """
        per_host_timeout = max(10, self.timeout // 3)
        ssh_common = self._ssh_common_opts()

        def stage_one(host: str) -> Tuple[str, Optional[str]]:
            target = self._ssh_target(host)
            try:
                mkdir_cmd = [
                    "ssh", *ssh_common, target, f"mkdir -p '{remote_dir}'"
                ]
                r = subprocess.run(
                    mkdir_cmd, capture_output=True, text=True,
                    timeout=per_host_timeout,
                )
                if r.returncode != 0:
                    return host, f"ssh mkdir failed: {r.stderr.strip() or r.stdout.strip()}"

                scp_cmd = [
                    "scp", *ssh_common, script_local_path,
                    f"{target}:{remote_dir}/mlps_collector.py",
                ]
                r = subprocess.run(
                    scp_cmd, capture_output=True, text=True,
                    timeout=per_host_timeout,
                )
                if r.returncode != 0:
                    return host, f"scp failed: {r.stderr.strip() or r.stdout.strip()}"
                return host, None
            except subprocess.TimeoutExpired:
                return host, f"timed out after {per_host_timeout}s"
            except FileNotFoundError as e:
                return host, f"ssh/scp binary not found: {e}"
            except Exception as e:  # pragma: no cover — defensive
                return host, f"unexpected error: {e}"

        results: Dict[str, Optional[str]] = {}
        max_workers = min(16, max(1, len(hosts)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(stage_one, h): h for h in hosts}
            for f in as_completed(futures):
                host, err = f.result()
                results[host] = err
                if err:
                    self.logger.warning(
                        f"Script staging on {host} failed: {err}"
                    )
                else:
                    self.logger.info(
                        f"Collector script staged on {host}:{remote_dir}"
                    )
        return results

    def collect(self) -> Dict[str, Any]:
        """
        Execute MPI collection across all nodes.

        The collector script is written to ``<results_dir>/collector-staging/``
        on the launch host and the same absolute path is created on each
        remote host via SSH before the script is copied there with SCP.
        Because ``results_dir`` is resolved to an absolute path at
        construction time, the path is identical on every participating
        node, which is what ``mpirun`` requires.

        When ``shared_staging_dir`` is set the script is written under that
        path and no SSH staging is performed (suitable for clusters with a
        shared NFS/Lustre/GPFS scratch FS).

        The staged script is **not removed** at the end of the run — it is
        kept as a persistent run artifact so users can inspect it after a
        failure. This is a deliberate design choice (see PR #347 review)
        : programmatic ``rm -rf`` over SSH is
        unacceptable. Consecutive runs against the same ``results_dir``
        simply overwrite the script, which is safe and idempotent.

        This fixes issue #303, where the previous implementation assumed
        ``tempfile.TemporaryDirectory()`` on the launch host was visible to
        every rank.

        Returns:
            Dictionary mapping hostname -> system_info dict.

        Raises:
            RuntimeError: If MPI collection fails completely.
        """
        unique_hosts = self._get_unique_hosts()
        self.logger.debug(
            f"Starting MPI cluster collection on {len(unique_hosts)} hosts"
        )

        # --- Decide where to place the helper script ---------------------
        if self.shared_staging_dir:
            staging_dir = self.shared_staging_dir
            use_staging = False
            self.logger.debug(
                f"Using shared staging dir (no SSH staging): {staging_dir}"
            )
        else:
            staging_dir = os.path.join(self.results_dir, "collector-staging")
            use_staging = True

        script_path = os.path.join(staging_dir, "mlps_collector.py")
        output_path = os.path.join(staging_dir, "cluster_info.json")

        remote_hosts_to_stage: List[str] = []

        os.makedirs(staging_dir, exist_ok=True)
        self._write_collector_script(script_path)
        self.logger.info(
            f"Collector script staged at {script_path} "
            f"(persisted as run artifact)"
        )

        # --- Stage the script on remote hosts if needed ------------------
        if use_staging:
            remote_hosts_to_stage = self._remote_hosts_needing_staging()
            if remote_hosts_to_stage:
                self.logger.info(
                    f"Staging collector script to "
                    f"{len(remote_hosts_to_stage)} remote host(s)..."
                )
                stage_results = self._stage_script_on_remote_hosts(
                    script_path, staging_dir, remote_hosts_to_stage
                )
                failures = {
                    h: e for h, e in stage_results.items() if e
                }
                if failures:
                    raise RuntimeError(
                        "Failed to stage collector script on "
                        f"{len(failures)} host(s): {failures}. "
                        "Verify passwordless SSH from the launch host, or "
                        "set --cluster-collector-shared-staging / "
                        "MLPS_CLUSTER_COLLECTOR_SHARED_STAGING to a "
                        "directory visible on every node."
                    )

        # --- Build and run the mpirun command ----------------------------
        cmd = self._generate_mpi_command(script_path, output_path)
        self.logger.info(
            f"Running MPI collection across {len(unique_hosts)} host(s)"
        )
        self.logger.debug(f"MPI command: {cmd}")

        # Silence OpenSSH X11-forwarding warnings that mpirun's rsh/ssh
        # PLM emits when XAUTHORITY is not set on the launch host
        # ('Authorization required, but no authorization protocol
        # specified'). Reported in issue #303.
        env = os.environ.copy()
        env.pop("DISPLAY", None)      # prevent SSH X11 forwarding handshake
        env.pop("XAUTHORITY", None)   # and its cookie lookup
        env.setdefault(
            "PLM_RSH_AGENT",
            "ssh -o ForwardX11=no -o ForwardX11Trusted=no "
            "-o StrictHostKeyChecking=accept-new",
        )

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"MPI collection timed out after {self.timeout} seconds"
            )

        # --- Parse the output written by rank 0 --------------------------
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                collected_data = json.load(f)

            if collected_data.get('_mpi_import_error'):
                error_msg = collected_data.get(
                    '_error_message', 'mpi4py not available'
                )
                error_host = collected_data.get('_hostname', 'unknown')
                raise RuntimeError(
                    f"MPI collection failed on host '{error_host}': "
                    f"{error_msg}. Ensure mpi4py is installed on all "
                    "cluster nodes."
                )

            if result.returncode != 0:
                self.logger.warning(
                    f"MPI collection returned non-zero exit code: "
                    f"{result.returncode}\nstderr: {result.stderr}"
                )

            self.logger.info(
                f"MPI collection completed successfully "
                f"({len(collected_data)} hosts reported)"
            )
            return collected_data

        # No output file — surface staging + mpirun context together.
        # The staged script is left in place on both launch and remote
        # hosts so the failure can be diagnosed post-mortem.
        staged_summary = (
            remote_hosts_to_stage if remote_hosts_to_stage
            else "[launch host only]"
        )
        raise RuntimeError(
            "MPI collection did not produce output file. "
            f"Return code: {result.returncode}. "
            f"Staged on: {staged_summary}. "
            f"Staged script (persisted for inspection): {script_path}. "
            f"stderr: {result.stderr}"
        )

    def collect_local_only(self) -> Dict[str, Any]:
        """
        Collect system info from local node only (fallback when MPI unavailable).

        Returns:
            Dictionary with single hostname -> system_info mapping.
        """
        local_info = collect_local_system_info()
        return {local_info['hostname']: local_info}


# =============================================================================
# CAP-02 Shared-Filesystem Probe Launcher (Phase 5 / Plan 05-04)
# =============================================================================


def _write_probe_script_to_tempfile(script_str):
    """Write the probe script body to a tempfile and return its path.

    Module-level helper so the launcher does not need to instantiate
    MPIClusterCollector just to stage the script. Mirrors the body of
    MPIClusterCollector._write_collector_script.
    """
    import tempfile
    fd, path = tempfile.mkstemp(prefix="mlps_cap02_probe_", suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(script_str)
        os.chmod(path, 0o755)
    except Exception:
        try:
            os.unlink(path)
        except Exception:
            pass
        raise
    return path


# Compiled once at module load; consumed by run_shared_fs_probe AND by the
# four integration tests at tests/integration/test_shared_fs_probe_real_mpi.py
# — single source of truth prevents the HARDEN-04 regression from recurring.
_TAG_OUTPUT_PREFIX_RE = re.compile(r"^\[[^\]]+\](?:<[a-z]+>:?)?\s*")


def _strip_tag_output_prefix(line: str) -> str:
    """Strip the OpenMPI ``--tag-output`` prefix from a single line.

    Consumes both the ``[rank,jobid]`` bracketed identifier AND the
    optional ``<channel>:`` marker (OpenMPI 4.x emits ``<stdout>:``,
    ``<stderr>:``, ``<stddiag>:`` glued directly to the bracketed
    identifier; OpenMPI 5.x sometimes omits the trailing colon).

    HARDEN-04 background: the original regex ``r'^\\[[^\\]]+\\]\\s*'``
    (from HARDEN-02 GREEN at 086b2a9) assumed the OpenMPI prefix was
    ``[host:rank] `` (space-separated). That was wrong for OpenMPI 4.x;
    verified on 4.1.6 per the debug session at
    ``.planning/debug/cap02-stdout-empty-payload-tag-output-multihost.md``.
    This helper is the consolidated fix consumed by both the CAP-02
    launcher and all four integration tests at
    ``tests/integration/test_shared_fs_probe_real_mpi.py``.

    Args:
        line: A single stdout line possibly carrying the --tag-output
            prefix. Whitespace-stripped is recommended before calling.

    Returns:
        The line with the prefix consumed, or the original line if no
        prefix was present (backward-compat with non-tagged output).
    """
    return _TAG_OUTPUT_PREFIX_RE.sub("", line)


def run_shared_fs_probe(destination, hosts, run_uuid, logger,
                        mpi_bin=None, allow_run_as_root=False,
                        timeout_seconds=60, ssh_username=None):
    """Run the CAP-02 shared-FS probe across ``hosts`` rooted at ``destination``.

    Contract (Phase 5 / Plan 05-04 / D-43 / D-44 / D-45 / D-49 / Pitfall 4
    / Pitfall 7):

    * **Single-host short-circuit (SC#8 silence lock):** if ``hosts`` is
      None / empty / single-element, the probe is a no-op. NOTHING is
      logged at info/error/warning (debug only). No sentinel is created.
      No mpirun is invoked. This matches REQUIREMENTS.md CAP-02 SC#8 — a
      single-host run has no shared-FS surface to verify.
    * **W-5 launcher UUID pass-through (LOAD-BEARING):** the ``run_uuid``
      argument is passed through to the mpirun subprocess argv as argv[2]
      verbatim. The launcher MUST NOT generate its own UUID — the
      Benchmark instance's ``self._run_uuid`` (uuid.uuid4().hex generated
      once in Benchmark.__init__ per Pitfall 7) flows end-to-end through
      a single Benchmark instance.
    * **Multi-host probe path:** the SHARED_FS_PROBE_SCRIPT body is
      staged to a tempfile and run via mpirun across the unique hosts.
      Rank 0's JSON output is parsed; on status='fail' the launcher
      raises FileSystemError(code=ErrorCode.FS_INVALID_STRUCTURE) with
      the human-readable failure summary from the script.

    Args:
        destination: The filesystem path to probe (typically args.data_dir
            or _capacity_gate_destination()).
        hosts: List of hostnames or "host:slots" strings. None or
            len(...) <= 1 triggers the silent no-op.
        run_uuid: The per-Benchmark-instance UUID (from self._run_uuid).
            Passed verbatim to the subprocess as argv[2].
        logger: Project logger.
        mpi_bin: MPI binary to use (MPIRUN or MPIEXEC). Defaults to MPIRUN.
        allow_run_as_root: If True, pass --allow-run-as-root to mpirun.
        timeout_seconds: Max time to wait for the probe.
        ssh_username: Optional SSH username for remote staging.

    Raises:
        FileSystemError: On any per-rank failure, cardinality > 1, missing
            mpi4py on a remote host, or missing probe output file.
    """
    # SC#8 silent no-op for single-host runs. Debug-only log.
    if not hosts or len(hosts) <= 1:
        logger.debug("CAP-02 skipped: single-host run")
        return None

    # Defense-in-depth: destination must be non-empty.
    if not destination:
        logger.debug("CAP-02 skipped: no destination provided")
        return None

    # Resolve mpi_bin default.
    if mpi_bin is None:
        mpi_bin = MPIRUN

    # ---- Stage the probe script: local tempfile + (optional) SCP to remotes.
    local_script_path = _write_probe_script_to_tempfile(SHARED_FS_PROBE_SCRIPT)

    # HARDEN-02 D-54/D-57: stdout-marker transport replaces the launch-host-local
    # result file. Rank 0 prints __CAP02_RESULT_BEGIN__/END framed JSON to
    # stdout; the launcher parses subprocess.run(...).stdout via the marker
    # regex below. No launch-host-local result-file tempfile, no cross-host file dependency.
    import tempfile

    # Unique hosts (strip slot counts like 'host:4' → 'host').
    unique_hosts = []
    seen = set()
    for h in hosts:
        hostname = h.split(':')[0] if ':' in h else h
        if hostname not in seen:
            seen.add(hostname)
            unique_hosts.append(hostname)

    # Stage on every non-localhost host via the existing
    # MPIClusterCollector._stage_script_on_remote_hosts pattern. Construct
    # a throwaway collector solely to reuse the staging helper.
    remote_hosts = [h for h in unique_hosts if not _is_localhost(h)]
    remote_script_path = local_script_path  # default for the single-localhost case
    if remote_hosts:
        # Build a temporary collector for SSH staging. results_dir is
        # required by the collector — we use the system tempdir since the
        # probe staging is per-invocation (NOT persisted as a run artifact;
        # the cluster-info collector keeps its own staged copy).
        staging_results_dir = tempfile.mkdtemp(prefix="mlps_cap02_stage_")
        staging_dir = os.path.join(staging_results_dir, "cap02-probe-staging")
        remote_script_path = os.path.join(staging_dir, "mlps_cap02_probe.py")
        try:
            staging_collector = MPIClusterCollector(
                hosts=unique_hosts,
                mpi_bin=mpi_bin,
                logger=logger,
                results_dir=staging_results_dir,
                allow_run_as_root=allow_run_as_root,
                timeout_seconds=timeout_seconds,
                ssh_username=ssh_username,
            )
            # Use the staging helper directly. The collector's full
            # collect() path is not invoked here — only the SCP helper.
            os.makedirs(staging_dir, exist_ok=True)
            # Copy the local probe script into the staging dir so the
            # remote path matches what we'll point mpirun at.
            shutil.copy2(local_script_path, remote_script_path)
            stage_results = staging_collector._stage_script_on_remote_hosts(
                remote_script_path, staging_dir, remote_hosts,
            )
            failures = {h: e for h, e in stage_results.items() if e}
            if failures:
                msg = (
                    "CAP-02: failed to stage shared-FS probe script on "
                    "{n} host(s): {f}.".format(n=len(failures), f=failures)
                )
                logger.error(msg)
                raise FileSystemError(
                    msg,
                    path=destination,
                    operation="cap02-shared-fs-probe",
                    code=ErrorCode.FS_INVALID_STRUCTURE,
                )
        except FileSystemError:
            raise
        except Exception as e:
            msg = (
                "CAP-02: probe staging error: {0}".format(e)
            )
            logger.error(msg)
            raise FileSystemError(
                msg,
                path=destination,
                operation="cap02-shared-fs-probe",
                code=ErrorCode.FS_INVALID_STRUCTURE,
            )

    # ---- Build the mpirun command.
    if mpi_bin == MPIRUN:
        mpi_executable = MPI_RUN_BIN
    elif mpi_bin == MPIEXEC:
        mpi_executable = MPI_EXEC_BIN
    else:
        mpi_executable = mpi_bin

    n = len(unique_hosts)
    host_slots = ",".join("{0}:1".format(h) for h in unique_hosts)
    cmd_parts = [
        mpi_executable,
        "-n", str(n),
        "-host", host_slots,
        "--bind-to", "none",
        "--map-by", "node",
        # HARDEN-02 D-55.1: --tag-output gives PRRTE per-line atomicity and
        # prefixes each forwarded line with [rank,jobid]<channel>: (OpenMPI 4.x
        # format; verified on 4.1.6 per HARDEN-04 — channel is <stdout>,
        # <stderr>, or <stddiag>, sometimes with a trailing colon). The launcher's
        # marker regex tolerates the prefix; the post-extract _strip_tag_output_prefix()
        # consumes both the bracketed identifier AND the optional <channel>: marker.
        # Must appear BEFORE --allow-run-as-root per OpenMPI's accepted arg ordering.
        "--tag-output",
    ]
    if allow_run_as_root:
        cmd_parts.append("--allow-run-as-root")
    cmd_parts += [
        "python3",
        remote_script_path,
        destination,
        run_uuid,
        # HARDEN-02 D-54: result-file positional REMOVED; rank-0 emits result
        # via stdout markers parsed below.
    ]
    cmd_str = " ".join(cmd_parts)

    logger.debug("CAP-02 probe command: {0}".format(cmd_str))

    # ---- Execute via subprocess (mirrors MPIClusterCollector.collect()).
    env = os.environ.copy()
    env.pop("DISPLAY", None)
    env.pop("XAUTHORITY", None)
    env.setdefault(
        "PLM_RSH_AGENT",
        "ssh -o ForwardX11=no -o ForwardX11Trusted=no "
        "-o StrictHostKeyChecking=accept-new",
    )

    try:
        result = subprocess.run(
            cmd_str,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired:
        msg = (
            "CAP-02: shared-FS probe timed out after {0}s".format(timeout_seconds)
        )
        logger.error(msg)
        raise FileSystemError(
            msg,
            path=destination,
            operation="cap02-shared-fs-probe",
            code=ErrorCode.FS_INVALID_STRUCTURE,
        )

    # ---- Parse the rank-0 JSON output from stdout markers (HARDEN-02 D-54/D-55).
    # The probe heredoc emits three lines on rank 0:
    #     __CAP02_RESULT_BEGIN__
    #     <compact single-line JSON payload>
    #     __CAP02_RESULT_END__
    # --tag-output prefixes each line with [rank,jobid]<channel>: (OpenMPI 4.x
    # format; channel is <stdout>/<stderr>/<stddiag>, optionally with trailing
    # colon). The non-greedy marker regex tolerates the prefix on the marker
    # lines (the prefix becomes part of the .*? non-greedy match), and the
    # post-extract _strip_tag_output_prefix() consumes both the [rank,jobid]
    # bracketed identifier AND the optional <channel>: marker from the
    # payload line. HARDEN-04 closes the regression where the old regex
    # `r'^\[[^\]]+\]\s*'` left <stdout>: glued to the JSON.
    _marker_re = re.compile(
        r"__CAP02_RESULT_BEGIN__\s*\n(?P<payload>.*?)\n.*?__CAP02_RESULT_END__",
        re.DOTALL,
    )
    _m = _marker_re.search(result.stdout or "")
    if _m is None:
        # Markers absent → real mpirun failure (mpi4py missing on a remote
        # host, mpirun crashed, etc.). Surface the ACTUAL cause (returncode +
        # stderr tail) — the old code raised a misleading "mpi4py not
        # installed" message even when the probe semantically succeeded.
        _stderr_tail = (result.stderr or "").strip()
        msg = (
            "CAP-02: shared-FS probe produced no rank-0 result markers in "
            "stdout. mpirun returncode={0}, stderr={1}".format(
                result.returncode, _stderr_tail
            )
        )
        logger.error(msg)
        raise FileSystemError(
            msg,
            path=destination,
            operation="cap02-shared-fs-probe",
            code=ErrorCode.FS_INVALID_STRUCTURE,
        )

    # Strip a single leading [host:rank] tag from --tag-output if present.
    _payload_raw = _m.group("payload").strip()
    _payload = _strip_tag_output_prefix(_payload_raw)

    try:
        probe_output = json.loads(_payload)
    except ValueError as e:
        msg = (
            "CAP-02: shared-FS probe rank-0 payload unreadable: {0}".format(e)
        )
        logger.error(msg)
        raise FileSystemError(
            msg,
            path=destination,
            operation="cap02-shared-fs-probe",
            code=ErrorCode.FS_INVALID_STRUCTURE,
        )

    # mpi4py-import-error case (Pitfall 8 carried forward).
    if probe_output.get("_mpi_import_error"):
        host = probe_output.get("_hostname", "unknown")
        err = probe_output.get("_error_message", "mpi4py not available")
        msg = (
            "CAP-02: shared-FS probe failed on host '{h}': {e}. "
            "Ensure mpi4py is installed on all cluster nodes.".format(h=host, e=err)
        )
        logger.error(msg)
        raise FileSystemError(
            msg,
            path=destination,
            operation="cap02-shared-fs-probe",
            code=ErrorCode.FS_INVALID_STRUCTURE,
        )

    status = probe_output.get("status")
    failure_summary = probe_output.get("failure_summary")
    unlink_warning = probe_output.get("unlink_warning")

    if unlink_warning:
        # D-44 cosmetic: surface the warning but do not raise.
        logger.warning("CAP-02: " + str(unlink_warning))

    if status == "ok":
        return None

    # status == 'fail' — build the user-facing message.
    if failure_summary and failure_summary.get("message"):
        msg = failure_summary["message"]
    else:
        msg = (
            "CAP-02: shared-FS probe failed with no failure summary "
            "(mpirun returncode={0})".format(result.returncode)
        )
    logger.error(msg)
    raise FileSystemError(
        msg,
        path=destination,
        operation="cap02-shared-fs-probe",
        code=ErrorCode.FS_INVALID_STRUCTURE,
    )


def collect_cluster_info(
    hosts: List[str],
    mpi_bin: str,
    logger,
    results_dir: str,
    allow_run_as_root: bool = False,
    timeout_seconds: int = 60,
    fallback_to_local: bool = True,
    shared_staging_dir: Optional[str] = None,
    shared_tmp_dir: Optional[str] = None,  # deprecated, see note below
    ssh_username: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High-level function to collect cluster information.

    This is the main entry point for collecting cluster information.
    It attempts MPI collection first and falls back to local collection
    if MPI fails.

    Args:
        hosts: List of hostnames/IPs to collect from.
        mpi_bin: MPI command to use.
        logger: Logger instance.
        results_dir: Benchmark results directory. The helper script will be
            staged under ``<results_dir>/collector-staging/`` and persists
            after the run as a debuggable artifact. Required.
        allow_run_as_root: Whether to allow running as root.
        timeout_seconds: Timeout for MPI collection.
        fallback_to_local: If True, fall back to local collection on failure.
        shared_staging_dir: Optional path visible on every node. If provided,
            the collector skips SSH-based script staging. See
            :class:`MPIClusterCollector` for details.
        shared_tmp_dir: Deprecated alias for ``shared_staging_dir``.
        ssh_username: Optional SSH username for remote script staging.
            Defaults to the current user.

    Returns:
        Dictionary mapping hostname -> system_info dict.
        Also includes a '_metadata' key with collection metadata.
    """
    collector = MPIClusterCollector(
        hosts=hosts,
        mpi_bin=mpi_bin,
        logger=logger,
        results_dir=results_dir,
        allow_run_as_root=allow_run_as_root,
        timeout_seconds=timeout_seconds,
        shared_staging_dir=shared_staging_dir,
        shared_tmp_dir=shared_tmp_dir,
        ssh_username=ssh_username,
    )

    metadata = {
        'collection_method': 'unknown',
        'requested_hosts': hosts,
        'collection_timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }

    try:
        result = collector.collect()
        metadata['collection_method'] = 'mpi'
        metadata['hosts_collected'] = list(result.keys())
        result['_metadata'] = metadata
        return result

    except Exception as e:
        logger.warning(f"MPI collection failed: {e}")

        if fallback_to_local:
            logger.info("Falling back to local-only collection")
            result = collector.collect_local_only()
            metadata['collection_method'] = 'local_fallback'
            metadata['mpi_error'] = str(e)
            metadata['hosts_collected'] = list(result.keys())
            result['_metadata'] = metadata
            return result
        else:
            raise


# =============================================================================
# SSH Collection Script
# =============================================================================

SSH_COLLECTOR_SCRIPT = '''
import json
import socket
import time

def collect():
    result = {"hostname": socket.gethostname(), "errors": {}}

    files = [
        ("/proc/meminfo", "meminfo"),
        ("/proc/cpuinfo", "cpuinfo"),
        ("/proc/diskstats", "diskstats"),
        ("/proc/net/dev", "netdev"),
        ("/proc/version", "version"),
        ("/proc/loadavg", "loadavg"),
        ("/proc/uptime", "uptime"),
        ("/proc/vmstat", "vmstat"),
        ("/proc/mounts", "mounts"),
        ("/proc/cgroups", "cgroups"),
    ]

    for path, key in files:
        try:
            with open(path) as f:
                result[key] = f.read()
        except Exception as e:
            result["errors"][key] = str(e)
            result[key] = ""

    try:
        with open("/etc/os-release") as f:
            result["os_release_raw"] = f.read()
    except Exception as e:
        result["errors"]["os_release"] = str(e)

    result["collection_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(json.dumps(result))

collect()
'''


# =============================================================================
# SSH Cluster Collector Class
# =============================================================================

class SSHClusterCollector(ClusterCollectorInterface):
    """Collects system information from hosts using SSH.

    This collector uses SSH to gather system information from remote hosts.
    For localhost, it uses direct local collection to avoid SSH overhead
    and configuration requirements.

    Attributes:
        hosts: List of hostnames or IP addresses to collect from.
        logger: Logger instance for output.
        ssh_username: Optional SSH username (defaults to current user).
        timeout: Timeout in seconds for SSH connections.
        max_workers: Maximum number of parallel SSH connections.
    """

    def __init__(
        self,
        hosts: List[str],
        logger,
        ssh_username: Optional[str] = None,
        timeout_seconds: int = 60,
        max_workers: int = 10
    ):
        """Initialize the SSH cluster collector.

        Args:
            hosts: List of hostnames/IPs, optionally with slot counts (e.g., "host1:4").
            logger: Logger instance for messages.
            ssh_username: Optional SSH username. If not provided, uses current user.
            timeout_seconds: Maximum time to wait for SSH connections.
            max_workers: Maximum number of parallel SSH connections.
        """
        self.hosts = hosts
        self.logger = logger
        self.ssh_username = ssh_username
        self.timeout = timeout_seconds
        self.max_workers = max_workers

    def _get_unique_hosts(self) -> List[str]:
        """Extract unique hostnames from the hosts list (removing slot counts)."""
        unique = []
        seen = set()
        for host in self.hosts:
            hostname = host.split(':')[0].strip() if ':' in host else host.strip()
            if hostname and hostname not in seen:
                seen.add(hostname)
                unique.append(hostname)
        return unique

    def _build_ssh_command(self, hostname: str, remote_cmd: str) -> List[str]:
        """Build SSH command with proper options for automation."""
        cmd = [
            'ssh',
            '-o', 'BatchMode=yes',
            '-o', f'ConnectTimeout={self.timeout}',
            '-o', 'StrictHostKeyChecking=accept-new',
        ]
        if self.ssh_username:
            cmd.extend(['-l', self.ssh_username])
        cmd.extend([hostname, remote_cmd])
        return cmd

    def _parse_raw_collection(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse raw /proc file contents into structured data."""
        parsed = {
            'hostname': raw_data.get('hostname', 'unknown'),
            'collection_timestamp': raw_data.get('collection_timestamp'),
            'errors': raw_data.get('errors', {}),
        }

        # Parse meminfo
        if raw_data.get('meminfo'):
            parsed['meminfo'] = parse_proc_meminfo(raw_data['meminfo'])
        else:
            parsed['meminfo'] = {}

        # Parse cpuinfo
        if raw_data.get('cpuinfo'):
            parsed['cpuinfo'] = parse_proc_cpuinfo(raw_data['cpuinfo'])
        else:
            parsed['cpuinfo'] = []

        # Parse diskstats
        if raw_data.get('diskstats'):
            disks = parse_proc_diskstats(raw_data['diskstats'])
            parsed['diskstats'] = [d.to_dict() for d in disks]
        else:
            parsed['diskstats'] = []

        # Parse netdev
        if raw_data.get('netdev'):
            interfaces = parse_proc_net_dev(raw_data['netdev'])
            parsed['netdev'] = [n.to_dict() for n in interfaces]
        else:
            parsed['netdev'] = []

        # Parse version
        parsed['version'] = parse_proc_version(raw_data.get('version', ''))

        # Parse loadavg
        if raw_data.get('loadavg'):
            load_1, load_5, load_15, running, total = parse_proc_loadavg(raw_data['loadavg'])
            parsed['loadavg'] = {
                'load_1min': load_1,
                'load_5min': load_5,
                'load_15min': load_15,
                'running_processes': running,
                'total_processes': total
            }
        else:
            parsed['loadavg'] = {}

        # Parse uptime
        parsed['uptime_seconds'] = parse_proc_uptime(raw_data.get('uptime', ''))

        # Parse os_release
        if raw_data.get('os_release_raw'):
            parsed['os_release'] = parse_os_release(raw_data['os_release_raw'])
        else:
            parsed['os_release'] = {}

        # Parse vmstat
        if raw_data.get('vmstat'):
            parsed['vmstat'] = parse_proc_vmstat(raw_data['vmstat'])
        else:
            parsed['vmstat'] = {}

        # Parse mounts
        if raw_data.get('mounts'):
            mounts = parse_proc_mounts(raw_data['mounts'])
            parsed['mounts'] = [m.to_dict() for m in mounts]
        else:
            parsed['mounts'] = []

        # Parse cgroups
        if raw_data.get('cgroups'):
            cgroups = parse_proc_cgroups(raw_data['cgroups'])
            parsed['cgroups'] = [c.to_dict() for c in cgroups]
        else:
            parsed['cgroups'] = []

        if not parsed['errors']:
            del parsed['errors']

        return parsed

    def _collect_from_single_host(self, hostname: str) -> Dict[str, Any]:
        """Collect system information from a single host via SSH."""
        if _is_localhost(hostname):
            self.logger.debug(f'Collecting from {hostname} (localhost) via direct access')
            return collect_local_system_info()

        self.logger.debug(f'Collecting from {hostname} via SSH')

        # Build the remote command to run the collector script
        remote_cmd = f"python3 -c '{SSH_COLLECTOR_SCRIPT}'"
        cmd = self._build_ssh_command(hostname, remote_cmd)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout + 10  # Extra buffer for SSH overhead
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() or f'SSH failed with code {result.returncode}'
                self.logger.warning(f'SSH collection from {hostname} failed: {error_msg}')
                return {'hostname': hostname, 'error': error_msg}

            # Parse the JSON output
            try:
                raw_data = json.loads(result.stdout)
                return self._parse_raw_collection(raw_data)
            except json.JSONDecodeError as e:
                self.logger.warning(f'Failed to parse JSON from {hostname}: {e}')
                return {'hostname': hostname, 'error': f'JSON parse error: {e}'}

        except subprocess.TimeoutExpired:
            self.logger.warning(f'SSH to {hostname} timed out after {self.timeout}s')
            return {'hostname': hostname, 'error': f'Timeout after {self.timeout}s'}

        except Exception as e:
            self.logger.warning(f'SSH collection from {hostname} failed: {e}')
            return {'hostname': hostname, 'error': str(e)}

    def collect(self, hosts: List[str], timeout: int = 60) -> CollectionResult:
        """Collect information from all specified hosts in parallel.

        Args:
            hosts: List of hostnames or IP addresses to collect from.
                   Note: This parameter is ignored; uses self.hosts instead.
            timeout: Maximum time in seconds to wait for collection.
                   Note: This parameter is ignored; uses self.timeout instead.

        Returns:
            CollectionResult with data from all hosts.
        """
        unique_hosts = self._get_unique_hosts()
        self.logger.debug(f'Starting SSH cluster collection on {len(unique_hosts)} hosts')

        results = {}
        errors = []

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(unique_hosts))) as executor:
            future_to_host = {
                executor.submit(self._collect_from_single_host, host): host
                for host in unique_hosts
            }

            for future in as_completed(future_to_host):
                host = future_to_host[future]
                try:
                    host_data = future.result()
                    if 'error' in host_data and len(host_data) <= 2:
                        # Collection failed for this host
                        errors.append(f"{host}: {host_data.get('error', 'Unknown error')}")
                    results[host] = host_data
                except Exception as e:
                    self.logger.warning(f'Exception collecting from {host}: {e}')
                    errors.append(f"{host}: {str(e)}")
                    results[host] = {'hostname': host, 'error': str(e)}

        success = len(errors) == 0 or len(results) > len(errors)

        return CollectionResult(
            success=success,
            data=results,
            errors=errors,
            collection_method='ssh',
            timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        )

    def collect_local(self) -> CollectionResult:
        """Collect information from local host only.

        Returns:
            CollectionResult with local host data.
        """
        local_info = collect_local_system_info()
        hostname = local_info.get('hostname', 'localhost')

        return CollectionResult(
            success=True,
            data={hostname: local_info},
            errors=[],
            collection_method='local',
            timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        )

    def is_available(self) -> bool:
        """Check if SSH is available for use.

        Returns:
            True if SSH command is available, False otherwise.
        """
        return shutil.which('ssh') is not None

    def get_collection_method(self) -> str:
        """Return the name of the collection method.

        Returns:
            String identifier 'ssh'.
        """
        return 'ssh'


# =============================================================================
# Time-Series Collection
# =============================================================================

def collect_timeseries_sample() -> Dict[str, Any]:
    """Collect time-varying system metrics for time-series analysis.

    Collects only dynamic metrics that change during benchmark execution:
    - diskstats: I/O statistics per device
    - vmstat: Virtual memory statistics
    - loadavg: System load averages
    - meminfo: Memory usage
    - netdev: Network interface statistics

    Static information (cpuinfo, os_release) is excluded as it doesn't
    change between samples.

    Returns:
        Dictionary containing timestamp, hostname, and metric data.
        Individual metric keys may be missing if collection fails.
    """
    sample = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'hostname': socket.gethostname(),
        'errors': {},
    }

    # Collect /proc/diskstats
    try:
        with open('/proc/diskstats', 'r') as f:
            disks = parse_proc_diskstats(f.read())
            sample['diskstats'] = [d.to_dict() for d in disks]
    except Exception as e:
        sample['errors']['diskstats'] = str(e)

    # Collect /proc/vmstat
    try:
        with open('/proc/vmstat', 'r') as f:
            sample['vmstat'] = parse_proc_vmstat(f.read())
    except Exception as e:
        sample['errors']['vmstat'] = str(e)

    # Collect /proc/loadavg
    try:
        with open('/proc/loadavg', 'r') as f:
            load_1, load_5, load_15, running, total = parse_proc_loadavg(f.read())
            sample['loadavg'] = {
                'load_1min': load_1,
                'load_5min': load_5,
                'load_15min': load_15,
                'running_processes': running,
                'total_processes': total,
            }
    except Exception as e:
        sample['errors']['loadavg'] = str(e)

    # Collect /proc/meminfo
    try:
        with open('/proc/meminfo', 'r') as f:
            sample['meminfo'] = parse_proc_meminfo(f.read())
    except Exception as e:
        sample['errors']['meminfo'] = str(e)

    # Collect /proc/net/dev
    try:
        with open('/proc/net/dev', 'r') as f:
            interfaces = parse_proc_net_dev(f.read())
            sample['netdev'] = [n.to_dict() for n in interfaces]
    except Exception as e:
        sample['errors']['netdev'] = str(e)

    # Remove errors dict if empty
    if not sample['errors']:
        del sample['errors']

    return sample


class TimeSeriesCollector:
    """Collects time-series system metrics in a background thread.

    Uses a non-daemon thread with Event signaling for graceful shutdown.
    Samples are collected at regular intervals and stored in memory.

    Usage:
        collector = TimeSeriesCollector(interval_seconds=10.0)
        collector.start()
        # ... run benchmark ...
        samples = collector.stop()

    Attributes:
        interval_seconds: Time between samples in seconds.
        max_samples: Maximum number of samples to keep (prevents memory issues).
    """

    def __init__(
        self,
        interval_seconds: float = 10.0,
        max_samples: int = 3600,
        logger=None
    ):
        """Initialize the time-series collector.

        Args:
            interval_seconds: Time between samples (default: 10 seconds).
            max_samples: Maximum samples to keep (default: 3600 = 10 hours at 10s).
            logger: Optional logger instance for debug output.
        """
        self.interval_seconds = interval_seconds
        self.max_samples = max_samples
        self.logger = logger

        self._stop_event = threading.Event()
        self._samples: List[Dict[str, Any]] = []
        self._start_time: Optional[str] = None
        self._end_time: Optional[str] = None
        self._thread = threading.Thread(
            target=self._collection_loop,
            daemon=False,  # Non-daemon for graceful shutdown
            name="TimeSeriesCollector"
        )
        self._started = False
        self._stopped = False

    def _collection_loop(self):
        """Run periodic collection until stop signal."""
        while not self._stop_event.is_set():
            try:
                sample = collect_timeseries_sample()

                # Enforce max_samples limit
                if len(self._samples) < self.max_samples:
                    self._samples.append(sample)
                elif self.logger:
                    # Only log once when we hit the limit
                    if len(self._samples) == self.max_samples:
                        self.logger.warning(
                            f'TimeSeriesCollector reached max_samples limit ({self.max_samples}). '
                            f'Further samples will be dropped.'
                        )

            except Exception as e:
                if self.logger:
                    self.logger.debug(f'TimeSeriesCollector sample error: {e}')

            # Use wait(timeout) instead of sleep() for quick response to stop signal
            self._stop_event.wait(timeout=self.interval_seconds)

    def start(self) -> None:
        """Start background collection.

        Raises:
            RuntimeError: If collector was already started or stopped.
        """
        if self._stopped:
            raise RuntimeError('TimeSeriesCollector already stopped; create a new instance')
        if self._started:
            raise RuntimeError('TimeSeriesCollector already started')

        self._start_time = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        self._started = True
        self._thread.start()

        if self.logger:
            self.logger.debug(
                f'TimeSeriesCollector started (interval={self.interval_seconds}s, '
                f'max_samples={self.max_samples})'
            )

    def stop(self) -> List[Dict[str, Any]]:
        """Stop collection and return all samples.

        Returns:
            List of sample dictionaries collected during the run.

        Raises:
            RuntimeError: If collector was not started.
        """
        if not self._started:
            raise RuntimeError('TimeSeriesCollector not started')
        if self._stopped:
            return self._samples

        self._stop_event.set()
        # Wait for thread with timeout slightly longer than interval
        self._thread.join(timeout=self.interval_seconds + 5)

        self._end_time = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        self._stopped = True

        if self.logger:
            self.logger.debug(
                f'TimeSeriesCollector stopped ({len(self._samples)} samples collected)'
            )

        return self._samples

    @property
    def samples(self) -> List[Dict[str, Any]]:
        """Get collected samples (may be incomplete if still running)."""
        return self._samples

    @property
    def start_time(self) -> Optional[str]:
        """Get collection start time (ISO format)."""
        return self._start_time

    @property
    def end_time(self) -> Optional[str]:
        """Get collection end time (ISO format)."""
        return self._end_time

    @property
    def is_running(self) -> bool:
        """Check if collector is currently running."""
        return self._started and not self._stopped


# =============================================================================
# Time-Series SSH Script
# =============================================================================

# Lightweight SSH script for time-series collection (collects only dynamic metrics)
TIMESERIES_SSH_SCRIPT = '''
import json
import socket
import time

def collect():
    result = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "hostname": socket.gethostname(), "errors": {}}

    files = [
        ("/proc/diskstats", "diskstats"),
        ("/proc/vmstat", "vmstat"),
        ("/proc/loadavg", "loadavg"),
        ("/proc/meminfo", "meminfo"),
        ("/proc/net/dev", "netdev"),
    ]

    for path, key in files:
        try:
            with open(path) as f:
                result[key] = f.read()
        except Exception as e:
            result["errors"][key] = str(e)
            result[key] = ""

    if not result["errors"]:
        del result["errors"]
    print(json.dumps(result))

collect()
'''


# =============================================================================
# Multi-Host Time-Series Collector
# =============================================================================

class MultiHostTimeSeriesCollector:
    """Collects time-series metrics from multiple hosts in parallel.

    Uses SSH for remote hosts and direct collection for localhost.
    Collection happens in a background thread with parallel SSH calls
    at each interval using ThreadPoolExecutor.

    Usage:
        collector = MultiHostTimeSeriesCollector(
            hosts=['host1', 'host2', 'localhost'],
            interval_seconds=10.0
        )
        collector.start()
        # ... run benchmark ...
        samples_by_host = collector.stop()

    Attributes:
        hosts: List of hostnames to collect from.
        interval_seconds: Time between collection rounds.
        max_samples: Maximum samples per host to keep.
    """

    def __init__(
        self,
        hosts: List[str],
        interval_seconds: float = 10.0,
        max_samples: int = 3600,
        ssh_username: Optional[str] = None,
        ssh_timeout: int = 30,
        max_workers: int = 10,
        logger=None
    ):
        """Initialize multi-host time-series collector.

        Args:
            hosts: List of hostnames/IPs to collect from.
            interval_seconds: Time between samples (default: 10 seconds).
            max_samples: Maximum samples per host (default: 3600).
            ssh_username: Optional SSH username for remote hosts.
            ssh_timeout: SSH connection timeout in seconds.
            max_workers: Maximum parallel SSH connections.
            logger: Optional logger instance.
        """
        self.hosts = self._get_unique_hosts(hosts)
        self.interval_seconds = interval_seconds
        self.max_samples = max_samples
        self.ssh_username = ssh_username
        self.ssh_timeout = ssh_timeout
        self.max_workers = max_workers
        self.logger = logger

        self._stop_event = threading.Event()
        self._samples_by_host: Dict[str, List[Dict[str, Any]]] = {h: [] for h in self.hosts}
        self._start_time: Optional[str] = None
        self._end_time: Optional[str] = None
        self._thread = threading.Thread(
            target=self._collection_loop,
            daemon=False,
            name="MultiHostTimeSeriesCollector"
        )
        self._started = False
        self._stopped = False

    def _get_unique_hosts(self, hosts: List[str]) -> List[str]:
        """Extract unique hostnames from hosts list (removing slot counts)."""
        unique = []
        seen = set()
        for host in hosts:
            hostname = host.split(':')[0].strip() if ':' in host else host.strip()
            if hostname and hostname not in seen:
                seen.add(hostname)
                unique.append(hostname)
        return unique

    def _build_ssh_command(self, hostname: str, remote_cmd: str) -> List[str]:
        """Build SSH command for remote collection."""
        cmd = [
            'ssh',
            '-o', 'BatchMode=yes',
            '-o', f'ConnectTimeout={self.ssh_timeout}',
            '-o', 'StrictHostKeyChecking=accept-new',
        ]
        if self.ssh_username:
            cmd.extend(['-l', self.ssh_username])
        cmd.extend([hostname, remote_cmd])
        return cmd

    def _parse_remote_sample(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse raw /proc file contents from SSH collection into structured data."""
        sample = {
            'timestamp': raw_data.get('timestamp', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())),
            'hostname': raw_data.get('hostname', 'unknown'),
            'errors': raw_data.get('errors', {}),
        }

        # Parse diskstats
        if raw_data.get('diskstats'):
            disks = parse_proc_diskstats(raw_data['diskstats'])
            sample['diskstats'] = [d.to_dict() for d in disks]

        # Parse vmstat
        if raw_data.get('vmstat'):
            sample['vmstat'] = parse_proc_vmstat(raw_data['vmstat'])

        # Parse loadavg
        if raw_data.get('loadavg'):
            load_1, load_5, load_15, running, total = parse_proc_loadavg(raw_data['loadavg'])
            sample['loadavg'] = {
                'load_1min': load_1,
                'load_5min': load_5,
                'load_15min': load_15,
                'running_processes': running,
                'total_processes': total,
            }

        # Parse meminfo
        if raw_data.get('meminfo'):
            sample['meminfo'] = parse_proc_meminfo(raw_data['meminfo'])

        # Parse netdev
        if raw_data.get('netdev'):
            interfaces = parse_proc_net_dev(raw_data['netdev'])
            sample['netdev'] = [n.to_dict() for n in interfaces]

        if not sample['errors']:
            del sample['errors']

        return sample

    def _collect_from_host(self, hostname: str) -> Dict[str, Any]:
        """Collect single sample from a host (local or remote)."""
        if _is_localhost(hostname):
            return collect_timeseries_sample()

        # Remote collection via SSH
        remote_cmd = f"python3 -c '{TIMESERIES_SSH_SCRIPT}'"
        cmd = self._build_ssh_command(hostname, remote_cmd)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.ssh_timeout + 10
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() or f'SSH failed with code {result.returncode}'
                return {
                    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'hostname': hostname,
                    'errors': {'ssh': error_msg}
                }

            raw_data = json.loads(result.stdout)
            return self._parse_remote_sample(raw_data)

        except subprocess.TimeoutExpired:
            return {
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'hostname': hostname,
                'errors': {'ssh': f'Timeout after {self.ssh_timeout}s'}
            }
        except json.JSONDecodeError as e:
            return {
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'hostname': hostname,
                'errors': {'json': str(e)}
            }
        except Exception as e:
            return {
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'hostname': hostname,
                'errors': {'collection': str(e)}
            }

    def _collect_all_hosts(self) -> None:
        """Collect from all hosts in parallel."""
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(self.hosts))) as executor:
            futures = {
                executor.submit(self._collect_from_host, host): host
                for host in self.hosts
            }

            for future in as_completed(futures, timeout=self.interval_seconds):
                host = futures[future]
                try:
                    sample = future.result(timeout=self.interval_seconds / 2)

                    # Enforce max_samples per host
                    if len(self._samples_by_host[host]) < self.max_samples:
                        self._samples_by_host[host].append(sample)

                except Exception as e:
                    # Log but continue - don't fail collection for one host
                    if self.logger:
                        self.logger.debug(f'Time-series collection from {host} failed: {e}')
                    # Add error sample
                    if len(self._samples_by_host[host]) < self.max_samples:
                        self._samples_by_host[host].append({
                            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                            'hostname': host,
                            'errors': {'collection': str(e)}
                        })

    def _collection_loop(self) -> None:
        """Run periodic collection until stop signal."""
        while not self._stop_event.is_set():
            try:
                self._collect_all_hosts()
            except Exception as e:
                if self.logger:
                    self.logger.debug(f'MultiHostTimeSeriesCollector collection error: {e}')

            self._stop_event.wait(timeout=self.interval_seconds)

    def start(self) -> None:
        """Start background collection.

        Raises:
            RuntimeError: If collector already started or stopped.
        """
        if self._started:
            raise RuntimeError('MultiHostTimeSeriesCollector already started')
        if self._stopped:
            raise RuntimeError('MultiHostTimeSeriesCollector already stopped; create new instance')

        self._start_time = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        self._started = True
        self._thread.start()

        if self.logger:
            self.logger.debug(
                f'MultiHostTimeSeriesCollector started ({len(self.hosts)} hosts, '
                f'interval={self.interval_seconds}s)'
            )

    def stop(self) -> Dict[str, List[Dict[str, Any]]]:
        """Stop collection and return samples organized by host.

        Returns:
            Dictionary mapping hostname -> list of samples.

        Raises:
            RuntimeError: If collector not started.
        """
        if not self._started:
            raise RuntimeError('MultiHostTimeSeriesCollector not started')
        if self._stopped:
            return self._samples_by_host

        self._stop_event.set()
        self._thread.join(timeout=self.interval_seconds + 10)

        self._end_time = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        self._stopped = True

        total_samples = sum(len(samples) for samples in self._samples_by_host.values())
        if self.logger:
            self.logger.debug(
                f'MultiHostTimeSeriesCollector stopped ({total_samples} total samples '
                f'from {len(self.hosts)} hosts)'
            )

        return self._samples_by_host

    @property
    def samples_by_host(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get collected samples organized by host."""
        return self._samples_by_host

    @property
    def start_time(self) -> Optional[str]:
        """Get collection start time (ISO format)."""
        return self._start_time

    @property
    def end_time(self) -> Optional[str]:
        """Get collection end time (ISO format)."""
        return self._end_time

    @property
    def is_running(self) -> bool:
        """Check if collector is currently running."""
        return self._started and not self._stopped

    def get_hosts_with_data(self) -> List[str]:
        """Get list of hosts that have at least one sample."""
        return [host for host, samples in self._samples_by_host.items() if samples]
