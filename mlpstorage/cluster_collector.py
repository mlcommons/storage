"""
MPI-based Cluster Information Collector for MLPerf Storage.

This module provides functionality to collect system information from all nodes
in a distributed cluster using MPI. It collects data from /proc filesystem
including meminfo, cpuinfo, diskstats, and network statistics.
"""

import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from mlpstorage.config import MPIRUN, MPIEXEC, MPI_RUN_BIN, MPI_EXEC_BIN
from mlpstorage.interfaces.collector import ClusterCollectorInterface, CollectionResult


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

import json
import os
import socket
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

    # Collect local info
    local_info = collect_local_info()
    local_info['mpi_rank'] = rank

    # Gather all info to rank 0
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
        allow_run_as_root: bool = False,
        timeout_seconds: int = 60
    ):
        """
        Initialize the MPI cluster collector.

        Args:
            hosts: List of hostnames/IPs, optionally with slot counts (e.g., "host1:4").
            mpi_bin: MPI binary to use (MPIRUN or MPIEXEC constant).
            logger: Logger instance for messages.
            allow_run_as_root: If True, adds --allow-run-as-root flag.
            timeout_seconds: Maximum time to wait for collection.
        """
        self.hosts = hosts
        self.mpi_bin = mpi_bin
        self.logger = logger
        self.allow_run_as_root = allow_run_as_root
        self.timeout = timeout_seconds

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

    def collect(self) -> Dict[str, Any]:
        """
        Execute MPI collection across all nodes.

        Returns:
            Dictionary mapping hostname -> system_info dict.

        Raises:
            RuntimeError: If MPI collection fails completely.
        """
        unique_hosts = self._get_unique_hosts()
        self.logger.debug(f"Starting MPI cluster collection on {len(unique_hosts)} hosts")

        # Create temporary files for script and output
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, 'mlps_collector.py')
            output_path = os.path.join(tmpdir, 'cluster_info.json')

            # Write the collector script
            self._write_collector_script(script_path)

            # Generate and run the MPI command
            cmd = self._generate_mpi_command(script_path, output_path)
            self.logger.debug(f"Running MPI collection command: {cmd}")

            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )

                # Read and parse the output if it exists
                if os.path.exists(output_path):
                    with open(output_path, 'r') as f:
                        collected_data = json.load(f)

                    # Check for MPI import error marker
                    if collected_data.get('_mpi_import_error'):
                        error_msg = collected_data.get('_error_message', 'mpi4py not available')
                        error_host = collected_data.get('_hostname', 'unknown')
                        raise RuntimeError(
                            f"MPI collection failed on host '{error_host}': {error_msg}. "
                            f"Ensure mpi4py is installed on all cluster nodes."
                        )

                    # Check for non-zero return code (other MPI errors)
                    if result.returncode != 0:
                        self.logger.warning(
                            f"MPI collection returned non-zero exit code: {result.returncode}\n"
                            f"stderr: {result.stderr}"
                        )

                    self.logger.debug(
                        f"Successfully collected info from {len(collected_data)} hosts"
                    )
                    return collected_data
                else:
                    raise RuntimeError(
                        f"MPI collection did not produce output file. "
                        f"Return code: {result.returncode}, stderr: {result.stderr}"
                    )

            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"MPI collection timed out after {self.timeout} seconds"
                )
            except Exception as e:
                raise RuntimeError(f"MPI collection failed: {e}")

    def collect_local_only(self) -> Dict[str, Any]:
        """
        Collect system info from local node only (fallback when MPI unavailable).

        Returns:
            Dictionary with single hostname -> system_info mapping.
        """
        local_info = collect_local_system_info()
        return {local_info['hostname']: local_info}


def collect_cluster_info(
    hosts: List[str],
    mpi_bin: str,
    logger,
    allow_run_as_root: bool = False,
    timeout_seconds: int = 60,
    fallback_to_local: bool = True
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
        allow_run_as_root: Whether to allow running as root.
        timeout_seconds: Timeout for MPI collection.
        fallback_to_local: If True, fall back to local collection on failure.

    Returns:
        Dictionary mapping hostname -> system_info dict.
        Also includes a '_metadata' key with collection metadata.
    """
    collector = MPIClusterCollector(
        hosts=hosts,
        mpi_bin=mpi_bin,
        logger=logger,
        allow_run_as_root=allow_run_as_root,
        timeout_seconds=timeout_seconds
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
        if self._started:
            raise RuntimeError('TimeSeriesCollector already started')
        if self._stopped:
            raise RuntimeError('TimeSeriesCollector already stopped; create a new instance')

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
