#!/usr/bin/env python3
"""
Tests for mlpstorage.cluster_collector module.

This module tests the /proc file parsers, data classes, local collection
functions, and MPI cluster collector.

Run with:
    pytest mlpstorage/tests/test_cluster_collector.py -v
"""

import json
import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch, mock_open

from mlpstorage.cluster_collector import (
    # Data classes
    HostDiskInfo,
    HostNetworkInfo,
    HostSystemInfo,
    # Parsers
    parse_proc_meminfo,
    parse_proc_cpuinfo,
    parse_proc_diskstats,
    parse_proc_net_dev,
    parse_proc_version,
    parse_proc_loadavg,
    parse_proc_uptime,
    parse_os_release,
    # Collection functions
    collect_local_system_info,
    summarize_cpuinfo,
    # MPI collector
    MPIClusterCollector,
    collect_cluster_info,
    MPI_COLLECTOR_SCRIPT,
)


class MockLogger:
    """Mock logger for testing."""
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


@pytest.fixture
def mock_logger():
    """Return a mock logger."""
    return MockLogger()


# =============================================================================
# Tests for Data Classes
# =============================================================================

class TestHostDiskInfo:
    """Tests for HostDiskInfo data class."""

    def test_create_host_disk_info(self):
        """Should create a HostDiskInfo with required fields."""
        disk = HostDiskInfo(device_name="sda")
        assert disk.device_name == "sda"
        assert disk.reads_completed == 0

    def test_to_dict_excludes_none(self):
        """to_dict should exclude None values."""
        disk = HostDiskInfo(
            device_name="sda",
            reads_completed=100,
            discards_completed=None
        )
        d = disk.to_dict()
        assert "device_name" in d
        assert "reads_completed" in d
        assert "discards_completed" not in d

    def test_from_dict(self):
        """from_dict should create instance from dictionary."""
        data = {
            "device_name": "nvme0n1",
            "reads_completed": 1000,
            "writes_completed": 500,
            "extra_field": "ignored"
        }
        disk = HostDiskInfo.from_dict(data)
        assert disk.device_name == "nvme0n1"
        assert disk.reads_completed == 1000
        assert disk.writes_completed == 500


class TestHostNetworkInfo:
    """Tests for HostNetworkInfo data class."""

    def test_create_host_network_info(self):
        """Should create a HostNetworkInfo with required fields."""
        net = HostNetworkInfo(interface_name="eth0")
        assert net.interface_name == "eth0"
        assert net.rx_bytes == 0
        assert net.tx_bytes == 0

    def test_to_dict(self):
        """to_dict should return all fields."""
        net = HostNetworkInfo(
            interface_name="eth0",
            rx_bytes=1024,
            tx_bytes=2048
        )
        d = net.to_dict()
        assert d["interface_name"] == "eth0"
        assert d["rx_bytes"] == 1024
        assert d["tx_bytes"] == 2048

    def test_from_dict(self):
        """from_dict should create instance from dictionary."""
        data = {
            "interface_name": "ens192",
            "rx_bytes": 1000000,
            "tx_bytes": 500000,
            "rx_packets": 10000,
            "tx_packets": 5000
        }
        net = HostNetworkInfo.from_dict(data)
        assert net.interface_name == "ens192"
        assert net.rx_bytes == 1000000


class TestHostSystemInfo:
    """Tests for HostSystemInfo data class."""

    def test_create_host_system_info(self):
        """Should create a HostSystemInfo with hostname."""
        sys_info = HostSystemInfo(hostname="node1")
        assert sys_info.hostname == "node1"
        assert sys_info.kernel_version == ""

    def test_to_dict(self):
        """to_dict should return all fields."""
        sys_info = HostSystemInfo(
            hostname="node1",
            kernel_version="5.4.0-42-generic",
            uptime_seconds=86400.0
        )
        d = sys_info.to_dict()
        assert d["hostname"] == "node1"
        assert d["kernel_version"] == "5.4.0-42-generic"
        assert d["uptime_seconds"] == 86400.0


# =============================================================================
# Tests for /proc File Parsers
# =============================================================================

class TestParseProcMeminfo:
    """Tests for parse_proc_meminfo function."""

    def test_parse_basic_meminfo(self):
        """Should parse basic meminfo content."""
        content = """MemTotal:       16384000 kB
MemFree:         8192000 kB
MemAvailable:   10240000 kB
Buffers:          512000 kB
Cached:          2048000 kB
"""
        result = parse_proc_meminfo(content)
        assert result["MemTotal"] == 16384000
        assert result["MemFree"] == 8192000
        assert result["MemAvailable"] == 10240000
        assert result["Buffers"] == 512000
        assert result["Cached"] == 2048000

    def test_parse_meminfo_with_whitespace(self):
        """Should handle varying whitespace."""
        content = "MemTotal:        32000000 kB\nSwapTotal:             0 kB\n"
        result = parse_proc_meminfo(content)
        assert result["MemTotal"] == 32000000
        assert result["SwapTotal"] == 0

    def test_parse_empty_meminfo(self):
        """Should return empty dict for empty input."""
        result = parse_proc_meminfo("")
        assert result == {}

    def test_parse_meminfo_ignores_invalid_lines(self):
        """Should skip lines without proper format."""
        content = """MemTotal:       16384000 kB
invalid line without colon
MemFree:         8192000 kB
"""
        result = parse_proc_meminfo(content)
        assert "MemTotal" in result
        assert "MemFree" in result
        assert len(result) == 2


class TestParseProcCpuinfo:
    """Tests for parse_proc_cpuinfo function."""

    def test_parse_single_cpu(self):
        """Should parse single CPU info."""
        content = """processor	: 0
vendor_id	: GenuineIntel
model name	: Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz
cpu MHz		: 2400.000
cpu cores	: 14
"""
        result = parse_proc_cpuinfo(content)
        assert len(result) == 1
        assert result[0]["processor"] == 0
        assert result[0]["vendor_id"] == "GenuineIntel"
        assert "Intel" in result[0]["model name"]
        assert result[0]["cpu MHz"] == 2400.0
        assert result[0]["cpu cores"] == 14

    def test_parse_multiple_cpus(self):
        """Should parse multiple CPUs separated by blank lines."""
        content = """processor	: 0
model name	: Intel Xeon

processor	: 1
model name	: Intel Xeon

processor	: 2
model name	: Intel Xeon
"""
        result = parse_proc_cpuinfo(content)
        assert len(result) == 3
        assert result[0]["processor"] == 0
        assert result[1]["processor"] == 1
        assert result[2]["processor"] == 2

    def test_parse_empty_cpuinfo(self):
        """Should return empty list for empty input."""
        result = parse_proc_cpuinfo("")
        assert result == []

    def test_parse_cpuinfo_numeric_conversion(self):
        """Should convert numeric values correctly."""
        content = """processor	: 5
cpu MHz		: 3200.50
cache size	: 12288 KB
"""
        result = parse_proc_cpuinfo(content)
        assert result[0]["processor"] == 5
        assert result[0]["cpu MHz"] == 3200.5


class TestParseProcDiskstats:
    """Tests for parse_proc_diskstats function."""

    def test_parse_basic_diskstats(self):
        """Should parse basic diskstats content."""
        content = """   8       0 sda 100 50 2000 1000 200 100 4000 2000 0 1500 3000
   8       1 sda1 80 40 1600 800 160 80 3200 1600 0 1200 2400
"""
        result = parse_proc_diskstats(content)
        assert len(result) == 2
        assert result[0].device_name == "sda"
        assert result[0].reads_completed == 100
        assert result[0].reads_merged == 50
        assert result[0].sectors_read == 2000
        assert result[0].writes_completed == 200
        assert result[0].sectors_written == 4000

    def test_parse_diskstats_with_discard_fields(self):
        """Should parse newer diskstats with discard fields."""
        content = """   8       0 sda 100 50 2000 1000 200 100 4000 2000 0 1500 3000 10 5 500 250
"""
        result = parse_proc_diskstats(content)
        assert len(result) == 1
        assert result[0].discards_completed == 10
        assert result[0].discards_merged == 5
        assert result[0].sectors_discarded == 500

    def test_parse_diskstats_with_flush_fields(self):
        """Should parse diskstats with flush fields."""
        content = """   8       0 sda 100 50 2000 1000 200 100 4000 2000 0 1500 3000 10 5 500 250 20 100
"""
        result = parse_proc_diskstats(content)
        assert result[0].flush_requests_completed == 20
        assert result[0].time_flushing_ms == 100

    def test_parse_empty_diskstats(self):
        """Should return empty list for empty input."""
        result = parse_proc_diskstats("")
        assert result == []


class TestParseProcNetDev:
    """Tests for parse_proc_net_dev function."""

    def test_parse_basic_net_dev(self):
        """Should parse basic /proc/net/dev content."""
        content = """Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
    lo: 1000000   10000    0    0    0     0          0         0  1000000   10000    0    0    0     0       0          0
  eth0: 5000000   50000    5   10    0     0          0       100  2500000   25000    2    5    0     0       0          0
"""
        result = parse_proc_net_dev(content)
        assert len(result) == 2

        lo = next(n for n in result if n.interface_name == "lo")
        assert lo.rx_bytes == 1000000
        assert lo.rx_packets == 10000
        assert lo.tx_bytes == 1000000

        eth0 = next(n for n in result if n.interface_name == "eth0")
        assert eth0.rx_bytes == 5000000
        assert eth0.rx_errors == 5
        assert eth0.tx_bytes == 2500000

    def test_parse_empty_net_dev(self):
        """Should return empty list for empty input."""
        result = parse_proc_net_dev("")
        assert result == []


class TestParseProcVersion:
    """Tests for parse_proc_version function."""

    def test_parse_version(self):
        """Should return the full version string."""
        content = "Linux version 5.4.0-42-generic (buildd@lgw01-amd64-038) (gcc version 9.3.0)\n"
        result = parse_proc_version(content)
        assert "Linux version 5.4.0-42-generic" in result

    def test_parse_empty_version(self):
        """Should return empty string for empty input."""
        result = parse_proc_version("")
        assert result == ""


class TestParseProcLoadavg:
    """Tests for parse_proc_loadavg function."""

    def test_parse_loadavg(self):
        """Should parse load averages and process counts."""
        content = "0.50 0.75 0.80 2/500 12345\n"
        load_1, load_5, load_15, running, total = parse_proc_loadavg(content)
        assert load_1 == 0.5
        assert load_5 == 0.75
        assert load_15 == 0.8
        assert running == 2
        assert total == 500

    def test_parse_empty_loadavg(self):
        """Should return zeros for empty input."""
        result = parse_proc_loadavg("")
        assert result == (0.0, 0.0, 0.0, 0, 0)

    def test_parse_loadavg_high_values(self):
        """Should handle high load values."""
        content = "150.25 100.50 80.75 100/2000 99999\n"
        load_1, load_5, load_15, running, total = parse_proc_loadavg(content)
        assert load_1 == 150.25
        assert load_5 == 100.50
        assert running == 100
        assert total == 2000


class TestParseProcUptime:
    """Tests for parse_proc_uptime function."""

    def test_parse_uptime(self):
        """Should parse uptime in seconds."""
        content = "12345.67 98765.43\n"
        result = parse_proc_uptime(content)
        assert result == 12345.67

    def test_parse_empty_uptime(self):
        """Should return 0 for empty input."""
        result = parse_proc_uptime("")
        assert result == 0.0


class TestParseOsRelease:
    """Tests for parse_os_release function."""

    def test_parse_os_release(self):
        """Should parse /etc/os-release content."""
        content = """NAME="Ubuntu"
VERSION="20.04.3 LTS (Focal Fossa)"
ID=ubuntu
VERSION_ID="20.04"
PRETTY_NAME="Ubuntu 20.04.3 LTS"
"""
        result = parse_os_release(content)
        assert result["NAME"] == "Ubuntu"
        assert result["VERSION"] == "20.04.3 LTS (Focal Fossa)"
        assert result["ID"] == "ubuntu"
        assert result["VERSION_ID"] == "20.04"

    def test_parse_os_release_single_quotes(self):
        """Should handle single-quoted values."""
        content = "NAME='CentOS Linux'\nVERSION='7 (Core)'\n"
        result = parse_os_release(content)
        assert result["NAME"] == "CentOS Linux"
        assert result["VERSION"] == "7 (Core)"

    def test_parse_empty_os_release(self):
        """Should return empty dict for empty input."""
        result = parse_os_release("")
        assert result == {}


# =============================================================================
# Tests for CPU Info Summary
# =============================================================================

class TestSummarizeCpuinfo:
    """Tests for summarize_cpuinfo function."""

    def test_summarize_empty_list(self):
        """Should handle empty CPU list."""
        result = summarize_cpuinfo([])
        assert result["num_logical_cores"] == 0
        assert result["num_physical_cores"] == 0

    def test_summarize_single_cpu(self):
        """Should summarize single CPU."""
        cpus = [{"processor": 0, "model name": "Intel Xeon", "physical id": 0, "core id": 0}]
        result = summarize_cpuinfo(cpus)
        assert result["num_logical_cores"] == 1
        assert result["num_physical_cores"] == 1
        assert result["model"] == "Intel Xeon"

    def test_summarize_hyperthreaded_cpus(self):
        """Should correctly count physical cores with hyperthreading."""
        # 2 physical cores with 2 threads each = 4 logical cores
        cpus = [
            {"processor": 0, "physical id": 0, "core id": 0, "model name": "Intel"},
            {"processor": 1, "physical id": 0, "core id": 0, "model name": "Intel"},
            {"processor": 2, "physical id": 0, "core id": 1, "model name": "Intel"},
            {"processor": 3, "physical id": 0, "core id": 1, "model name": "Intel"},
        ]
        result = summarize_cpuinfo(cpus)
        assert result["num_logical_cores"] == 4
        assert result["num_physical_cores"] == 2
        assert result["num_sockets"] == 1

    def test_summarize_multi_socket(self):
        """Should detect multiple sockets."""
        cpus = [
            {"processor": 0, "physical id": 0, "core id": 0, "model name": "Intel"},
            {"processor": 1, "physical id": 1, "core id": 0, "model name": "Intel"},
        ]
        result = summarize_cpuinfo(cpus)
        assert result["num_sockets"] == 2


# =============================================================================
# Tests for Local System Info Collection
# =============================================================================

class TestCollectLocalSystemInfo:
    """Tests for collect_local_system_info function."""

    def test_collect_returns_dict_with_hostname(self):
        """Should return dictionary with hostname."""
        result = collect_local_system_info()
        assert "hostname" in result
        assert isinstance(result["hostname"], str)

    def test_collect_returns_timestamp(self):
        """Should return collection timestamp."""
        result = collect_local_system_info()
        assert "collection_timestamp" in result

    def test_collect_includes_meminfo(self):
        """Should include meminfo data."""
        result = collect_local_system_info()
        assert "meminfo" in result
        # On a real Linux system, this should have MemTotal
        if result["meminfo"]:
            assert "MemTotal" in result["meminfo"]

    def test_collect_includes_cpuinfo(self):
        """Should include cpuinfo data."""
        result = collect_local_system_info()
        assert "cpuinfo" in result
        assert isinstance(result["cpuinfo"], list)

    def test_collect_includes_diskstats(self):
        """Should include diskstats data."""
        result = collect_local_system_info()
        assert "diskstats" in result
        assert isinstance(result["diskstats"], list)

    def test_collect_includes_netdev(self):
        """Should include network device data."""
        result = collect_local_system_info()
        assert "netdev" in result
        assert isinstance(result["netdev"], list)

    def test_collect_includes_loadavg(self):
        """Should include load average data."""
        result = collect_local_system_info()
        assert "loadavg" in result

    def test_collect_handles_missing_files_gracefully(self):
        """Should handle missing /proc files gracefully."""
        # This is tested implicitly - if the system doesn't have
        # certain files, the function should still return valid data
        result = collect_local_system_info()
        assert isinstance(result, dict)


# =============================================================================
# Tests for MPI Cluster Collector
# =============================================================================

class TestMPIClusterCollector:
    """Tests for MPIClusterCollector class."""

    def test_init(self, mock_logger):
        """Should initialize with correct attributes."""
        collector = MPIClusterCollector(
            hosts=["host1", "host2"],
            mpi_bin="mpirun",
            logger=mock_logger
        )
        assert collector.hosts == ["host1", "host2"]
        assert collector.mpi_bin == "mpirun"
        assert collector.timeout == 60

    def test_get_unique_hosts(self, mock_logger):
        """Should extract unique hostnames from host list."""
        collector = MPIClusterCollector(
            hosts=["host1:4", "host2:4", "host1:4"],
            mpi_bin="mpirun",
            logger=mock_logger
        )
        unique = collector._get_unique_hosts()
        assert len(unique) == 2
        assert "host1" in unique
        assert "host2" in unique

    def test_generate_mpi_command(self, mock_logger):
        """Should generate valid MPI command."""
        collector = MPIClusterCollector(
            hosts=["host1", "host2"],
            mpi_bin="mpirun",
            logger=mock_logger
        )
        cmd = collector._generate_mpi_command("/tmp/script.py", "/tmp/output.json")
        assert "mpirun" in cmd
        assert "-n 2" in cmd
        assert "host1:1" in cmd
        assert "host2:1" in cmd
        assert "/tmp/script.py" in cmd
        assert "/tmp/output.json" in cmd

    def test_generate_mpi_command_with_root(self, mock_logger):
        """Should include --allow-run-as-root when specified."""
        collector = MPIClusterCollector(
            hosts=["host1"],
            mpi_bin="mpirun",
            logger=mock_logger,
            allow_run_as_root=True
        )
        cmd = collector._generate_mpi_command("/tmp/script.py", "/tmp/output.json")
        assert "--allow-run-as-root" in cmd

    def test_write_collector_script(self, mock_logger):
        """Should write collector script to file."""
        collector = MPIClusterCollector(
            hosts=["host1"],
            mpi_bin="mpirun",
            logger=mock_logger
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "collector.py")
            collector._write_collector_script(script_path)

            assert os.path.exists(script_path)
            with open(script_path, 'r') as f:
                content = f.read()
            assert "def collect_local_info" in content
            assert "MPI" in content

    def test_collect_local_only(self, mock_logger):
        """collect_local_only should return local system info."""
        collector = MPIClusterCollector(
            hosts=["host1"],
            mpi_bin="mpirun",
            logger=mock_logger
        )
        result = collector.collect_local_only()
        assert isinstance(result, dict)
        assert len(result) == 1
        # Should have one entry keyed by hostname
        hostname = list(result.keys())[0]
        assert "meminfo" in result[hostname]


class TestCollectClusterInfo:
    """Tests for collect_cluster_info high-level function."""

    def test_collect_cluster_info_with_fallback(self, mock_logger):
        """Should fall back to local collection when MPI fails."""
        # This test assumes MPI is not available or configured
        result = collect_cluster_info(
            hosts=["localhost"],
            mpi_bin="mpirun",
            logger=mock_logger,
            fallback_to_local=True,
            timeout_seconds=5
        )
        assert "_metadata" in result
        # Should have fallen back due to MPI failure
        assert result["_metadata"]["collection_method"] in ["mpi", "local_fallback"]

    def test_collect_cluster_info_metadata(self, mock_logger):
        """Should include metadata in result."""
        result = collect_cluster_info(
            hosts=["localhost"],
            mpi_bin="mpirun",
            logger=mock_logger,
            fallback_to_local=True,
            timeout_seconds=5
        )
        metadata = result["_metadata"]
        assert "collection_method" in metadata
        assert "requested_hosts" in metadata
        assert "collection_timestamp" in metadata


# =============================================================================
# Tests for MPI Collector Script
# =============================================================================

class TestMPICollectorScript:
    """Tests for the embedded MPI collector script."""

    def test_script_is_valid_python(self):
        """The embedded script should be valid Python."""
        # This will raise SyntaxError if invalid
        compile(MPI_COLLECTOR_SCRIPT, '<string>', 'exec')

    def test_script_contains_required_functions(self):
        """Script should contain all required functions."""
        assert "def parse_proc_meminfo" in MPI_COLLECTOR_SCRIPT
        assert "def parse_proc_cpuinfo" in MPI_COLLECTOR_SCRIPT
        assert "def parse_proc_diskstats" in MPI_COLLECTOR_SCRIPT
        assert "def parse_proc_net_dev" in MPI_COLLECTOR_SCRIPT
        assert "def collect_local_info" in MPI_COLLECTOR_SCRIPT
        assert "def main" in MPI_COLLECTOR_SCRIPT

    def test_script_handles_no_mpi(self):
        """Script should handle case where mpi4py is not available."""
        assert "except ImportError" in MPI_COLLECTOR_SCRIPT


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests that test multiple components together."""

    def test_collect_and_serialize(self):
        """Should be able to collect and serialize to JSON."""
        info = collect_local_system_info()
        # Should be JSON serializable
        json_str = json.dumps(info, indent=2)
        # Should be parseable
        parsed = json.loads(json_str)
        assert parsed["hostname"] == info["hostname"]

    def test_disk_info_roundtrip(self):
        """HostDiskInfo should survive JSON roundtrip."""
        disk = HostDiskInfo(
            device_name="sda",
            reads_completed=100,
            writes_completed=200
        )
        d = disk.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        restored = HostDiskInfo.from_dict(parsed)
        assert restored.device_name == disk.device_name
        assert restored.reads_completed == disk.reads_completed

    def test_network_info_roundtrip(self):
        """HostNetworkInfo should survive JSON roundtrip."""
        net = HostNetworkInfo(
            interface_name="eth0",
            rx_bytes=1000000,
            tx_bytes=500000
        )
        d = net.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        restored = HostNetworkInfo.from_dict(parsed)
        assert restored.interface_name == net.interface_name
        assert restored.rx_bytes == net.rx_bytes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
