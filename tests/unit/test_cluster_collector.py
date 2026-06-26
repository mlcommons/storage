"""Unit tests for cluster_collector module."""

import json
import os
import subprocess
import time
import pytest
from unittest.mock import MagicMock, patch, Mock

from mlpstorage_py.cluster_collector import (
    parse_proc_vmstat,
    parse_proc_mounts,
    parse_proc_cgroups,
    MountInfo,
    CgroupInfo,
    SSHClusterCollector,
    _is_localhost,
    collect_local_system_info,
    collect_timeseries_sample,
    TimeSeriesCollector,
    MultiHostTimeSeriesCollector,
    MPI_COLLECTOR_SCRIPT,
    _strip_tag_output_prefix,
)
from mlpstorage_py.interfaces.collector import CollectionResult


class TestParseProcVmstat:
    """Tests for parse_proc_vmstat function."""

    def test_parses_key_value_pairs(self):
        """Test parsing simple key-value pairs."""
        content = """nr_free_pages 12345
nr_zone_inactive_anon 6789
nr_zone_active_anon 1111"""
        result = parse_proc_vmstat(content)
        assert result['nr_free_pages'] == 12345
        assert result['nr_zone_inactive_anon'] == 6789
        assert result['nr_zone_active_anon'] == 1111

    def test_handles_empty_content(self):
        """Test parsing empty content."""
        result = parse_proc_vmstat("")
        assert result == {}

    def test_skips_invalid_lines(self):
        """Test that invalid lines are skipped."""
        content = """nr_free_pages 12345
invalid_line_no_value
nr_active 100"""
        result = parse_proc_vmstat(content)
        assert 'nr_free_pages' in result
        assert 'nr_active' in result
        assert 'invalid_line_no_value' not in result

    def test_skips_non_integer_values(self):
        """Test that non-integer values are skipped."""
        content = """nr_free_pages 12345
some_metric not_a_number
nr_active 100"""
        result = parse_proc_vmstat(content)
        assert result['nr_free_pages'] == 12345
        assert 'some_metric' not in result
        assert result['nr_active'] == 100

    def test_handles_whitespace(self):
        """Test parsing content with extra whitespace."""
        content = """  nr_free_pages   12345
nr_active 100"""
        result = parse_proc_vmstat(content)
        # Leading spaces on line mean we get 3+ parts, so first line is skipped
        # But second line should work
        assert result['nr_active'] == 100

    def test_parses_large_numbers(self):
        """Test parsing large integer values."""
        content = "nr_pages 9999999999999"
        result = parse_proc_vmstat(content)
        assert result['nr_pages'] == 9999999999999

    def test_parses_zero_values(self):
        """Test parsing zero values."""
        content = "nr_zero 0"
        result = parse_proc_vmstat(content)
        assert result['nr_zero'] == 0


class TestParseProcMounts:
    """Tests for parse_proc_mounts function."""

    def test_parses_mount_entries(self):
        """Test parsing standard mount entries."""
        content = """/dev/sda1 / ext4 rw,relatime 0 1
tmpfs /run tmpfs rw,nosuid,nodev 0 0"""
        result = parse_proc_mounts(content)
        assert len(result) == 2
        assert result[0].device == '/dev/sda1'
        assert result[0].mount_point == '/'
        assert result[0].fs_type == 'ext4'
        assert result[0].options == 'rw,relatime'
        assert result[0].dump_freq == 0
        assert result[0].pass_num == 1

    def test_parses_second_mount(self):
        """Test parsing the second mount entry correctly."""
        content = """/dev/sda1 / ext4 rw,relatime 0 1
tmpfs /run tmpfs rw,nosuid,nodev 0 0"""
        result = parse_proc_mounts(content)
        assert result[1].device == 'tmpfs'
        assert result[1].mount_point == '/run'
        assert result[1].fs_type == 'tmpfs'
        assert result[1].options == 'rw,nosuid,nodev'
        assert result[1].dump_freq == 0
        assert result[1].pass_num == 0

    def test_handles_minimal_fields(self):
        """Test parsing with only required 4 fields."""
        content = "/dev/sda1 /mnt ext4 defaults"
        result = parse_proc_mounts(content)
        assert len(result) == 1
        assert result[0].device == '/dev/sda1'
        assert result[0].mount_point == '/mnt'
        assert result[0].fs_type == 'ext4'
        assert result[0].options == 'defaults'
        assert result[0].dump_freq == 0
        assert result[0].pass_num == 0

    def test_handles_empty_content(self):
        """Test parsing empty content."""
        result = parse_proc_mounts("")
        assert result == []

    def test_handles_blank_lines(self):
        """Test parsing content with blank lines."""
        content = """/dev/sda1 / ext4 rw 0 1

/dev/sdb1 /data xfs rw 0 2"""
        result = parse_proc_mounts(content)
        assert len(result) == 2

    def test_mount_info_to_dict(self):
        """Test MountInfo.to_dict method."""
        mount = MountInfo(
            device='/dev/sda1',
            mount_point='/',
            fs_type='ext4',
            options='rw'
        )
        d = mount.to_dict()
        assert d['device'] == '/dev/sda1'
        assert d['mount_point'] == '/'
        assert d['fs_type'] == 'ext4'
        assert d['options'] == 'rw'
        assert d['dump_freq'] == 0
        assert d['pass_num'] == 0

    def test_mount_info_from_dict(self):
        """Test MountInfo.from_dict method."""
        data = {
            'device': '/dev/nvme0n1',
            'mount_point': '/data',
            'fs_type': 'xfs',
            'options': 'rw,noatime',
            'dump_freq': 1,
            'pass_num': 2
        }
        mount = MountInfo.from_dict(data)
        assert mount.device == '/dev/nvme0n1'
        assert mount.mount_point == '/data'
        assert mount.fs_type == 'xfs'
        assert mount.options == 'rw,noatime'
        assert mount.dump_freq == 1
        assert mount.pass_num == 2

    def test_mount_info_from_dict_ignores_extra_keys(self):
        """Test MountInfo.from_dict ignores extra keys."""
        data = {
            'device': '/dev/sda1',
            'mount_point': '/',
            'fs_type': 'ext4',
            'options': 'rw',
            'extra_key': 'ignored'
        }
        mount = MountInfo.from_dict(data)
        assert mount.device == '/dev/sda1'
        assert not hasattr(mount, 'extra_key')

    def test_parses_special_filesystems(self):
        """Test parsing special filesystem types."""
        content = """proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0
sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0
devtmpfs /dev devtmpfs rw,nosuid,size=8139548k,nr_inodes=2034887,mode=755 0 0"""
        result = parse_proc_mounts(content)
        assert len(result) == 3
        assert result[0].fs_type == 'proc'
        assert result[1].fs_type == 'sysfs'
        assert result[2].fs_type == 'devtmpfs'


class TestParseProcCgroups:
    """Tests for parse_proc_cgroups function."""

    def test_parses_cgroup_entries(self):
        """Test parsing cgroup entries."""
        content = """#subsys_name	hierarchy	num_cgroups	enabled
cpu	0	1	1
memory	0	1	1
pids	0	1	0"""
        result = parse_proc_cgroups(content)
        assert len(result) == 3
        assert result[0].subsys_name == 'cpu'
        assert result[0].hierarchy == 0
        assert result[0].num_cgroups == 1
        assert result[0].enabled is True
        assert result[2].subsys_name == 'pids'
        assert result[2].enabled is False

    def test_skips_header_line(self):
        """Test that header line is skipped."""
        content = """#subsys_name	hierarchy	num_cgroups	enabled
cpu	0	1	1"""
        result = parse_proc_cgroups(content)
        assert len(result) == 1
        assert result[0].subsys_name == 'cpu'

    def test_handles_empty_content(self):
        """Test parsing empty content."""
        result = parse_proc_cgroups("")
        assert result == []

    def test_handles_only_header(self):
        """Test parsing with only header line."""
        content = "#subsys_name	hierarchy	num_cgroups	enabled"
        result = parse_proc_cgroups(content)
        assert result == []

    def test_cgroup_info_to_dict(self):
        """Test CgroupInfo.to_dict method."""
        cgroup = CgroupInfo(
            subsys_name='cpu',
            hierarchy=0,
            num_cgroups=1,
            enabled=True
        )
        d = cgroup.to_dict()
        assert d['subsys_name'] == 'cpu'
        assert d['hierarchy'] == 0
        assert d['num_cgroups'] == 1
        assert d['enabled'] is True

    def test_cgroup_info_from_dict(self):
        """Test CgroupInfo.from_dict method."""
        data = {
            'subsys_name': 'memory',
            'hierarchy': 5,
            'num_cgroups': 100,
            'enabled': False
        }
        cgroup = CgroupInfo.from_dict(data)
        assert cgroup.subsys_name == 'memory'
        assert cgroup.hierarchy == 5
        assert cgroup.num_cgroups == 100
        assert cgroup.enabled is False

    def test_cgroup_info_from_dict_ignores_extra_keys(self):
        """Test CgroupInfo.from_dict ignores extra keys."""
        data = {
            'subsys_name': 'cpu',
            'hierarchy': 0,
            'num_cgroups': 1,
            'enabled': True,
            'extra_key': 'ignored'
        }
        cgroup = CgroupInfo.from_dict(data)
        assert cgroup.subsys_name == 'cpu'
        assert not hasattr(cgroup, 'extra_key')

    def test_parses_various_cgroup_subsystems(self):
        """Test parsing various cgroup subsystem names."""
        content = """#subsys_name	hierarchy	num_cgroups	enabled
cpuset	1	1	1
cpu	2	100	1
cpuacct	2	100	1
blkio	3	50	1
memory	4	200	1
devices	5	80	1
freezer	6	1	1
net_cls	7	1	1
perf_event	8	1	1
net_prio	7	1	1
hugetlb	9	1	1
pids	10	150	1
rdma	11	1	1
misc	12	1	0"""
        result = parse_proc_cgroups(content)
        assert len(result) == 14
        subsys_names = [c.subsys_name for c in result]
        assert 'cpuset' in subsys_names
        assert 'memory' in subsys_names
        assert 'blkio' in subsys_names
        assert 'pids' in subsys_names

    def test_parses_disabled_cgroup(self):
        """Test that disabled cgroups are properly identified."""
        content = """#subsys_name	hierarchy	num_cgroups	enabled
misc	0	1	0"""
        result = parse_proc_cgroups(content)
        assert len(result) == 1
        assert result[0].enabled is False

    def test_parses_nonzero_hierarchy(self):
        """Test parsing cgroups with non-zero hierarchy values."""
        content = """#subsys_name	hierarchy	num_cgroups	enabled
memory	4	250	1"""
        result = parse_proc_cgroups(content)
        assert result[0].hierarchy == 4
        assert result[0].num_cgroups == 250


class TestCollectLocalSystemInfo:
    """Tests for collect_local_system_info integration with new parsers."""

    def test_includes_vmstat(self):
        """Test that collect_local_system_info includes vmstat data."""
        from mlpstorage_py.cluster_collector import collect_local_system_info

        info = collect_local_system_info()
        assert 'vmstat' in info
        assert isinstance(info['vmstat'], dict)
        # Should have at least some vmstat entries on a Linux system
        if info['vmstat']:
            # Check for a common vmstat key
            assert any(k.startswith('nr_') for k in info['vmstat'].keys())

    def test_includes_mounts(self):
        """Test that collect_local_system_info includes mounts data."""
        from mlpstorage_py.cluster_collector import collect_local_system_info

        info = collect_local_system_info()
        assert 'mounts' in info
        assert isinstance(info['mounts'], list)
        # Should have at least some mounts on a Linux system
        if info['mounts']:
            mount = info['mounts'][0]
            assert 'device' in mount
            assert 'mount_point' in mount
            assert 'fs_type' in mount
            assert 'options' in mount

    def test_includes_cgroups(self):
        """Test that collect_local_system_info includes cgroups data."""
        from mlpstorage_py.cluster_collector import collect_local_system_info

        info = collect_local_system_info()
        assert 'cgroups' in info
        assert isinstance(info['cgroups'], list)
        # Should have at least some cgroups on a Linux system
        if info['cgroups']:
            cgroup = info['cgroups'][0]
            assert 'subsys_name' in cgroup
            assert 'hierarchy' in cgroup
            assert 'num_cgroups' in cgroup
            assert 'enabled' in cgroup


class TestIsLocalhost:
    """Tests for _is_localhost helper function."""

    def test_localhost_string(self):
        """Test that 'localhost' is detected."""
        assert _is_localhost('localhost') is True

    def test_localhost_ipv4(self):
        """Test that 127.0.0.1 is detected."""
        assert _is_localhost('127.0.0.1') is True

    def test_localhost_ipv6(self):
        """Test that ::1 is detected."""
        assert _is_localhost('::1') is True

    def test_localhost_case_insensitive(self):
        """Test case insensitivity."""
        assert _is_localhost('LOCALHOST') is True
        assert _is_localhost('LocalHost') is True

    def test_remote_host(self):
        """Test that remote host is not localhost."""
        assert _is_localhost('node1.example.com') is False
        assert _is_localhost('192.168.1.100') is False

    @patch('socket.gethostname')
    def test_matches_local_hostname(self, mock_gethostname):
        """Test that local hostname is detected as localhost."""
        mock_gethostname.return_value = 'myhost'
        assert _is_localhost('myhost') is True
        assert _is_localhost('MYHOST') is True

    @patch('socket.gethostname')
    @patch('socket.getfqdn')
    def test_matches_local_fqdn(self, mock_getfqdn, mock_gethostname):
        """Test that local FQDN is detected as localhost."""
        mock_gethostname.return_value = 'myhost'
        mock_getfqdn.return_value = 'myhost.example.com'
        assert _is_localhost('myhost.example.com') is True
        assert _is_localhost('MYHOST.EXAMPLE.COM') is True


class TestSSHClusterCollector:
    """Tests for SSHClusterCollector class."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    @pytest.fixture
    def collector(self, mock_logger):
        """Create a collector instance."""
        return SSHClusterCollector(
            hosts=['node1', 'node2:4', 'localhost'],
            logger=mock_logger
        )

    def test_get_unique_hosts(self, collector):
        """Test that unique hosts are extracted correctly."""
        unique = collector._get_unique_hosts()
        assert unique == ['node1', 'node2', 'localhost']

    def test_get_unique_hosts_removes_duplicates(self, mock_logger):
        """Test that duplicate hosts are removed."""
        collector = SSHClusterCollector(
            hosts=['node1', 'node1:4', 'node2'],
            logger=mock_logger
        )
        unique = collector._get_unique_hosts()
        assert unique == ['node1', 'node2']

    def test_get_unique_hosts_handles_empty_strings(self, mock_logger):
        """Test that empty strings and whitespace are handled."""
        collector = SSHClusterCollector(
            hosts=['node1', '', '  ', 'node2'],
            logger=mock_logger
        )
        unique = collector._get_unique_hosts()
        assert unique == ['node1', 'node2']

    def test_build_ssh_command_basic(self, collector):
        """Test basic SSH command construction."""
        cmd = collector._build_ssh_command('node1', 'echo test')
        assert 'ssh' in cmd
        assert '-o' in cmd
        assert 'BatchMode=yes' in cmd
        assert 'node1' in cmd
        assert 'echo test' in cmd

    def test_build_ssh_command_with_username(self, mock_logger):
        """Test SSH command with username."""
        collector = SSHClusterCollector(
            hosts=['node1'],
            logger=mock_logger,
            ssh_username='testuser'
        )
        cmd = collector._build_ssh_command('node1', 'echo test')
        assert '-l' in cmd
        assert 'testuser' in cmd

    def test_build_ssh_command_has_connect_timeout(self, collector):
        """Test SSH command includes connect timeout."""
        cmd = collector._build_ssh_command('node1', 'echo test')
        # Find the ConnectTimeout option
        connect_timeout_found = False
        for item in cmd:
            if 'ConnectTimeout' in item:
                connect_timeout_found = True
                break
        assert connect_timeout_found

    def test_build_ssh_command_has_strict_host_key(self, collector):
        """Test SSH command includes StrictHostKeyChecking option."""
        cmd = collector._build_ssh_command('node1', 'echo test')
        assert 'StrictHostKeyChecking=accept-new' in cmd

    def test_is_available_with_ssh(self, collector):
        """Test is_available when SSH exists."""
        with patch('shutil.which', return_value='/usr/bin/ssh'):
            assert collector.is_available() is True

    def test_is_available_without_ssh(self, collector):
        """Test is_available when SSH is missing."""
        with patch('shutil.which', return_value=None):
            assert collector.is_available() is False

    def test_get_collection_method(self, collector):
        """Test get_collection_method returns 'ssh'."""
        assert collector.get_collection_method() == 'ssh'

    def test_collect_local(self, collector):
        """Test collect_local returns local system info."""
        result = collector.collect_local()
        assert isinstance(result, CollectionResult)
        assert result.success is True
        assert result.collection_method == 'local'
        assert len(result.data) == 1

    @patch('mlpstorage_py.cluster_collector.collect_local_system_info')
    def test_collect_from_localhost_uses_direct_collection(self, mock_local, collector):
        """Test that localhost uses direct collection, not SSH."""
        mock_local.return_value = {'hostname': 'localhost', 'meminfo': {}}
        result = collector._collect_from_single_host('localhost')
        mock_local.assert_called_once()
        assert result['hostname'] == 'localhost'

    @patch('mlpstorage_py.cluster_collector.collect_local_system_info')
    def test_collect_from_127_uses_direct_collection(self, mock_local, collector):
        """Test that 127.0.0.1 uses direct collection, not SSH."""
        mock_local.return_value = {'hostname': 'localhost', 'meminfo': {}}
        result = collector._collect_from_single_host('127.0.0.1')
        mock_local.assert_called_once()

    @patch('subprocess.run')
    def test_collect_from_remote_host(self, mock_run, collector):
        """Test collecting from a remote host via SSH."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps({
                'hostname': 'node1',
                'meminfo': 'MemTotal:       16384000 kB\n',
                'cpuinfo': '',
                'diskstats': '',
                'netdev': '',
                'version': 'Linux version 5.4.0',
                'loadavg': '0.1 0.2 0.3 1/100 12345',
                'uptime': '12345.67',
                'vmstat': 'nr_free_pages 12345\n',
                'mounts': '/dev/sda1 / ext4 rw 0 1\n',
                'cgroups': '#subsys_name\thierarchy\tnum_cgroups\tenabled\ncpu\t0\t1\t1\n',
                'os_release_raw': 'NAME="Ubuntu"\n',
                'collection_timestamp': '2026-01-24T12:00:00Z'
            }),
            stderr=''
        )

        result = collector._collect_from_single_host('node1')
        mock_run.assert_called_once()
        assert result['hostname'] == 'node1'
        assert 'meminfo' in result

    @patch('subprocess.run')
    def test_collect_parses_meminfo(self, mock_run, collector):
        """Test that collected meminfo is properly parsed."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps({
                'hostname': 'node1',
                'meminfo': 'MemTotal:       16384000 kB\nMemFree:        8192000 kB\n',
                'cpuinfo': '',
                'diskstats': '',
                'netdev': '',
                'version': '',
                'loadavg': '0.1 0.2 0.3 1/100 12345',
                'uptime': '12345.67',
                'vmstat': '',
                'mounts': '',
                'cgroups': '',
                'os_release_raw': '',
                'collection_timestamp': '2026-01-24T12:00:00Z'
            }),
            stderr=''
        )

        result = collector._collect_from_single_host('node1')
        assert 'meminfo' in result
        assert result['meminfo'].get('MemTotal') == 16384000
        assert result['meminfo'].get('MemFree') == 8192000

    @patch('subprocess.run')
    def test_collect_handles_ssh_failure(self, mock_run, collector):
        """Test handling of SSH connection failure."""
        mock_run.return_value = Mock(
            returncode=255,
            stdout='',
            stderr='Connection refused'
        )

        result = collector._collect_from_single_host('node1')
        assert 'error' in result
        assert 'Connection refused' in result['error']

    @patch('subprocess.run')
    def test_collect_handles_ssh_timeout(self, mock_run, collector):
        """Test handling of SSH timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired('ssh', 60)

        result = collector._collect_from_single_host('node1')
        assert 'error' in result
        assert 'Timeout' in result['error']

    @patch('subprocess.run')
    def test_collect_handles_json_parse_error(self, mock_run, collector):
        """Test handling of invalid JSON from remote host."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout='not valid json',
            stderr=''
        )

        result = collector._collect_from_single_host('node1')
        assert 'error' in result
        assert 'JSON parse error' in result['error']

    @patch('subprocess.run')
    def test_collect_handles_generic_exception(self, mock_run, collector):
        """Test handling of generic exceptions during SSH."""
        mock_run.side_effect = OSError("Network unreachable")

        result = collector._collect_from_single_host('node1')
        assert 'error' in result
        assert 'Network unreachable' in result['error']

    @patch('mlpstorage_py.cluster_collector.SSHClusterCollector._collect_from_single_host')
    def test_collect_parallel_execution(self, mock_collect_single, mock_logger):
        """Test that collect uses parallel execution."""
        collector = SSHClusterCollector(
            hosts=['node1', 'node2', 'node3'],
            logger=mock_logger,
            max_workers=3
        )
        mock_collect_single.return_value = {'hostname': 'test', 'meminfo': {}}

        result = collector.collect([], 60)

        # Should collect from all 3 hosts
        assert mock_collect_single.call_count == 3
        assert isinstance(result, CollectionResult)
        assert result.collection_method == 'ssh'
        assert len(result.data) == 3

    @patch('mlpstorage_py.cluster_collector.SSHClusterCollector._collect_from_single_host')
    def test_collect_returns_success_when_all_succeed(self, mock_collect_single, mock_logger):
        """Test collect returns success when all hosts succeed."""
        collector = SSHClusterCollector(
            hosts=['node1', 'node2'],
            logger=mock_logger
        )
        mock_collect_single.return_value = {'hostname': 'test', 'meminfo': {}}

        result = collector.collect([], 60)

        assert result.success is True
        assert len(result.errors) == 0

    @patch('mlpstorage_py.cluster_collector.SSHClusterCollector._collect_from_single_host')
    def test_collect_returns_success_with_partial_failure(self, mock_collect_single, mock_logger):
        """Test collect returns success if majority of hosts succeed."""
        collector = SSHClusterCollector(
            hosts=['node1', 'node2', 'node3'],
            logger=mock_logger
        )
        # First call succeeds, second succeeds, third fails
        mock_collect_single.side_effect = [
            {'hostname': 'node1', 'meminfo': {}},
            {'hostname': 'node2', 'meminfo': {}},
            {'hostname': 'node3', 'error': 'Connection refused'},
        ]

        result = collector.collect([], 60)

        # Success because more hosts succeeded than failed
        assert result.success is True
        assert len(result.errors) == 1
        assert len(result.data) == 3

    @patch('mlpstorage_py.cluster_collector.SSHClusterCollector._collect_from_single_host')
    def test_collect_returns_error_list(self, mock_collect_single, mock_logger):
        """Test collect includes errors in result."""
        collector = SSHClusterCollector(
            hosts=['node1'],
            logger=mock_logger
        )
        mock_collect_single.return_value = {'hostname': 'node1', 'error': 'Test error'}

        result = collector.collect([], 60)

        assert len(result.errors) == 1
        assert 'node1' in result.errors[0]
        assert 'Test error' in result.errors[0]

    def test_collect_local_returns_collection_result(self, collector):
        """Test collect_local returns proper CollectionResult."""
        result = collector.collect_local()

        assert isinstance(result, CollectionResult)
        assert result.success is True
        assert result.collection_method == 'local'
        assert result.errors == []
        assert result.timestamp is not None

    def test_collector_init_defaults(self, mock_logger):
        """Test collector initializes with default values."""
        collector = SSHClusterCollector(
            hosts=['node1'],
            logger=mock_logger
        )

        assert collector.hosts == ['node1']
        assert collector.logger == mock_logger
        assert collector.ssh_username is None
        assert collector.timeout == 60
        assert collector.max_workers == 10

    def test_collector_init_custom_values(self, mock_logger):
        """Test collector initializes with custom values."""
        collector = SSHClusterCollector(
            hosts=['node1', 'node2'],
            logger=mock_logger,
            ssh_username='admin',
            timeout_seconds=120,
            max_workers=5
        )

        assert collector.hosts == ['node1', 'node2']
        assert collector.ssh_username == 'admin'
        assert collector.timeout == 120
        assert collector.max_workers == 5


# =============================================================================
# Time-Series Collection Tests
# =============================================================================

class TestCollectTimeseriesSample:
    """Tests for collect_timeseries_sample function."""

    def test_returns_dict_with_required_fields(self):
        """Sample should contain timestamp and hostname."""
        sample = collect_timeseries_sample()

        assert isinstance(sample, dict)
        assert 'timestamp' in sample
        assert 'hostname' in sample
        # Timestamp should be ISO format
        assert 'T' in sample['timestamp']
        assert sample['timestamp'].endswith('Z')

    def test_contains_diskstats(self):
        """Sample should contain diskstats if available."""
        sample = collect_timeseries_sample()

        # On Linux, diskstats should be present
        if 'diskstats' in sample:
            assert isinstance(sample['diskstats'], list)
            if sample['diskstats']:
                # Each disk should have device_name
                assert 'device_name' in sample['diskstats'][0]

    def test_contains_vmstat(self):
        """Sample should contain vmstat if available."""
        sample = collect_timeseries_sample()

        if 'vmstat' in sample:
            assert isinstance(sample['vmstat'], dict)

    def test_contains_loadavg(self):
        """Sample should contain loadavg if available."""
        sample = collect_timeseries_sample()

        if 'loadavg' in sample:
            assert isinstance(sample['loadavg'], dict)
            assert 'load_1min' in sample['loadavg']
            assert 'load_5min' in sample['loadavg']
            assert 'load_15min' in sample['loadavg']

    def test_contains_meminfo(self):
        """Sample should contain meminfo if available."""
        sample = collect_timeseries_sample()

        if 'meminfo' in sample:
            assert isinstance(sample['meminfo'], dict)

    def test_contains_netdev(self):
        """Sample should contain netdev if available."""
        sample = collect_timeseries_sample()

        if 'netdev' in sample:
            assert isinstance(sample['netdev'], list)

    def test_no_errors_key_when_successful(self):
        """Sample should not have errors key if all collections succeed."""
        sample = collect_timeseries_sample()

        # On a normal Linux system, there should be no errors
        # (but we don't assert this as the test env may vary)
        if 'errors' in sample:
            # If errors present, it should be a dict
            assert isinstance(sample['errors'], dict)


class TestTimeSeriesCollector:
    """Tests for TimeSeriesCollector class."""

    def test_init_sets_defaults(self):
        """Collector should initialize with default values."""
        collector = TimeSeriesCollector()

        assert collector.interval_seconds == 10.0
        assert collector.max_samples == 3600
        assert collector.samples == []
        assert collector.start_time is None
        assert collector.end_time is None
        assert not collector.is_running

    def test_init_custom_values(self):
        """Collector should accept custom interval and max_samples."""
        collector = TimeSeriesCollector(interval_seconds=5.0, max_samples=100)

        assert collector.interval_seconds == 5.0
        assert collector.max_samples == 100

    def test_start_sets_running(self):
        """start() should set is_running to True."""
        collector = TimeSeriesCollector(interval_seconds=0.1)

        try:
            collector.start()
            assert collector.is_running
            assert collector.start_time is not None
        finally:
            collector.stop()

    def test_stop_returns_samples(self):
        """stop() should return collected samples."""
        collector = TimeSeriesCollector(interval_seconds=0.1)

        collector.start()
        time.sleep(0.25)  # Allow a couple samples
        samples = collector.stop()

        assert isinstance(samples, list)
        assert not collector.is_running
        assert collector.end_time is not None

    def test_collects_samples_at_interval(self):
        """Collector should gather samples at specified interval."""
        collector = TimeSeriesCollector(interval_seconds=0.1)

        collector.start()
        time.sleep(0.35)  # Should get 3-4 samples
        samples = collector.stop()

        # Should have collected some samples
        assert len(samples) >= 2

    def test_max_samples_limit_enforced(self):
        """Collector should not exceed max_samples."""
        collector = TimeSeriesCollector(interval_seconds=0.05, max_samples=3)

        collector.start()
        time.sleep(0.3)  # Would collect ~6 samples without limit
        samples = collector.stop()

        assert len(samples) <= 3

    def test_start_twice_raises_error(self):
        """Starting collector twice should raise RuntimeError."""
        collector = TimeSeriesCollector(interval_seconds=0.1)

        try:
            collector.start()
            with pytest.raises(RuntimeError, match="already started"):
                collector.start()
        finally:
            collector.stop()

    def test_stop_without_start_raises_error(self):
        """Stopping without starting should raise RuntimeError."""
        collector = TimeSeriesCollector()

        with pytest.raises(RuntimeError, match="not started"):
            collector.stop()

    def test_reuse_after_stop_raises_error(self):
        """Cannot restart a stopped collector."""
        collector = TimeSeriesCollector(interval_seconds=0.1)

        collector.start()
        collector.stop()

        with pytest.raises(RuntimeError, match="already stopped"):
            collector.start()

    def test_samples_contain_expected_fields(self):
        """Collected samples should have timestamp and hostname."""
        collector = TimeSeriesCollector(interval_seconds=0.1)

        collector.start()
        time.sleep(0.15)
        samples = collector.stop()

        if samples:
            sample = samples[0]
            assert 'timestamp' in sample
            assert 'hostname' in sample


class TestTimeSeriesSampleDataclass:
    """Tests for TimeSeriesSample dataclass."""

    def test_create_with_required_fields(self):
        """Can create sample with just timestamp and hostname."""
        from mlpstorage_py.rules.models import TimeSeriesSample

        sample = TimeSeriesSample(
            timestamp='2026-01-24T12:00:00Z',
            hostname='testhost'
        )

        assert sample.timestamp == '2026-01-24T12:00:00Z'
        assert sample.hostname == 'testhost'

    def test_to_dict_excludes_none(self):
        """to_dict should exclude None values."""
        from mlpstorage_py.rules.models import TimeSeriesSample

        sample = TimeSeriesSample(
            timestamp='2026-01-24T12:00:00Z',
            hostname='testhost',
            vmstat={'nr_free_pages': 12345}
        )

        d = sample.to_dict()
        assert 'timestamp' in d
        assert 'hostname' in d
        assert 'vmstat' in d
        assert 'diskstats' not in d  # None value excluded

    def test_from_dict_roundtrip(self):
        """Can roundtrip through to_dict/from_dict."""
        from mlpstorage_py.rules.models import TimeSeriesSample

        original = TimeSeriesSample(
            timestamp='2026-01-24T12:00:00Z',
            hostname='testhost',
            vmstat={'nr_free_pages': 12345},
            loadavg={'load_1min': 0.5, 'load_5min': 0.6, 'load_15min': 0.7}
        )

        d = original.to_dict()
        restored = TimeSeriesSample.from_dict(d)

        assert restored.timestamp == original.timestamp
        assert restored.hostname == original.hostname
        assert restored.vmstat == original.vmstat
        assert restored.loadavg == original.loadavg


class TestTimeSeriesDataDataclass:
    """Tests for TimeSeriesData dataclass."""

    def test_create_with_fields(self):
        """Can create TimeSeriesData with all fields."""
        from mlpstorage_py.rules.models import TimeSeriesSample, TimeSeriesData

        sample = TimeSeriesSample(
            timestamp='2026-01-24T12:00:00Z',
            hostname='host1'
        )

        data = TimeSeriesData(
            collection_interval_seconds=10.0,
            start_time='2026-01-24T12:00:00Z',
            end_time='2026-01-24T12:01:00Z',
            num_samples=6,
            samples_by_host={'host1': [sample]},
            collection_method='local',
            hosts_requested=['host1'],
            hosts_collected=['host1']
        )

        assert data.collection_interval_seconds == 10.0
        assert data.num_samples == 6

    def test_to_dict_serializes_samples(self):
        """to_dict should serialize nested samples."""
        from mlpstorage_py.rules.models import TimeSeriesSample, TimeSeriesData

        sample = TimeSeriesSample(
            timestamp='2026-01-24T12:00:00Z',
            hostname='host1',
            vmstat={'key': 123}
        )

        data = TimeSeriesData(
            collection_interval_seconds=10.0,
            start_time='2026-01-24T12:00:00Z',
            end_time='2026-01-24T12:01:00Z',
            num_samples=1,
            samples_by_host={'host1': [sample]},
            collection_method='local',
            hosts_requested=['host1'],
            hosts_collected=['host1']
        )

        d = data.to_dict()
        assert 'samples_by_host' in d
        assert 'host1' in d['samples_by_host']
        assert len(d['samples_by_host']['host1']) == 1
        assert d['samples_by_host']['host1'][0]['vmstat'] == {'key': 123}

    def test_from_dict_roundtrip(self):
        """Can roundtrip TimeSeriesData through to_dict/from_dict."""
        from mlpstorage_py.rules.models import TimeSeriesSample, TimeSeriesData

        sample = TimeSeriesSample(
            timestamp='2026-01-24T12:00:00Z',
            hostname='host1'
        )

        original = TimeSeriesData(
            collection_interval_seconds=10.0,
            start_time='2026-01-24T12:00:00Z',
            end_time='2026-01-24T12:01:00Z',
            num_samples=1,
            samples_by_host={'host1': [sample]},
            collection_method='ssh',
            hosts_requested=['host1', 'host2'],
            hosts_collected=['host1']
        )

        d = original.to_dict()
        restored = TimeSeriesData.from_dict(d)

        assert restored.collection_interval_seconds == original.collection_interval_seconds
        assert restored.collection_method == original.collection_method
        assert restored.hosts_requested == original.hosts_requested
        assert len(restored.samples_by_host['host1']) == 1


# =============================================================================
# Multi-Host Time-Series Collection Tests
# =============================================================================

class TestMultiHostTimeSeriesCollector:
    """Tests for MultiHostTimeSeriesCollector class."""

    def test_init_sets_defaults(self):
        """Collector should initialize with default values."""
        collector = MultiHostTimeSeriesCollector(hosts=['localhost'])

        assert collector.interval_seconds == 10.0
        assert collector.max_samples == 3600
        assert 'localhost' in collector.hosts
        assert not collector.is_running

    def test_init_custom_values(self):
        """Collector should accept custom parameters."""
        collector = MultiHostTimeSeriesCollector(
            hosts=['host1', 'host2'],
            interval_seconds=5.0,
            max_samples=100,
            ssh_timeout=15
        )

        assert collector.interval_seconds == 5.0
        assert collector.max_samples == 100
        assert len(collector.hosts) == 2

    def test_deduplicates_hosts(self):
        """Collector should remove duplicate hosts."""
        collector = MultiHostTimeSeriesCollector(
            hosts=['host1', 'host1', 'host2', 'host2:2']
        )

        assert len(collector.hosts) == 2
        assert 'host1' in collector.hosts
        assert 'host2' in collector.hosts

    def test_removes_slot_counts(self):
        """Collector should strip slot counts from hosts."""
        collector = MultiHostTimeSeriesCollector(
            hosts=['host1:4', 'host2:8']
        )

        assert 'host1' in collector.hosts
        assert 'host2' in collector.hosts
        assert 'host1:4' not in collector.hosts

    def test_start_sets_running(self):
        """start() should set is_running to True."""
        collector = MultiHostTimeSeriesCollector(
            hosts=['localhost'],
            interval_seconds=0.1
        )

        try:
            collector.start()
            assert collector.is_running
            assert collector.start_time is not None
        finally:
            collector.stop()

    def test_stop_returns_samples_by_host(self):
        """stop() should return dict organized by host."""
        collector = MultiHostTimeSeriesCollector(
            hosts=['localhost'],
            interval_seconds=0.1
        )

        collector.start()
        time.sleep(0.25)
        samples_by_host = collector.stop()

        assert isinstance(samples_by_host, dict)
        assert not collector.is_running
        assert collector.end_time is not None

    def test_collects_from_localhost(self):
        """Should collect samples from localhost."""
        collector = MultiHostTimeSeriesCollector(
            hosts=['localhost'],
            interval_seconds=0.1
        )

        collector.start()
        time.sleep(0.25)
        samples_by_host = collector.stop()

        # Should have localhost data
        assert 'localhost' in samples_by_host
        assert len(samples_by_host['localhost']) >= 1

    def test_samples_have_expected_structure(self):
        """Collected samples should have timestamp and hostname."""
        collector = MultiHostTimeSeriesCollector(
            hosts=['localhost'],
            interval_seconds=0.1
        )

        collector.start()
        time.sleep(0.15)
        samples_by_host = collector.stop()

        if samples_by_host.get('localhost'):
            sample = samples_by_host['localhost'][0]
            assert 'timestamp' in sample
            assert 'hostname' in sample

    def test_max_samples_per_host_enforced(self):
        """Should not exceed max_samples per host."""
        collector = MultiHostTimeSeriesCollector(
            hosts=['localhost'],
            interval_seconds=0.05,
            max_samples=3
        )

        collector.start()
        time.sleep(0.3)  # Would collect ~6 without limit
        samples_by_host = collector.stop()

        assert len(samples_by_host.get('localhost', [])) <= 3

    def test_start_twice_raises_error(self):
        """Starting twice should raise RuntimeError."""
        collector = MultiHostTimeSeriesCollector(
            hosts=['localhost'],
            interval_seconds=0.1
        )

        try:
            collector.start()
            with pytest.raises(RuntimeError, match="already started"):
                collector.start()
        finally:
            collector.stop()

    def test_stop_without_start_raises_error(self):
        """Stopping without starting should raise RuntimeError."""
        collector = MultiHostTimeSeriesCollector(hosts=['localhost'])

        with pytest.raises(RuntimeError, match="not started"):
            collector.stop()

    def test_get_hosts_with_data(self):
        """get_hosts_with_data should return hosts that have samples."""
        collector = MultiHostTimeSeriesCollector(
            hosts=['localhost'],
            interval_seconds=0.1
        )

        collector.start()
        time.sleep(0.15)
        collector.stop()

        hosts_with_data = collector.get_hosts_with_data()
        assert 'localhost' in hosts_with_data

    def test_handles_unreachable_host_gracefully(self):
        """Should continue collecting even if one host fails."""
        collector = MultiHostTimeSeriesCollector(
            hosts=['localhost', 'nonexistent-host-12345.invalid'],
            interval_seconds=0.2,
            ssh_timeout=1  # Short timeout for test
        )

        collector.start()
        time.sleep(0.5)
        samples_by_host = collector.stop()

        # localhost should still have data
        assert len(samples_by_host.get('localhost', [])) >= 1

        # Unreachable host should have error samples
        bad_host_samples = samples_by_host.get('nonexistent-host-12345.invalid', [])
        if bad_host_samples:
            # If we got samples, they should have errors
            assert any('errors' in s for s in bad_host_samples)


class TestMPICollectorScriptMain:
    """Tests for the main() function embedded in MPI_COLLECTOR_SCRIPT.

    Verifies that every rank always calls comm.gather() even when
    collect_local_info() raises, preventing a deadlock on surviving ranks.
    """

    @staticmethod
    def _load_script_ns():
        """Exec MPI_COLLECTOR_SCRIPT into a fresh namespace and return it."""
        from mlpstorage_py.cluster_collector import MPI_COLLECTOR_SCRIPT
        ns = {'__name__': 'mlps_collector'}
        exec(MPI_COLLECTOR_SCRIPT, ns)
        return ns

    @staticmethod
    def _mock_mpi(mock_comm):
        """Return a sys.modules patch dict wiring mock_comm as MPI.COMM_WORLD."""
        mock_mpi = MagicMock()
        mock_mpi.COMM_WORLD = mock_comm
        mock_mpi4py = MagicMock()
        mock_mpi4py.MPI = mock_mpi
        return {'mpi4py': mock_mpi4py}

    def test_gather_called_on_successful_collection(self, tmp_path):
        """Normal path: gather is called with local info dict including mpi_rank."""
        output_file = str(tmp_path / 'out.json')
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 1
        mock_comm.Get_size.return_value = 2
        mock_comm.gather.return_value = None

        ns = self._load_script_ns()
        ns['collect_local_info'] = MagicMock(return_value={'hostname': 'node1'})

        with patch.dict('sys.modules', self._mock_mpi(mock_comm)), \
             patch('sys.argv', ['script', output_file]):
            ns['main']()

        mock_comm.gather.assert_called_once()
        gathered_info = mock_comm.gather.call_args[0][0]
        assert gathered_info['hostname'] == 'node1'
        assert gathered_info['mpi_rank'] == 1
        assert '_collection_error' not in gathered_info

    def test_gather_still_called_when_collection_raises(self, tmp_path):
        """Error path: gather is called even when collect_local_info() raises."""
        output_file = str(tmp_path / 'out.json')
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 1
        mock_comm.Get_size.return_value = 2
        mock_comm.gather.return_value = None

        ns = self._load_script_ns()
        ns['collect_local_info'] = MagicMock(side_effect=RuntimeError('disk read failed'))

        with patch.dict('sys.modules', self._mock_mpi(mock_comm)), \
             patch('sys.argv', ['script', output_file]):
            ns['main']()

        mock_comm.gather.assert_called_once()

    def test_sentinel_has_collection_error_key(self, tmp_path):
        """Error sentinel must contain _collection_error so callers can detect failures."""
        output_file = str(tmp_path / 'out.json')
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 1
        mock_comm.Get_size.return_value = 2
        mock_comm.gather.return_value = None

        ns = self._load_script_ns()
        ns['collect_local_info'] = MagicMock(side_effect=RuntimeError('disk read failed'))

        with patch.dict('sys.modules', self._mock_mpi(mock_comm)), \
             patch('sys.argv', ['script', output_file]):
            ns['main']()

        gathered_info = mock_comm.gather.call_args[0][0]
        assert '_collection_error' in gathered_info
        assert 'disk read failed' in gathered_info['_collection_error']

    def test_sentinel_has_hostname_and_rank(self, tmp_path):
        """Error sentinel must carry hostname and mpi_rank so rank 0 can identify the source."""
        output_file = str(tmp_path / 'out.json')
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 2
        mock_comm.Get_size.return_value = 4
        mock_comm.gather.return_value = None

        ns = self._load_script_ns()
        ns['collect_local_info'] = MagicMock(side_effect=OSError('permission denied'))

        with patch.dict('sys.modules', self._mock_mpi(mock_comm)), \
             patch('sys.argv', ['script', output_file]):
            ns['main']()

        gathered_info = mock_comm.gather.call_args[0][0]
        assert gathered_info['mpi_rank'] == 2
        assert 'hostname' in gathered_info

    def test_rank_zero_writes_output_file(self, tmp_path):
        """Rank 0 writes the JSON output file when collection succeeds."""
        output_file = str(tmp_path / 'out.json')
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 1
        mock_comm.gather.return_value = [{'hostname': 'node0', 'mpi_rank': 0}]

        ns = self._load_script_ns()
        ns['collect_local_info'] = MagicMock(return_value={'hostname': 'node0'})

        with patch.dict('sys.modules', self._mock_mpi(mock_comm)), \
             patch('sys.argv', ['script', output_file]):
            ns['main']()

        with open(output_file) as f:
            data = json.load(f)
        assert 'node0' in data

    def test_rank_zero_writes_output_when_another_rank_sent_sentinel(self, tmp_path):
        """Rank 0 writes JSON even when another rank's payload is an error sentinel."""
        output_file = str(tmp_path / 'out.json')
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 2
        mock_comm.gather.return_value = [
            {'hostname': 'node0', 'mpi_rank': 0},
            {'hostname': 'node1', 'mpi_rank': 1, '_collection_error': 'disk read failed'},
        ]

        ns = self._load_script_ns()
        ns['collect_local_info'] = MagicMock(return_value={'hostname': 'node0'})

        with patch.dict('sys.modules', self._mock_mpi(mock_comm)), \
             patch('sys.argv', ['script', output_file]):
            ns['main']()

        with open(output_file) as f:
            data = json.load(f)
        assert 'node0' in data
        assert 'node1' in data
        assert '_collection_error' in data['node1']


class TestHostCPUInfoNumSockets:
    """Tests for HostCPUInfo.num_sockets field and wiring (D-16, COLL-01).

    Plan 02-01 adds an additive `num_sockets: int = 0` field on HostCPUInfo
    and wires it through HostCPUInfo.from_dict and HostInfo.from_collected_data
    so plans 02-02 / 02-03 can populate chassis.cpu_qty from host.cpu.num_sockets.
    """

    def test_host_cpu_info_carries_num_sockets(self):
        """End-to-end: the new field exists, defaults to 0, and from_dict /
        from_collected_data both populate it from the cpuinfo / summarize_cpuinfo
        result. Mirrors the existing factory-roundtrip style in this file."""
        from mlpstorage_py.rules.models import HostCPUInfo, HostInfo

        # Case 1: default-zero on a no-arg construction.
        cpu_default = HostCPUInfo()
        assert cpu_default.num_sockets == 0

        # Case 2: explicit keyword construction.
        cpu_explicit = HostCPUInfo(num_sockets=2)
        assert cpu_explicit.num_sockets == 2

        # Case 3: from_dict reads num_sockets from the dict.
        cpu_from_dict = HostCPUInfo.from_dict({
            'num_cores': 4,
            'num_logical_cores': 8,
            'model': 'X',
            'architecture': 'x86_64',
            'num_sockets': 2,
        })
        assert cpu_from_dict.num_sockets == 2

        # Case 4: from_dict defaults num_sockets to 0 when missing.
        cpu_from_dict_default = HostCPUInfo.from_dict({})
        assert cpu_from_dict_default.num_sockets == 0

        # Case 5 (primary D-16 wiring): two-socket cpuinfo → num_sockets == 2.
        # Build a minimal /proc/cpuinfo-shaped list with two distinct physical ids.
        # summarize_cpuinfo counts unique 'physical id' values.
        cpuinfo_two_sockets = [
            {'processor': 0, 'physical id': 0, 'core id': 0,
             'model name': 'Test CPU', 'flags': 'fpu lm'},
            {'processor': 1, 'physical id': 0, 'core id': 1,
             'model name': 'Test CPU', 'flags': 'fpu lm'},
            {'processor': 2, 'physical id': 1, 'core id': 0,
             'model name': 'Test CPU', 'flags': 'fpu lm'},
            {'processor': 3, 'physical id': 1, 'core id': 1,
             'model name': 'Test CPU', 'flags': 'fpu lm'},
        ]
        host_two = HostInfo.from_collected_data({
            'hostname': 'h',
            'cpuinfo': cpuinfo_two_sockets,
        })
        assert host_two.cpu is not None
        assert host_two.cpu.num_sockets == 2

        # Case 6 (single-socket fallback): cpuinfo with no 'physical id' keys
        # falls through to summarize_cpuinfo's `else 1` branch.
        cpuinfo_no_phys_id = [
            {'processor': 0, 'model name': 'Test CPU', 'flags': 'fpu lm'},
            {'processor': 1, 'model name': 'Test CPU', 'flags': 'fpu lm'},
        ]
        host_one = HostInfo.from_collected_data({
            'hostname': 'h',
            'cpuinfo': cpuinfo_no_phys_id,
        })
        assert host_one.cpu is not None
        assert host_one.cpu.num_sockets == 1


# =============================================================================
# Phase 3 Plan 02 — Chassis Model Collector (D-21, COLL-03)
# =============================================================================
#
# Tests for the chassis-model sysfs reader: _DMI_PLACEHOLDERS frozenset,
# _normalize_dmi case-insensitive helper, collect_chassis_model file reader,
# and MPI_COLLECTOR_SCRIPT vs. module parity for the duplicated implementations.
#
# Pattern: tmp_path + path-indirection (RESEARCH lines 738-792). Avoids
# patching builtins.open which would corrupt PyYAML and any other I/O the
# test runner is doing concurrently.
# =============================================================================


# Per D-21, the ten placeholder strings that BIOS vendors leave when a board
# ships without a real product_name. The collector must collapse all ten to ""
# regardless of case, after .strip().
_DMI_PLACEHOLDER_ORIGINALS = [
    "",
    "To Be Filled By O.E.M.",
    "Default string",
    "System Product Name",
    "System manufacturer",
    "None",
    "Not Specified",
    "Not Applicable",
    "OEM",
    "unknown",
]


def _mixed_case(s: str) -> str:
    """Build a deterministic mixed-case variant of s by alternating .upper()/.lower()
    per character. Used to exercise the case-insensitive comparison without
    depending on a hand-typed case-variant table."""
    out = []
    for i, ch in enumerate(s):
        out.append(ch.upper() if i % 2 == 0 else ch.lower())
    return "".join(out)


_DMI_PLACEHOLDER_CASE_CASES = (
    [(s, "") for s in _DMI_PLACEHOLDER_ORIGINALS]
    + [(s.upper(), "") for s in _DMI_PLACEHOLDER_ORIGINALS]
    + [(_mixed_case(s), "") for s in _DMI_PLACEHOLDER_ORIGINALS]
)


class TestDMIPlaceholders:
    """D-21 placeholder normalization: case-insensitive, post-strip."""

    @pytest.mark.parametrize("inp,expected", _DMI_PLACEHOLDER_CASE_CASES)
    def test_each_placeholder_normalizes_to_empty(self, inp, expected):
        """All 10 D-21 placeholders in 3 case variants each (30 cases total)
        normalize to the empty string. _normalize_dmi must lower() the stripped
        input before set-membership testing."""
        from mlpstorage_py.cluster_collector import _normalize_dmi
        assert _normalize_dmi(inp) == expected

    def test_real_product_name_passes_through(self):
        """A legitimate vendor model string passes through unchanged (modulo strip).
        The .strip() preserves internal whitespace and case for real names."""
        from mlpstorage_py.cluster_collector import _normalize_dmi
        assert _normalize_dmi("PowerEdge R760") == "PowerEdge R760"
        assert _normalize_dmi("Supermicro AS-1024US-TRT") == "Supermicro AS-1024US-TRT"

    def test_strip_then_compare(self):
        """The strip happens BEFORE the placeholder lookup, so leading/trailing
        whitespace (including newlines and tabs as sysfs reads commonly carry
        them) does not prevent placeholder collapse."""
        from mlpstorage_py.cluster_collector import _normalize_dmi
        assert _normalize_dmi("  Default string\n") == ""
        assert _normalize_dmi("\tDefault string  ") == ""


class TestCollectChassisModel:
    """Sysfs reader for /sys/class/dmi/id/product_name with universal D-2
    collection-failure rule (any exception → empty string)."""

    def test_reads_dmi_file(self, tmp_path):
        """Happy path: file present and readable, returns the normalized
        product name. Uses dmi_path indirection (RESEARCH 738-792) so we
        never patch builtins.open."""
        from mlpstorage_py.cluster_collector import collect_chassis_model
        p = tmp_path / "product_name"
        p.write_text("PowerEdge R760\n")
        assert collect_chassis_model(dmi_path=str(p)) == "PowerEdge R760"

    def test_placeholder_file_returns_empty(self, tmp_path):
        """Real BIOS junk on disk: file present but contents are a D-21
        placeholder → empty string per D-21 normalization."""
        from mlpstorage_py.cluster_collector import collect_chassis_model
        p = tmp_path / "product_name"
        p.write_text("To Be Filled By O.E.M.\n")
        assert collect_chassis_model(dmi_path=str(p)) == ""

    def test_missing_file_returns_empty(self, tmp_path):
        """D-2 universal failure rule: missing DMI file (e.g., container
        without DMI passthrough, WSL2 kernel) → empty string, no exception."""
        from mlpstorage_py.cluster_collector import collect_chassis_model
        nonexistent = tmp_path / "no_such_file"
        assert collect_chassis_model(dmi_path=str(nonexistent)) == ""

    @pytest.mark.skipif(
        os.geteuid() == 0,
        reason="root bypasses chmod 0o000 file-mode permission denial",
    )
    def test_unreadable_file_returns_empty(self, tmp_path):
        """D-2 universal failure rule: permission denied (hardened container,
        SELinux confinement) → empty string. We restore perms in finally
        so tmp_path cleanup does not error."""
        from mlpstorage_py.cluster_collector import collect_chassis_model
        p = tmp_path / "product_name"
        p.write_text("PowerEdge R760\n")
        try:
            os.chmod(str(p), 0o000)
            assert collect_chassis_model(dmi_path=str(p)) == ""
        finally:
            os.chmod(str(p), 0o644)

    def test_local_system_info_includes_chassis_model_key(self):
        """The collect_local_system_info top-level orchestrator wires
        chassis_model into its result dict. Universal-failure means the
        value is always a string (possibly empty) and the key is always
        present — this contract is what HostInfo.from_collected_data
        will rely on in Plan 03-05."""
        result = collect_local_system_info()
        assert "chassis_model" in result
        assert isinstance(result["chassis_model"], str)


class TestMPIScriptParity:
    """Pattern B (RESEARCH 675-679): the MPI worker script duplicates
    parse_proc_meminfo + parse_os_release inline because it runs across
    SSH on heterogeneous Python environments. Phase 3 extends this with
    _DMI_PLACEHOLDERS, _normalize_dmi, and collect_chassis_model. Drift
    between the two copies produces divergent per-host data shapes;
    this parity test catches any such drift at unit-test time."""

    def test_chassis_functions_match_module(self):
        """exec the MPI_COLLECTOR_SCRIPT in a controlled namespace, then
        compare its _normalize_dmi against the module's. The script's
        top-level code may raise (no mpi4py on dev shells, SystemExit
        from the exit(1) path), but the function DEFs hit the namespace
        BEFORE the top-level code executes, so we wrap exec in a broad
        try/except to keep the parity check workable on any host."""
        from mlpstorage_py.cluster_collector import _normalize_dmi
        ns = {}
        try:
            exec(MPI_COLLECTOR_SCRIPT, ns)
        except BaseException:
            # SystemExit, ImportError, NameError, AttributeError, and
            # anything else from the MPI-only top-level — we only care
            # that the function DEFs landed in ns before the exception.
            pass
        assert '_normalize_dmi' in ns, (
            "MPI_COLLECTOR_SCRIPT must define _normalize_dmi inline (Pattern B)."
        )
        assert 'collect_chassis_model' in ns, (
            "MPI_COLLECTOR_SCRIPT must define collect_chassis_model inline (Pattern B)."
        )
        assert '_DMI_PLACEHOLDERS' in ns, (
            "MPI_COLLECTOR_SCRIPT must define _DMI_PLACEHOLDERS inline (Pattern B)."
        )
        for sample in [
            "PowerEdge R760",
            "Default string",
            "",
            "  default STRING  ",
        ]:
            assert ns['_normalize_dmi'](sample) == _normalize_dmi(sample), (
                f"MPI-script _normalize_dmi diverged from module on {sample!r}"
            )


# =============================================================================
# Phase 3 Plan 03 — Networking Collector (D-18 filter scope, D-19 IB-first,
# D-20 operstate mapping + effective-state demotion). RESEARCH 484-519 decision
# tree; RESEARCH 763-851 tmp_path fixture patterns (Pattern D — NOT mock_open).
# =============================================================================


def _make_iface(net_dir, name, *, type_val="1", operstate="up", speed="10000",
                bonding_slaves=None, master_target=None, bridge=False):
    """Build a fake /sys/class/net/<name>/ directory tree.

    - type_val: contents of <iface>/type (default "1" == ARPHRD_ETHER).
    - operstate: contents of <iface>/operstate.
    - speed: contents of <iface>/speed; pass None to OMIT the file entirely
      (simulates drivers that don't expose the speed file at all, where read
      raises FileNotFoundError and the helper returns the default -1).
    - bonding_slaves: when not None, makes the iface a bond master by creating
      <iface>/bonding/slaves with the given content.
    - master_target: when set, creates <iface>/master symlink pointing at
      net_dir/<master_target> (matches the D-18 bond-slave detection branch).
    - bridge: when True, creates <iface>/bridge/ subdir → bridge-master skip.

    Returns the iface directory Path (caller may add iflink/ifindex etc.).
    """
    from pathlib import Path
    d = Path(net_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "type").write_text(type_val)
    (d / "operstate").write_text(operstate)
    if speed is not None:
        (d / "speed").write_text(speed)
    if bonding_slaves is not None:
        (d / "bonding").mkdir(exist_ok=True)
        (d / "bonding" / "slaves").write_text(bonding_slaves)
    if master_target is not None:
        # Permissive (Linux allows dangling symlinks); os.readlink reads the
        # raw target string without dereferencing.
        try:
            (d / "master").symlink_to(Path(net_dir) / master_target)
        except (OSError, NotImplementedError):
            # macOS / locked-down envs may refuse; we don't run there in CI
            # but be defensive so the fixture still constructs.
            pass
    if bridge:
        # D-18: bridge masters carry /sys/class/net/<iface>/bridge/ subdir.
        (d / "bridge").mkdir(exist_ok=True)
    return d


def _make_ib_port(ib_dir, dev, port, *, state="4: ACTIVE\n",
                  rate="100 Gb/sec (4X EDR)\n"):
    """Build a fake /sys/class/infiniband/<dev>/ports/<port>/{state,rate}."""
    from pathlib import Path
    p = Path(ib_dir) / dev / "ports" / port
    p.mkdir(parents=True, exist_ok=True)
    (p / "state").write_text(state)
    (p / "rate").write_text(rate)
    return p


class TestNetworkingFilters:
    """D-18 interface filtering: lo, docker*, virbr*, veth*, tun*, tap*, gre*,
    wg*, ib*, iboeth*, ib_eth*, bridge masters, VLAN sub-interfaces, MACVLAN/
    IPVLAN sub-interfaces, bond slaves. All cases include a real eth0 to lock
    that the filter excludes ONLY the offender, not the whole walk."""

    @pytest.mark.parametrize("excluded_name", [
        "lo",
        "docker0",
        "docker123",
        "virbr0",
        "veth123abc",
        "veth9d8e3f",
        "tun0",
        "tap0",
        "gre0",
        "wg0",
    ])
    def test_excluded_iface_by_name_prefix(self, tmp_path, excluded_name):
        """D-18 name-prefix shortcut: each excluded prefix variant is filtered
        from the net walk; eth0 (always included) confirms the walk did not
        early-exit on the offender."""
        from mlpstorage_py.cluster_collector import collect_networking
        net = tmp_path / "net"
        net.mkdir()
        _make_iface(net, "eth0", operstate="up", speed="10000")
        _make_iface(net, excluded_name, operstate="up", speed="10000")
        result = collect_networking(net_root=str(net),
                                    ib_root=str(tmp_path / "no_ib"))
        types = [e["type"] for e in result]
        assert types == ["ethernet"], (
            f"Expected only eth0 to survive; got {result!r}"
        )

    def test_bridge_master_filtered(self, tmp_path):
        """D-18: a bridge master (br0 with /bridge/ subdir) is skipped because
        its speed is meaningless (kernel reports 10 regardless of real ports)."""
        from mlpstorage_py.cluster_collector import collect_networking
        net = tmp_path / "net"
        net.mkdir()
        _make_iface(net, "eth0", operstate="up", speed="10000")
        _make_iface(net, "br0", operstate="up", speed="10", bridge=True)
        result = collect_networking(net_root=str(net),
                                    ib_root=str(tmp_path / "no_ib"))
        types_and_speeds = [(e["type"], e.get("speed")) for e in result]
        assert types_and_speeds == [("ethernet", 10)], (
            f"Bridge master br0 must be filtered; got {result!r}"
        )

    def test_vlan_subif_filtered_via_iflink_mismatch(self, tmp_path):
        """D-18: VLAN sub-interface eth0.100 has iflink != ifindex pointing
        at the parent. Filter detects the mismatch and skips the sub-iface."""
        from mlpstorage_py.cluster_collector import collect_networking
        net = tmp_path / "net"
        net.mkdir()
        # Parent eth0: iflink == ifindex
        eth0 = _make_iface(net, "eth0", operstate="up", speed="10000")
        (eth0 / "iflink").write_text("2")
        (eth0 / "ifindex").write_text("2")
        # VLAN child: iflink points back at eth0 (2), ifindex is its own (5)
        vlan = _make_iface(net, "eth0.100", operstate="up", speed="10000")
        (vlan / "iflink").write_text("2")
        (vlan / "ifindex").write_text("5")
        result = collect_networking(net_root=str(net),
                                    ib_root=str(tmp_path / "no_ib"))
        assert len(result) == 1, f"VLAN sub-iface must be filtered; got {result!r}"
        assert result[0]["type"] == "ethernet"
        assert result[0]["speed"] == 10

    def test_bond_slave_filtered(self, tmp_path):
        """D-18: a NIC whose <iface>/master symlink target basename starts
        with 'bond' is a bond slave and must be filtered. Bond aggregation
        is exercised separately (TestNetworkingBond)."""
        from mlpstorage_py.cluster_collector import collect_networking
        net = tmp_path / "net"
        net.mkdir()
        _make_iface(net, "eth0", operstate="up", speed="10000")
        # Make a bond master so the master target exists in this fixture
        _make_iface(net, "bond0", operstate="up", speed="10000",
                    bonding_slaves="eth_slave\n")
        # Slave: master symlinks to bond0
        _make_iface(net, "eth_slave", operstate="up", speed="10000",
                    master_target="bond0")
        result = collect_networking(net_root=str(net),
                                    ib_root=str(tmp_path / "no_ib"))
        # eth0 (1) + bond0 (1, aggregated) = 2 entries; eth_slave filtered.
        assert len(result) == 2, (
            f"Bond slave must be filtered; got {result!r}"
        )

    def test_ib_prefix_filtered_from_net_walk(self, tmp_path):
        """D-19 belt-and-suspenders: ib0 (IPoIB shadow) under /sys/class/net
        is skipped because the IB walk is the authoritative source. iboeth*
        and ib_eth* are also skipped per D-18 forward-compat."""
        from mlpstorage_py.cluster_collector import collect_networking
        net = tmp_path / "net"
        net.mkdir()
        _make_iface(net, "eth0", operstate="up", speed="10000")
        _make_iface(net, "ib0", operstate="up", speed="100000")
        _make_iface(net, "iboeth0", operstate="up", speed="100000")
        _make_iface(net, "ib_eth0", operstate="up", speed="100000")
        result = collect_networking(net_root=str(net),
                                    ib_root=str(tmp_path / "no_ib"))
        types_and_speeds = [(e["type"], e.get("speed")) for e in result]
        assert types_and_speeds == [("ethernet", 10)], (
            f"ib* prefixes must be filtered from net walk; got {result!r}"
        )


class TestNetworkingBond:
    """D-18 bond master aggregation: emit one entry per LAG with speed = sum
    of active slave speeds (Mbps) // 1000. Bond's own speed file is ignored
    per Pitfall 4 (unreliable on many drivers)."""

    def test_bond_master_aggregate_speed(self, tmp_path):
        """Two-slave 10G LAG: bond0 emits ONE entry with speed=20 (Gbps)."""
        from mlpstorage_py.cluster_collector import collect_networking
        net = tmp_path / "net"
        net.mkdir()
        _make_iface(net, "eth1", operstate="up", speed="10000",
                    master_target="bond0")
        _make_iface(net, "eth2", operstate="up", speed="10000",
                    master_target="bond0")
        _make_iface(net, "bond0", operstate="up", speed="10000",
                    bonding_slaves="eth1 eth2\n")
        result = collect_networking(net_root=str(net),
                                    ib_root=str(tmp_path / "no_ib"))
        assert len(result) == 1, f"Bond aggregate should be ONE entry; got {result!r}"
        assert result[0] == {"type": "ethernet", "speed": 20, "state": "up"}

    def test_bond_master_with_all_slaves_down(self, tmp_path):
        """All slaves report speed=-1; aggregate=0 → bond emits as down with
        no speed key. Submitter sees a visible degraded LAG."""
        from mlpstorage_py.cluster_collector import collect_networking
        net = tmp_path / "net"
        net.mkdir()
        _make_iface(net, "eth1", operstate="down", speed="-1",
                    master_target="bond0")
        _make_iface(net, "eth2", operstate="down", speed="-1",
                    master_target="bond0")
        _make_iface(net, "bond0", operstate="up", speed="-1",
                    bonding_slaves="eth1 eth2\n")
        result = collect_networking(net_root=str(net),
                                    ib_root=str(tmp_path / "no_ib"))
        assert len(result) == 1
        assert result[0] == {"type": "ethernet", "state": "down"}, (
            f"Down bond must emit no speed key; got {result[0]!r}"
        )

    def test_bond_master_single_active_slave(self, tmp_path):
        """Active-backup: one slave up @10G, one down. Aggregate = 10 Gbps."""
        from mlpstorage_py.cluster_collector import collect_networking
        net = tmp_path / "net"
        net.mkdir()
        _make_iface(net, "eth1", operstate="up", speed="10000",
                    master_target="bond0")
        _make_iface(net, "eth2", operstate="down", speed="-1",
                    master_target="bond0")
        _make_iface(net, "bond0", operstate="up", speed="10000",
                    bonding_slaves="eth1 eth2\n")
        result = collect_networking(net_root=str(net),
                                    ib_root=str(tmp_path / "no_ib"))
        assert len(result) == 1
        assert result[0] == {"type": "ethernet", "speed": 10, "state": "up"}


class TestNetworkingOperstate:
    """D-20 operstate mapping: up | unknown → up; everything else → down."""

    def test_operstate_up_passes_through(self, tmp_path):
        from mlpstorage_py.cluster_collector import collect_networking
        net = tmp_path / "net"
        net.mkdir()
        _make_iface(net, "eth0", operstate="up", speed="10000")
        result = collect_networking(net_root=str(net),
                                    ib_root=str(tmp_path / "no_ib"))
        assert result == [{"type": "ethernet", "speed": 10, "state": "up"}]

    def test_operstate_unknown_treated_as_up(self, tmp_path):
        """D-20 permissive mapping: 'unknown' → up (virtio drivers don't
        update operstate reliably; ignoring this would systematically
        misreport VM NICs as down)."""
        from mlpstorage_py.cluster_collector import collect_networking
        net = tmp_path / "net"
        net.mkdir()
        _make_iface(net, "eth0", operstate="unknown", speed="10000")
        result = collect_networking(net_root=str(net),
                                    ib_root=str(tmp_path / "no_ib"))
        assert result == [{"type": "ethernet", "speed": 10, "state": "up"}]

    def test_operstate_down_emits_down(self, tmp_path):
        """D-20: operstate=down → state=down with no speed key emitted.
        Same shape for dormant/notpresent/lowerlayerdown/testing (everything
        not in {up,unknown}); we exercise the canonical 'down' value here."""
        from mlpstorage_py.cluster_collector import collect_networking
        net = tmp_path / "net"
        net.mkdir()
        _make_iface(net, "eth0", operstate="down", speed="-1")
        result = collect_networking(net_root=str(net),
                                    ib_root=str(tmp_path / "no_ib"))
        assert result == [{"type": "ethernet", "state": "down"}]


class TestNetworkingEffectiveState:
    """D-20 effective-state demotion: operstate=up AND speed in {-1, 0}
    means state=down. Pitfall 2 (virtio speed=-1) lock."""

    def test_virtio_speed_minus_one_demotes_to_down(self, tmp_path):
        """The Pitfall 2 lock: a virtio NIC reports operstate=up + speed=-1.
        Without demotion, the emit would be (speed: -1, state: up) which is
        Pydantic-invalid (speed has ge=1). D-20 fixes this at the collector
        layer so downstream never sees the invalid combination."""
        from mlpstorage_py.cluster_collector import collect_networking
        net = tmp_path / "net"
        net.mkdir()
        _make_iface(net, "eth0", operstate="up", speed="-1")
        result = collect_networking(net_root=str(net),
                                    ib_root=str(tmp_path / "no_ib"))
        assert result == [{"type": "ethernet", "state": "down"}], (
            f"virtio speed=-1 must demote to down; got {result!r}"
        )

    def test_speed_zero_demotes_to_down(self, tmp_path):
        """D-20: speed=0 with operstate=up is operationally equivalent to
        down — an interface that negotiated nothing is not passing traffic."""
        from mlpstorage_py.cluster_collector import collect_networking
        net = tmp_path / "net"
        net.mkdir()
        _make_iface(net, "eth0", operstate="up", speed="0")
        result = collect_networking(net_root=str(net),
                                    ib_root=str(tmp_path / "no_ib"))
        assert result == [{"type": "ethernet", "state": "down"}]


class TestNetworkingInfiniband:
    """D-19 IB-first: walk /sys/class/infiniband/<dev>/ports/<port>/. One
    entry per port (a dual-port HCA produces two entries). State parses
    '4:' prefix → up; everything else → down. Rate parses int(rate.split()[0])."""

    def test_active_ib_port_emits_up(self, tmp_path):
        from mlpstorage_py.cluster_collector import collect_networking
        ib = tmp_path / "ib"
        ib.mkdir()
        _make_ib_port(ib, "mlx5_0", "1",
                      state="4: ACTIVE\n", rate="100 Gb/sec (4X EDR)\n")
        result = collect_networking(net_root=str(tmp_path / "no_net"),
                                    ib_root=str(ib))
        assert result == [{"type": "infiniband", "speed": 100, "state": "up"}]

    def test_down_ib_port_emits_down_no_speed(self, tmp_path):
        """D-19: state != '4:' → down; emit no speed key (consistent with
        ethernet down emission so the splice path handles both identically)."""
        from mlpstorage_py.cluster_collector import collect_networking
        ib = tmp_path / "ib"
        ib.mkdir()
        _make_ib_port(ib, "mlx5_0", "1",
                      state="1: DOWN\n", rate="0 Gb/sec\n")
        result = collect_networking(net_root=str(tmp_path / "no_net"),
                                    ib_root=str(ib))
        assert result == [{"type": "infiniband", "state": "down"}]

    def test_dual_port_hca_emits_two_entries(self, tmp_path):
        """D-19 port-per-entry: a single device with two ports produces
        two networking entries. Plan 03-04's group_by_fingerprint then
        collapses identical (type,speed,state) tuples to unit_count=2."""
        from mlpstorage_py.cluster_collector import collect_networking
        ib = tmp_path / "ib"
        ib.mkdir()
        _make_ib_port(ib, "mlx5_0", "1")
        _make_ib_port(ib, "mlx5_0", "2")
        result = collect_networking(net_root=str(tmp_path / "no_net"),
                                    ib_root=str(ib))
        assert len(result) == 2
        for entry in result:
            assert entry == {"type": "infiniband", "speed": 100, "state": "up"}

    def test_no_ib_root_returns_empty_ib_section(self, tmp_path):
        """The os.path.isdir(ib_root) guard makes a missing /sys/class/
        infiniband (the common case on hosts without IB hardware) a clean
        no-op rather than an exception."""
        from mlpstorage_py.cluster_collector import collect_networking
        net = tmp_path / "net"
        net.mkdir()
        _make_iface(net, "eth0", operstate="up", speed="10000")
        result = collect_networking(net_root=str(net),
                                    ib_root=str(tmp_path / "definitely_no_ib"))
        # Just the ethernet entry; no IB-derived entries.
        assert result == [{"type": "ethernet", "speed": 10, "state": "up"}]

    def test_unparseable_rate_demotes_to_down(self, tmp_path):
        """D-19 blank-splice for unparseable rate: state says ACTIVE but the
        rate file is empty / garbled — without a parseable speed the entry
        cannot truthfully claim 'up' (Pydantic NetworkPort.state=='up'
        requires speed). Emit as down with no speed key."""
        from mlpstorage_py.cluster_collector import collect_networking
        ib = tmp_path / "ib"
        ib.mkdir()
        _make_ib_port(ib, "mlx5_0", "1",
                      state="4: ACTIVE\n", rate="")
        result = collect_networking(net_root=str(tmp_path / "no_net"),
                                    ib_root=str(ib))
        assert result == [{"type": "infiniband", "state": "down"}]


class TestNetworkingHotUnplug:
    """Per-iface defense (D-2 universal rule applied at iface scope): a NIC
    that disappears between os.listdir and the per-iface reads must be
    silently skipped; no exception escapes collect_networking."""

    def test_iface_disappears_between_listdir_and_read_skipped(self, tmp_path):
        """Simulate hot-unplug: os.listdir returns eth0, but open(eth0/type)
        raises FileNotFoundError because the kernel removed the iface dir
        between enumeration and read. Function must return [], not raise."""
        from mlpstorage_py.cluster_collector import collect_networking
        net = tmp_path / "net"
        net.mkdir()
        # Pretend eth0 was there at listdir time but its files are absent
        # by the time we try to read. Use os.listdir patching to inject
        # the name, then the natural FileNotFoundError on open(net/eth0/type)
        # exercises the per-iface try/except.
        real_listdir = os.listdir

        def fake_listdir(path):
            if path == str(net):
                return ["eth0"]
            return real_listdir(path)

        with patch("mlpstorage_py.cluster_collector.os.listdir",
                   side_effect=fake_listdir):
            result = collect_networking(net_root=str(net),
                                        ib_root=str(tmp_path / "no_ib"))
        assert result == [], (
            f"Hot-unplug must skip the iface, not raise; got {result!r}"
        )


class TestNetworkingIntegration:
    """The top-level collect_local_system_info orchestrator wires networking
    into its result dict per the same Pattern A try/except as the rest of
    the per-field reads."""

    def test_local_system_info_includes_networking_key(self):
        """Universal-rule contract: the networking key is always present and
        is always a list (possibly empty on a host with no real NICs to
        report, e.g. dev shell with only virtuals)."""
        result = collect_local_system_info()
        assert "networking" in result
        assert isinstance(result["networking"], list)


class TestNetworkingMPIScriptParity:
    """Pattern B (RESEARCH 675-679) extension to networking: every new
    module-scope symbol added to the collector also lives inside the
    MPI worker script string. Drift between the two copies produces
    divergent per-host data shapes; this parity test catches drift on
    the canonical sysfs fixture from RESEARCH 763-822."""

    def test_networking_functions_match_module(self, tmp_path):
        """exec MPI_COLLECTOR_SCRIPT in a controlled namespace; build a
        small ethernet+IB fixture; assert the script's collect_networking
        produces the same output as the module's on that fixture."""
        from mlpstorage_py.cluster_collector import collect_networking
        ns = {}
        try:
            exec(MPI_COLLECTOR_SCRIPT, ns)
        except BaseException:
            # SystemExit / ImportError / etc from the MPI-only top-level;
            # function DEFs already landed before the raise.
            pass
        assert "collect_networking" in ns, (
            "MPI_COLLECTOR_SCRIPT must define collect_networking inline "
            "(Pattern B duplication)."
        )
        # Ethernet fixture
        net = tmp_path / "net"
        net.mkdir()
        _make_iface(net, "eth0", operstate="up", speed="10000")
        # IB fixture
        ib = tmp_path / "ib"
        ib.mkdir()
        _make_ib_port(ib, "mlx5_0", "1")
        a = ns["collect_networking"](str(net), str(ib))
        b = collect_networking(str(net), str(ib))
        assert a == b, (
            f"MPI-script collect_networking diverged from module: "
            f"script={a!r} module={b!r}"
        )


# =============================================================================
# Phase 3 Plan 05 — HostInfo dataclass extensions (COLL-03, COLL-04)
# =============================================================================
#
# Plan 03-05 closes the Phase 3 vertical end-to-end by adding two new fields to
# the HostInfo dataclass — chassis_model (str, populated from
# data['chassis_model'] in from_collected_data) and networking
# (List[Dict[str, Any]], populated from data['networking']).
#
# These tests are the data-model-side RED for Plan 03-05 Task 1. They mirror
# the D-16 num_sockets precedent already exercised by
# TestHostCPUInfoNumSockets (this file, ~line 1361). Defaults are "" and []
# (the universal D-2 collection-failure blanks); missing keys flow through to
# those defaults; populated keys flow through to the dataclass.
# =============================================================================


class TestHostInfoChassisField:
    """Phase 3 / Plan 03-05 — HostInfo.chassis_model field (COLL-03).

    Mirror of D-16 / TestHostCPUInfoNumSockets: dataclass field with a
    sensible default exists; from_collected_data reads the corresponding
    dict key and populates it; missing dict key flows to the default.
    """

    def test_default_empty_string(self):
        """HostInfo(hostname='h').chassis_model defaults to ''."""
        from mlpstorage_py.rules.models import HostInfo

        host = HostInfo(hostname="h")
        assert host.chassis_model == ""
        assert isinstance(host.chassis_model, str)

    def test_populated_via_from_collected_data(self):
        """from_collected_data reads data['chassis_model'] and stores it
        on the resulting HostInfo."""
        from mlpstorage_py.rules.models import HostInfo

        host = HostInfo.from_collected_data({
            "hostname": "h",
            "chassis_model": "PowerEdge R760",
        })
        assert host.chassis_model == "PowerEdge R760"

    def test_missing_chassis_model_key_defaults_to_empty(self):
        """from_collected_data with no 'chassis_model' key → '' default
        (universal D-2 collection-failure blank)."""
        from mlpstorage_py.rules.models import HostInfo

        host = HostInfo.from_collected_data({"hostname": "h"})
        assert host.chassis_model == ""


class TestHostInfoNetworkingField:
    """Phase 3 / Plan 03-05 — HostInfo.networking field (COLL-04).

    Same shape as TestHostInfoChassisField but for the per-host networking
    list. Default is [] (empty list); populated value flows through
    verbatim; missing key flows to the default.
    """

    def test_default_empty_list(self):
        """HostInfo(hostname='h').networking defaults to []."""
        from mlpstorage_py.rules.models import HostInfo

        host = HostInfo(hostname="h")
        assert host.networking == []
        assert isinstance(host.networking, list)

    def test_populated_via_from_collected_data(self):
        """from_collected_data reads data['networking'] (list of dicts)
        verbatim onto the dataclass field."""
        from mlpstorage_py.rules.models import HostInfo

        networking = [
            {"type": "ethernet", "speed": 100, "state": "up"},
            {"type": "infiniband", "speed": 200, "state": "up"},
        ]
        host = HostInfo.from_collected_data({
            "hostname": "h",
            "networking": networking,
        })
        assert host.networking == networking

    def test_missing_networking_key_defaults_to_empty_list(self):
        """from_collected_data with no 'networking' key → [] default
        (universal D-2 collection-failure blank, list-shape)."""
        from mlpstorage_py.rules.models import HostInfo

        host = HostInfo.from_collected_data({"hostname": "h"})
        assert host.networking == []


# =============================================================================
# Phase 4 / Plan 04-05 — HostInfo.sysctl / .environment / .drives field
# extensions (COLL-05 / COLL-06 / COLL-07 data-model wire-through).
#
# Mirrors the Phase 3 D-16 / TestHostInfoChassisField + TestHostInfoNetworkingField
# precedent: three new list-typed dataclass fields with `default_factory=list`
# and corresponding `data.get(..., [])` reads in `from_collected_data`. Defaults
# are []; missing keys flow through to the default; populated lists flow
# through verbatim. `from_dict` remains untouched (Phase 3 precedent).
# =============================================================================


class TestHostInfoSysctlField:
    """Phase 4 / Plan 04-05 — HostInfo.sysctl field (COLL-05).

    Mirror of TestHostInfoNetworkingField: dataclass field with a sensible
    default (empty list), `from_collected_data` reads `data['sysctl']`, missing
    key flows to default.
    """

    def test_default_empty_list(self):
        """HostInfo(hostname='h').sysctl defaults to []."""
        from mlpstorage_py.rules.models import HostInfo

        host = HostInfo(hostname="h")
        assert host.sysctl == []
        assert isinstance(host.sysctl, list)

    def test_constructed_with_populated_list(self):
        """Direct dataclass construction with a populated sysctl list."""
        from mlpstorage_py.rules.models import HostInfo

        sysctl = [{"name": "vm.dirty_ratio", "value": "20"}]
        host = HostInfo(hostname="h", sysctl=sysctl)
        assert host.sysctl == [{"name": "vm.dirty_ratio", "value": "20"}]

    def test_from_collected_data_reads_sysctl(self):
        """from_collected_data reads data['sysctl'] (list of dicts) verbatim
        onto the dataclass field."""
        from mlpstorage_py.rules.models import HostInfo

        sysctl = [
            {"name": "vm.dirty_ratio", "value": "20"},
            {"name": "net.core.somaxconn", "value": "4096"},
        ]
        host = HostInfo.from_collected_data({
            "hostname": "h",
            "sysctl": sysctl,
        })
        assert host.sysctl == sysctl

    def test_from_collected_data_missing_sysctl_defaults_to_empty(self):
        """from_collected_data with no 'sysctl' key → [] default
        (universal D-2 collection-failure blank, list-shape)."""
        from mlpstorage_py.rules.models import HostInfo

        host = HostInfo.from_collected_data({"hostname": "h"})
        assert host.sysctl == []


class TestHostInfoEnvironmentField:
    """Phase 4 / Plan 04-05 — HostInfo.environment field (COLL-06).

    Same shape as TestHostInfoSysctlField for the per-host environment list.
    Note: values are assumed already-redacted per the COLL-06 collector
    contract (Plan 04-02 unified `_redact_secret` / `_mask_credential_id`).
    """

    def test_default_empty_list(self):
        """HostInfo(hostname='h').environment defaults to []."""
        from mlpstorage_py.rules.models import HostInfo

        host = HostInfo(hostname="h")
        assert host.environment == []
        assert isinstance(host.environment, list)

    def test_constructed_with_populated_list(self):
        """Direct dataclass construction with a populated environment list."""
        from mlpstorage_py.rules.models import HostInfo

        environment = [
            {"name": "AWS_SECRET_ACCESS_KEY", "value": "[SET — 40 chars]"},
            {"name": "BUCKET", "value": "my-bucket"},
        ]
        host = HostInfo(hostname="h", environment=environment)
        assert host.environment == environment

    def test_from_collected_data_reads_environment(self):
        """from_collected_data reads data['environment'] verbatim onto the
        dataclass field."""
        from mlpstorage_py.rules.models import HostInfo

        environment = [
            {"name": "NCCL_DEBUG", "value": "INFO"},
            {"name": "AWS_ACCESS_KEY_ID", "value": "AKIA****MPLE"},
        ]
        host = HostInfo.from_collected_data({
            "hostname": "h",
            "environment": environment,
        })
        assert host.environment == environment

    def test_from_collected_data_missing_environment_defaults_to_empty(self):
        """from_collected_data with no 'environment' key → [] default."""
        from mlpstorage_py.rules.models import HostInfo

        host = HostInfo.from_collected_data({"hostname": "h"})
        assert host.environment == []


class TestHostInfoDrivesField:
    """Phase 4 / Plan 04-05 — HostInfo.drives field (COLL-07).

    Same shape as TestHostInfoSysctlField for the per-host drives list.
    Drives are emitted as ungrouped per-row dicts by `collect_drives()`;
    `node_dict_from_host` performs the per-host `group_by_fingerprint`
    collapse downstream.
    """

    def test_default_empty_list(self):
        """HostInfo(hostname='h').drives defaults to []."""
        from mlpstorage_py.rules.models import HostInfo

        host = HostInfo(hostname="h")
        assert host.drives == []
        assert isinstance(host.drives, list)

    def test_constructed_with_populated_list(self):
        """Direct dataclass construction with a populated drives list."""
        from mlpstorage_py.rules.models import HostInfo

        drives = [
            {"vendor_name": "INTEL", "model_name": "SSDPED1K375GA",
             "interface": "nvme", "capacity_in_GB": 375},
        ]
        host = HostInfo(hostname="h", drives=drives)
        assert host.drives == drives

    def test_from_collected_data_reads_drives(self):
        """from_collected_data reads data['drives'] verbatim onto the
        dataclass field."""
        from mlpstorage_py.rules.models import HostInfo

        drives = [
            {"vendor_name": "INTEL", "model_name": "X",
             "interface": "nvme", "capacity_in_GB": 500},
            {"vendor_name": "INTEL", "model_name": "X",
             "interface": "nvme", "capacity_in_GB": 500},
        ]
        host = HostInfo.from_collected_data({
            "hostname": "h",
            "drives": drives,
        })
        assert host.drives == drives

    def test_from_collected_data_missing_drives_defaults_to_empty(self):
        """from_collected_data with no 'drives' key → [] default
        (D-33 path: lsblk absent / no devices / all filtered)."""
        from mlpstorage_py.rules.models import HostInfo

        host = HostInfo.from_collected_data({"hostname": "h"})
        assert host.drives == []


# =============================================================================
# Phase 4 Plan 04-01 — Sysctl Collector (D-27 allowlist file, D-28 /proc/sys
# walk semantics, D-29 multi-value verbatim emit, D-36 Pattern B parity).
# RESEARCH Q2 (write-only leaves), Q3 (fnmatch deep-match gotcha), Q5
# (3.8-safe stdlib for the script twin). COLL-05.
# =============================================================================


def _make_sysctl_leaf(proc_sys_root, dotted_name, content):
    """Build a fake /proc/sys leaf at <proc_sys_root>/<slashed_name>.

    `dotted_name` like 'net.ipv4.tcp_rmem' becomes
    '<proc_sys_root>/net/ipv4/tcp_rmem'. Returns the Path of the leaf.
    """
    from pathlib import Path
    parts = dotted_name.split(".")
    p = Path(proc_sys_root)
    for seg in parts[:-1]:
        p = p / seg
    p.mkdir(parents=True, exist_ok=True)
    leaf = p / parts[-1]
    leaf.write_text(content)
    return leaf


def _make_proc_sys_root(tmp_path):
    """Build a tmp_path/proc/sys root directory and return it as a str."""
    root = tmp_path / "proc" / "sys"
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


class TestSysctlAllowlistFile:
    """The shipped allowlist file at
    mlpstorage_py/system_description/sysctl_allowlist.txt is the load-bearing
    artifact for COLL-05: editing it adds keys to the next run's output with
    no code change. These tests pin its existence, structure, and the four
    initial patterns locked by D-27."""

    def test_allowlist_file_exists(self):
        """D-27: the package ships sysctl_allowlist.txt as a data file."""
        from pathlib import Path
        import mlpstorage_py.system_description as sd_pkg
        sd_dir = Path(sd_pkg.__file__).parent
        path = sd_dir / "sysctl_allowlist.txt"
        assert path.exists(), (
            f"D-27: package must ship sysctl_allowlist.txt at {path}"
        )

    def test_allowlist_file_parses_to_four_patterns(self):
        """D-27: the shipped file contains exactly four glob lines
        (vm.dirty_*, net.core.*, net.ipv4.tcp_*, kernel.numa_balancing)."""
        from mlpstorage_py.cluster_collector import _load_sysctl_allowlist
        patterns = _load_sysctl_allowlist()
        assert len(patterns) == 4, (
            f"D-27: expected 4 initial globs, got {len(patterns)}: {patterns!r}"
        )

    def test_shipped_globs_match_canonical_keys(self):
        """D-27: the four globs round-trip to regex objects matching
        vm.dirty_ratio, net.core.rmem_max, net.ipv4.tcp_rmem, kernel.numa_balancing."""
        from mlpstorage_py.cluster_collector import _load_sysctl_allowlist
        patterns = _load_sysctl_allowlist()
        # For each canonical key, at least one pattern must match it.
        canonical_keys = [
            "vm.dirty_ratio",
            "net.core.rmem_max",
            "net.ipv4.tcp_rmem",
            "kernel.numa_balancing",
        ]
        for key in canonical_keys:
            assert any(p.match(key) for p in patterns), (
                f"D-27: no shipped glob matches canonical key {key!r}"
            )


class TestLoadSysctlAllowlist:
    """The _load_sysctl_allowlist helper parses a glob-per-line text file
    into a tuple of compiled regex objects. Comments and blank lines are
    skipped; missing files yield an empty tuple (universal D-2 rule)."""

    def test_skips_blank_lines_and_comments(self, tmp_path):
        """D-27: lines starting with '#' (after lstrip) and empty/whitespace
        lines are ignored. Returns a tuple of length == count of glob lines."""
        from mlpstorage_py.cluster_collector import _load_sysctl_allowlist
        p = tmp_path / "allowlist.txt"
        p.write_text(
            "# header comment\n"
            "\n"
            "vm.dirty_*\n"
            "   # indented comment\n"
            "net.core.*\n"
            "\n"
        )
        patterns = _load_sysctl_allowlist(str(p))
        assert len(patterns) == 2

    def test_strips_trailing_whitespace_per_line(self, tmp_path):
        """Whitespace around the glob is stripped before translate()."""
        from mlpstorage_py.cluster_collector import _load_sysctl_allowlist
        p = tmp_path / "allowlist.txt"
        p.write_text("vm.dirty_*   \n  net.core.*\t\n")
        patterns = _load_sysctl_allowlist(str(p))
        assert len(patterns) == 2
        # The first pattern must match a vm.dirty_* key.
        assert any(pat.match("vm.dirty_ratio") for pat in patterns)
        # The second pattern must match net.core.*.
        assert any(pat.match("net.core.rmem_max") for pat in patterns)

    def test_missing_file_returns_empty_tuple(self, tmp_path):
        """D-2 universal failure rule: missing file → empty tuple, no
        exception. Outer collect_sysctl will receive an empty tuple and emit []."""
        from mlpstorage_py.cluster_collector import _load_sysctl_allowlist
        nonexistent = tmp_path / "no_such_file.txt"
        result = _load_sysctl_allowlist(str(nonexistent))
        assert result == tuple()

    def test_returned_patterns_have_match_method(self, tmp_path):
        """Each returned object must be a compiled regex (exposes .match)."""
        from mlpstorage_py.cluster_collector import _load_sysctl_allowlist
        p = tmp_path / "allowlist.txt"
        p.write_text("vm.dirty_*\n")
        patterns = _load_sysctl_allowlist(str(p))
        assert len(patterns) == 1
        assert hasattr(patterns[0], "match")
        assert patterns[0].match("vm.dirty_ratio") is not None
        assert patterns[0].match("net.core.rmem_max") is None


class TestSysctlCollector:
    """The /proc/sys walk: per-leaf filter against the allowlist, per-leaf
    try/except (universal D-2 / RESEARCH Q2), 8 KiB read cap (D-28),
    multi-value verbatim emit (D-29). Synthetic /proc/sys via tmp_path so
    no real sysctl access is needed."""

    def test_emits_allowlisted_leaves(self, tmp_path):
        """D-27/D-28: only leaves whose dotted form matches at least one
        allowlist pattern are emitted, as {name, value} dicts."""
        import re
        import fnmatch
        from mlpstorage_py.cluster_collector import collect_sysctl
        root = _make_proc_sys_root(tmp_path)
        _make_sysctl_leaf(root, "vm.dirty_ratio", "20\n")
        _make_sysctl_leaf(root, "vm.swappiness", "60\n")
        _make_sysctl_leaf(root, "net.core.rmem_max", "212992\n")
        allowlist = tuple(
            re.compile(fnmatch.translate(g))
            for g in ("vm.dirty_*", "net.core.*")
        )
        out = collect_sysctl(proc_sys_root=root, allowlist=allowlist)
        sorted_out = sorted(out, key=lambda d: d["name"])
        assert sorted_out == [
            {"name": "net.core.rmem_max", "value": "212992"},
            {"name": "vm.dirty_ratio", "value": "20"},
        ]

    def test_excludes_non_allowlisted(self, tmp_path):
        """D-27: leaves matching no allowlist pattern are skipped."""
        import re
        import fnmatch
        from mlpstorage_py.cluster_collector import collect_sysctl
        root = _make_proc_sys_root(tmp_path)
        _make_sysctl_leaf(root, "vm.dirty_ratio", "20\n")
        _make_sysctl_leaf(root, "vm.swappiness", "60\n")
        _make_sysctl_leaf(root, "net.core.rmem_max", "212992\n")
        allowlist = (re.compile(fnmatch.translate("vm.dirty_*")),)
        out = collect_sysctl(proc_sys_root=root, allowlist=allowlist)
        names = {e["name"] for e in out}
        assert "vm.dirty_ratio" in names
        assert "vm.swappiness" not in names
        assert "net.core.rmem_max" not in names

    def test_multi_value_verbatim_per_d29(self, tmp_path):
        """D-29: multi-value leaves (e.g., tcp_rmem returning tab-separated
        triplets) emit verbatim — only the trailing newline is stripped,
        internal tabs preserved."""
        import re
        import fnmatch
        from mlpstorage_py.cluster_collector import collect_sysctl
        root = _make_proc_sys_root(tmp_path)
        _make_sysctl_leaf(root, "net.ipv4.tcp_rmem", "4096\t87380\t16777216\n")
        allowlist = (re.compile(fnmatch.translate("net.ipv4.tcp_*")),)
        out = collect_sysctl(proc_sys_root=root, allowlist=allowlist)
        assert len(out) == 1
        assert out[0]["name"] == "net.ipv4.tcp_rmem"
        assert out[0]["value"] == "4096\t87380\t16777216"

    def test_permission_error_on_leaf_isolates_per_d2(self, tmp_path, monkeypatch):
        """D-2 / RESEARCH Q2: a single write-only or PermissionError leaf
        (vm.drop_caches, sysrq, route/flush) skips itself but never aborts
        the walk. The rest of the matching leaves still emit."""
        import re
        import fnmatch
        from mlpstorage_py.cluster_collector import collect_sysctl
        root = _make_proc_sys_root(tmp_path)
        _make_sysctl_leaf(root, "vm.dirty_ratio", "20\n")
        drop_path = _make_sysctl_leaf(root, "vm.drop_caches", "0\n")

        real_open = open

        def fake_open(file, *args, **kwargs):
            if str(file) == str(drop_path):
                raise PermissionError("write-only")
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)
        allowlist = (re.compile(fnmatch.translate("vm.*")),)
        out = collect_sysctl(proc_sys_root=root, allowlist=allowlist)
        names = {e["name"] for e in out}
        assert "vm.dirty_ratio" in names
        assert "vm.drop_caches" not in names

    def test_missing_proc_sys_root_returns_empty(self, tmp_path):
        """D-2: catastrophic failure (no /proc/sys) → empty list, no
        exception. Outer collect_local_system_info wraps in another
        try/except for defense-in-depth."""
        from mlpstorage_py.cluster_collector import collect_sysctl
        nonexistent = tmp_path / "no_such" / "proc" / "sys"
        out = collect_sysctl(proc_sys_root=str(nonexistent), allowlist=())
        assert out == []

    def test_8kib_read_cap_per_d28(self, tmp_path):
        """D-28 / RESEARCH Q2: 8 KiB defense-in-depth cap on each leaf read.
        Sysfs/procsys are PAGE_SIZE-buffered (~4 KiB) in practice; this
        guards against any future kernel exposing an unbounded blob."""
        import re
        import fnmatch
        from mlpstorage_py.cluster_collector import collect_sysctl
        root = _make_proc_sys_root(tmp_path)
        # Write 16 KiB of "x" — twice the 8 KiB cap.
        _make_sysctl_leaf(root, "vm.dirty_ratio", "x" * 16384)
        allowlist = (re.compile(fnmatch.translate("vm.dirty_*")),)
        out = collect_sysctl(proc_sys_root=root, allowlist=allowlist)
        assert len(out) == 1
        assert len(out[0]["value"]) <= 8192

    def test_dotted_form_conversion(self, tmp_path):
        """D-28: /proc/sys/net/ipv4/tcp_rmem reports as 'net.ipv4.tcp_rmem',
        not 'net/ipv4/tcp_rmem'."""
        import re
        import fnmatch
        from mlpstorage_py.cluster_collector import collect_sysctl
        root = _make_proc_sys_root(tmp_path)
        _make_sysctl_leaf(root, "net.ipv4.tcp_rmem", "4096\t87380\t16777216\n")
        allowlist = (re.compile(fnmatch.translate("net.ipv4.tcp_*")),)
        out = collect_sysctl(proc_sys_root=root, allowlist=allowlist)
        assert len(out) == 1
        assert out[0]["name"] == "net.ipv4.tcp_rmem"
        assert "/" not in out[0]["name"]


class TestSysctlMPIScriptParity:
    """Pattern B (D-36): collect_sysctl and _load_sysctl_allowlist live
    inline in MPI_COLLECTOR_SCRIPT. Drift between the two copies produces
    divergent per-host data shapes; this parity test catches drift on the
    same tmp_path /proc/sys fixture used by TestSysctlCollector."""

    def test_sysctl_functions_match_module(self, tmp_path):
        """Script copy and module copy must agree on the same fixture."""
        import re
        import fnmatch
        from mlpstorage_py.cluster_collector import collect_sysctl
        ns = {}
        try:
            exec(MPI_COLLECTOR_SCRIPT, ns)
        except BaseException:
            # SystemExit / ImportError from the MPI-only top-level; DEFs
            # landed before the raise.
            pass
        assert "collect_sysctl" in ns, (
            "MPI_COLLECTOR_SCRIPT must define collect_sysctl inline (D-36)."
        )
        assert "_load_sysctl_allowlist" in ns, (
            "MPI_COLLECTOR_SCRIPT must define _load_sysctl_allowlist inline (D-36)."
        )
        # Build fixture identical to TestSysctlCollector.
        root = _make_proc_sys_root(tmp_path)
        _make_sysctl_leaf(root, "vm.dirty_ratio", "20\n")
        _make_sysctl_leaf(root, "net.core.rmem_max", "212992\n")
        _make_sysctl_leaf(root, "net.ipv4.tcp_rmem", "4096\t87380\t16777216\n")
        allowlist = tuple(
            re.compile(fnmatch.translate(g))
            for g in ("vm.dirty_*", "net.core.*", "net.ipv4.tcp_*")
        )
        a = ns["collect_sysctl"](root, allowlist)
        b = collect_sysctl(root, allowlist)
        # Order-independent equality (os.walk order isn't promised).
        assert sorted(a, key=lambda d: d["name"]) == sorted(
            b, key=lambda d: d["name"]
        ), (
            f"MPI-script collect_sysctl diverged from module: "
            f"script={a!r} module={b!r}"
        )


class TestSysctlWiring:
    """collect_local_system_info wires sysctl into result via the same
    try/except + default shape as chassis_model and networking."""

    def test_collect_local_system_info_sysctl_wiring(self):
        """Universal-rule contract: result['sysctl'] is always present and
        is a list (possibly empty). The wiring mirrors the chassis_model /
        networking blocks at cluster_collector.py:1145-1167."""
        from mlpstorage_py import cluster_collector as cc
        result = cc.collect_local_system_info()
        assert "sysctl" in result
        assert isinstance(result["sysctl"], list)

    def test_collect_local_system_info_sysctl_uses_collect_sysctl(self, monkeypatch):
        """Wiring contract: result['sysctl'] is exactly what collect_sysctl
        returns on the happy path; no errors key set."""
        from mlpstorage_py import cluster_collector as cc
        monkeypatch.setattr(
            cc, "collect_sysctl",
            lambda *a, **kw: [{"name": "vm.dirty_ratio", "value": "10"}],
        )
        result = cc.collect_local_system_info()
        assert result["sysctl"] == [{"name": "vm.dirty_ratio", "value": "10"}]
        # Either no 'errors' key (deleted when empty) or no 'sysctl' subkey
        # in it.
        assert "sysctl" not in result.get("errors", {})

    def test_collect_local_system_info_sysctl_failure_isolated(self, monkeypatch):
        """Wiring contract: a raise inside collect_sysctl is caught at the
        wiring layer; result['sysctl'] defaults to [] and errors['sysctl']
        captures the message. Mirrors chassis_model/networking try/except."""
        from mlpstorage_py import cluster_collector as cc

        def boom(*a, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(cc, "collect_sysctl", boom)
        result = cc.collect_local_system_info()
        assert result["sysctl"] == []
        assert result.get("errors", {}).get("sysctl") == "boom"


# =============================================================================
# Phase 4 / Plan 04-02 — Environment collector (COLL-06, D-23/24/25/26/36)
# =============================================================================
#
# The environment collector applies a prefix-or-literal allowlist over
# os.environ, dispatches AWS_ACCESS_KEY_ID through `_mask_credential_id` and
# AWS_SECRET_ACCESS_KEY through `_redact_secret`, and emits sorted
# {name, value} dicts ready for `clients[].environment[]` in the YAML.
#
# Pattern B (D-36): The script body inside MPI_COLLECTOR_SCRIPT duplicates
# the collector + both redactors inline (storage_config can't be imported
# from inside the exec'd script).
# =============================================================================


_ENV_ALLOWLIST_VARS_FOR_CLEANUP = (
    # Vars the tests below set/check. Helper clears them between cases so
    # one test's leftover doesn't contaminate the next. Also includes the
    # negative-match anchors PATH/HOME/MY_RANDOM_VAR (we never delete PATH;
    # those are only checked for "not in output", not cleared).
    "BUCKET",
    "BUCKET_NAME",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "STORAGE_BACKEND",
    "OMPI_COMM_WORLD_RANK",
    "UCX_NET_DEVICES",
    "NCCL_DEBUG",
    "MY_RANDOM_VAR",
)


def _clear_env_allowlist_vars(monkeypatch):
    for v in _ENV_ALLOWLIST_VARS_FOR_CLEANUP:
        monkeypatch.delenv(v, raising=False)


class TestEnvAllowlistMatch:
    """`_env_allowlist_match(name)` enforces D-26 (`BUCKET` literal only;
    `AWS_*`, `STORAGE_*`, `OMPI_*`, `UCX_*`, `NCCL_*` prefix matches)."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("BUCKET", True),
            ("BUCKET_NAME", False),  # literal-only per D-26
            ("AWS_ACCESS_KEY_ID", True),
            ("AWS_SECRET_ACCESS_KEY", True),
            ("AWS_REGION", True),
            ("STORAGE_BACKEND", True),
            ("STORAGE_URI_SCHEME", True),
            ("OMPI_COMM_WORLD_RANK", True),
            ("UCX_NET_DEVICES", True),
            ("NCCL_DEBUG", True),
            ("PATH", False),
            ("HOME", False),
            ("LD_LIBRARY_PATH", False),
            ("PYTHONPATH", False),
            ("bucket", False),  # case sensitive
            ("aws_region", False),
        ],
    )
    def test_allowlist_match(self, name, expected):
        from mlpstorage_py.cluster_collector import _env_allowlist_match

        assert _env_allowlist_match(name) is expected


class TestEnvironmentCollector:
    """`collect_environment()` returns sorted {name, value} dicts filtered by
    the D-26 allowlist with credential dispatch through the two helpers."""

    def test_empty_environ_returns_empty(self, monkeypatch):
        """Clear every allowlist-prefixed var → output is empty.

        Production environments may have other AWS_* / STORAGE_* / OMPI_*
        / UCX_* / NCCL_* vars set that we can't enumerate; the universe
        we control is _ENV_ALLOWLIST_VARS_FOR_CLEANUP. We still verify
        the function returns a list and that NONE of our cleared anchor
        vars surface."""
        from mlpstorage_py.cluster_collector import collect_environment

        _clear_env_allowlist_vars(monkeypatch)
        out = collect_environment()
        assert isinstance(out, list)
        cleared = {"BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                   "STORAGE_BACKEND", "OMPI_COMM_WORLD_RANK",
                   "UCX_NET_DEVICES", "NCCL_DEBUG", "AWS_REGION"}
        emitted = {e["name"] for e in out}
        assert cleared.isdisjoint(emitted)

    def test_bucket_literal_emitted(self, monkeypatch):
        """BUCKET literal is in the allowlist (D-26)."""
        from mlpstorage_py.cluster_collector import collect_environment

        _clear_env_allowlist_vars(monkeypatch)
        monkeypatch.setenv("BUCKET", "my-bucket")
        out = collect_environment()
        assert {"name": "BUCKET", "value": "my-bucket"} in out

    def test_bucket_name_not_in_allowlist(self, monkeypatch):
        """BUCKET_NAME has the `BUCKET` substring but is NOT a literal match
        and `BUCKET_` is NOT in the prefix tuple → must be excluded."""
        from mlpstorage_py.cluster_collector import collect_environment

        _clear_env_allowlist_vars(monkeypatch)
        monkeypatch.setenv("BUCKET_NAME", "should-not-appear")
        out = collect_environment()
        names = {e["name"] for e in out}
        assert "BUCKET_NAME" not in names

    def test_aws_access_key_id_masked(self, monkeypatch):
        """AWS_ACCESS_KEY_ID flows through `_mask_credential_id` (D-23)."""
        from mlpstorage_py.cluster_collector import collect_environment

        _clear_env_allowlist_vars(monkeypatch)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        out = collect_environment()
        assert {"name": "AWS_ACCESS_KEY_ID", "value": "AKIA****MPLE"} in out

    def test_aws_secret_access_key_length_only(self, monkeypatch):
        """AWS_SECRET_ACCESS_KEY flows through `_redact_secret` (D-24)."""
        from mlpstorage_py.cluster_collector import collect_environment

        _clear_env_allowlist_vars(monkeypatch)
        monkeypatch.setenv(
            "AWS_SECRET_ACCESS_KEY",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        )
        out = collect_environment()
        assert {
            "name": "AWS_SECRET_ACCESS_KEY",
            "value": "[SET — 40 chars]",
        } in out

    def test_other_aws_vars_verbatim(self, monkeypatch):
        """Non-credential AWS_* vars are emitted verbatim (no redaction)."""
        from mlpstorage_py.cluster_collector import collect_environment

        _clear_env_allowlist_vars(monkeypatch)
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
        out = collect_environment()
        assert {"name": "AWS_REGION", "value": "us-east-1"} in out
        assert {"name": "AWS_DEFAULT_REGION", "value": "us-west-2"} in out

    def test_storage_omp_ucx_nccl_prefixes(self, monkeypatch):
        """One representative var from each non-AWS prefix is emitted verbatim."""
        from mlpstorage_py.cluster_collector import collect_environment

        _clear_env_allowlist_vars(monkeypatch)
        monkeypatch.setenv("STORAGE_BACKEND", "s3dlio")
        monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
        monkeypatch.setenv("UCX_NET_DEVICES", "mlx5_0:1")
        monkeypatch.setenv("NCCL_DEBUG", "INFO")
        out = collect_environment()
        assert {"name": "STORAGE_BACKEND", "value": "s3dlio"} in out
        assert {"name": "OMPI_COMM_WORLD_RANK", "value": "0"} in out
        assert {"name": "UCX_NET_DEVICES", "value": "mlx5_0:1"} in out
        assert {"name": "NCCL_DEBUG", "value": "INFO"} in out

    def test_non_allowlist_excluded(self, monkeypatch):
        """PATH and random non-allowlist vars must not appear in output."""
        from mlpstorage_py.cluster_collector import collect_environment

        _clear_env_allowlist_vars(monkeypatch)
        monkeypatch.setenv("MY_RANDOM_VAR", "x")
        out = collect_environment()
        names = {e["name"] for e in out}
        assert "MY_RANDOM_VAR" not in names
        assert "PATH" not in names
        assert "HOME" not in names

    def test_output_sorted_by_name(self, monkeypatch):
        """D-34 fingerprint stability: output sorted by `name`."""
        from mlpstorage_py.cluster_collector import collect_environment

        _clear_env_allowlist_vars(monkeypatch)
        monkeypatch.setenv("NCCL_DEBUG", "INFO")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        monkeypatch.setenv("BUCKET", "my-bucket")
        monkeypatch.setenv("UCX_NET_DEVICES", "mlx5_0:1")
        out = collect_environment()
        # Filter to just the names we set, in case ambient env has others.
        ours = [e for e in out if e["name"] in {
            "NCCL_DEBUG", "AWS_REGION", "BUCKET", "UCX_NET_DEVICES",
        }]
        names = [e["name"] for e in ours]
        assert names == sorted(names), (
            f"collect_environment output not sorted by name: {names!r}"
        )

    def test_collect_environment_does_not_raise(self, monkeypatch):
        """D-2 envelope: even on a hostile failure, collect_environment
        returns [] rather than raising. Force the inner allowlist check to
        explode and verify."""
        import mlpstorage_py.cluster_collector as cc

        _clear_env_allowlist_vars(monkeypatch)

        def boom(name):
            raise RuntimeError("hostile allowlist failure")

        monkeypatch.setattr(cc, "_env_allowlist_match", boom)
        out = cc.collect_environment()
        assert out == []


class TestEnvironmentRuntimeDenylist:
    """OMPI_ is in the allowlist (correct — captures stable MPI config like
    OMPI_COMM_WORLD_SIZE, OMPI_MCA_btl). But a sub-set of OMPI_ vars
    contain per-run launcher metadata (PIDs, TCP sockets, jobids, crypto
    tokens, session dirs, command-line strings) that change on EVERY
    mpirun invocation. Without a denylist for these, the fingerprint
    captured in systemname.yaml differs from any subsequent re-run's
    fingerprint, so SystemDriftError fires for every legitimate operator
    re-run — making LIFE-04 hand-fill survival unverifiable on real
    runs.

    Surfaced during Phase 5 UAT Test 3 (LIFE-04 hand-fill flow):
    diff hunks at fingerprint positions 282, 296, 302, 318, 320, 324, 328
    showed the on-disk fingerprint differed from the recomputed one in
    OMPI_ARGV, OMPI_FILE_LOCATION, OMPI_MCA_ess_base_jobid,
    OMPI_MCA_orte_hnp_uri, OMPI_MCA_orte_jobfam_session_dir,
    OMPI_MCA_orte_local_daemon_uri, OMPI_MCA_orte_precondition_transports.

    Phase 5.1 (.planning/todos/pending/phase-5.1-env-sysctl-fingerprint-audit.md)
    will broaden this to other launchers (PMI/SLURM/PALS/HYDRA/PBS/LSF).
    """

    _RUNTIME_VOLATILE_OMPI_VARS = [
        "OMPI_ARGV",
        "OMPI_FILE_LOCATION",
        "OMPI_MCA_ess_base_jobid",
        "OMPI_MCA_orte_hnp_uri",
        "OMPI_MCA_orte_jobfam_session_dir",
        "OMPI_MCA_orte_local_daemon_uri",
        "OMPI_MCA_orte_precondition_transports",
    ]

    @pytest.mark.parametrize("varname", _RUNTIME_VOLATILE_OMPI_VARS)
    def test_runtime_volatile_ompi_var_excluded(self, monkeypatch, varname):
        """Each volatile OMPI var must NOT appear in collect_environment output
        even when set in os.environ (which is the live state inside any
        mpirun'd benchmark process)."""
        from mlpstorage_py.cluster_collector import collect_environment

        _clear_env_allowlist_vars(monkeypatch)
        monkeypatch.setenv(varname, "some-per-run-value-12345")
        out = collect_environment()
        names = {e["name"] for e in out}
        assert varname not in names, (
            f"{varname} is a runtime-volatile OMPI launcher variable and must "
            f"be excluded from the fingerprint to prevent spurious "
            f"SystemDriftError on every re-run."
        )

    def test_stable_ompi_vars_still_captured(self, monkeypatch):
        """The denylist must NOT remove the STABLE OMPI vars that describe
        the actual MPI configuration (rank counts, oversubscribe policy,
        binding policy). These ARE part of legitimate client identity."""
        from mlpstorage_py.cluster_collector import collect_environment

        _clear_env_allowlist_vars(monkeypatch)
        # Set both a volatile + a stable var; assert only the stable one
        # survives.
        monkeypatch.setenv("OMPI_MCA_ess_base_jobid", "2108096513")
        monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "8")
        monkeypatch.setenv("OMPI_MCA_mpi_oversubscribe", "0")
        monkeypatch.setenv("OMPI_MCA_hwloc_base_binding_policy", "core")
        out = collect_environment()
        names = {e["name"] for e in out}
        assert "OMPI_MCA_ess_base_jobid" not in names
        assert "OMPI_COMM_WORLD_SIZE" in names
        assert "OMPI_MCA_mpi_oversubscribe" in names
        assert "OMPI_MCA_hwloc_base_binding_policy" in names


class TestEnvironmentMPIScriptParity:
    """Pattern B (D-36): collect_environment + _env_allowlist_match +
    _mask_credential_id + _redact_secret live inline in MPI_COLLECTOR_SCRIPT.
    Drift between the script and the module copy is caught by exec'ing the
    script in a controlled namespace and asserting behavioral equivalence
    on a monkeypatched os.environ."""

    def test_environment_functions_match_module(self, monkeypatch):
        """exec the script body; verify the four symbols landed; assert
        the script's collect_environment matches the module's on the same
        os.environ snapshot."""
        from mlpstorage_py.cluster_collector import collect_environment

        ns = {}
        try:
            exec(MPI_COLLECTOR_SCRIPT, ns)
        except BaseException:
            # SystemExit / ImportError from the MPI-only top-level; DEFs
            # landed before the raise.
            pass
        assert "collect_environment" in ns, (
            "MPI_COLLECTOR_SCRIPT must define collect_environment inline (D-36)."
        )
        assert "_mask_credential_id" in ns, (
            "MPI_COLLECTOR_SCRIPT must define _mask_credential_id inline (D-36)."
        )
        assert "_redact_secret" in ns, (
            "MPI_COLLECTOR_SCRIPT must define _redact_secret inline (D-36)."
        )
        assert "_env_allowlist_match" in ns, (
            "MPI_COLLECTOR_SCRIPT must define _env_allowlist_match inline (D-36)."
        )

        _clear_env_allowlist_vars(monkeypatch)
        # Cover one var from every allowlist prefix + both credential vars +
        # the BUCKET literal so the parity test exercises every dispatch.
        monkeypatch.setenv("BUCKET", "parity-bucket")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        monkeypatch.setenv(
            "AWS_SECRET_ACCESS_KEY",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        )
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        monkeypatch.setenv("STORAGE_BACKEND", "s3dlio")
        monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
        monkeypatch.setenv("UCX_NET_DEVICES", "mlx5_0:1")
        monkeypatch.setenv("NCCL_DEBUG", "INFO")

        a = ns["collect_environment"]()
        b = collect_environment()
        # Both copies must produce the same emit for the same env snapshot.
        assert a == b, (
            f"MPI-script collect_environment diverged from module: "
            f"script={a!r} module={b!r}"
        )


class TestEnvironmentWiring:
    """collect_local_system_info wires `environment` into result via the
    same try/except + default-list shape as chassis_model, networking,
    and sysctl. D-2 universal-rule contract: the key is always present
    and always a list."""

    def test_collect_local_system_info_environment_present(self):
        """Universal-rule contract: result['environment'] is always present
        and is a list (possibly empty)."""
        from mlpstorage_py import cluster_collector as cc

        result = cc.collect_local_system_info()
        assert "environment" in result
        assert isinstance(result["environment"], list)

    def test_collect_local_system_info_environment_uses_collect_environment(
        self, monkeypatch
    ):
        """Wiring contract: result['environment'] is exactly what
        collect_environment returns on the happy path."""
        from mlpstorage_py import cluster_collector as cc

        monkeypatch.setattr(
            cc,
            "collect_environment",
            lambda *a, **kw: [{"name": "BUCKET", "value": "wired-bucket"}],
        )
        result = cc.collect_local_system_info()
        assert result["environment"] == [
            {"name": "BUCKET", "value": "wired-bucket"}
        ]
        assert "environment" not in result.get("errors", {})

    def test_collect_local_system_info_environment_failure_isolated(
        self, monkeypatch
    ):
        """Wiring contract: a raise inside collect_environment is caught at
        the wiring layer; result['environment'] defaults to [] and
        errors['environment'] captures the message."""
        from mlpstorage_py import cluster_collector as cc

        def boom(*a, **kw):
            raise RuntimeError("env boom")

        monkeypatch.setattr(cc, "collect_environment", boom)
        result = cc.collect_local_system_info()
        assert result["environment"] == []
        assert result.get("errors", {}).get("environment") == "env boom"


# =============================================================================
# Phase 4 / Plan 04-03 — Drives collector (COLL-07, D-30/31/32/33/36)
# =============================================================================
#
# The drives collector invokes `lsblk -J -b -d -o NAME,MODEL,VENDOR,SIZE,ROTA,
# TRAN,RM` via subprocess.run, JSON-parses the output, applies the D-31 four-
# rule filter chain (RM=1 skip, virtual-prefix/TRAN skip, unknown-TRAN drop
# with empty-TRAN-nvme-name rescue per RESEARCH Q1), and emits one
# {vendor_name, model_name, interface, capacity_in_GB} dict per surviving row
# per D-30.
#
# D-33 universal-failure rule: lsblk absent / timeout / non-JSON / non-zero
# returncode / empty blockdevices / all-filtered → []. The collector never
# raises.
#
# Pattern B (D-36): collect_drives + _LSBLK_ARGS + the three filter constants
# are all duplicated inline in MPI_COLLECTOR_SCRIPT; TestDrivesMPIScriptParity
# asserts behavioral equivalence under the same monkeypatched subprocess.run.
# =============================================================================


def _lsblk_cp(payload_dict):
    """Build a subprocess.CompletedProcess-shaped mock returning a JSON
    payload as stdout. Matches the production call shape:
    subprocess.run(..., capture_output=True, text=True) → .stdout / .returncode.
    """
    cp = MagicMock()
    cp.stdout = json.dumps(payload_dict)
    cp.returncode = 0
    return cp


def _raises_filenotfound(*args, **kwargs):
    raise FileNotFoundError("lsblk not found")


def _raises_timeout(*args, **kwargs):
    raise subprocess.TimeoutExpired(cmd="lsblk", timeout=10)


class TestDrivesCollector:
    """collect_drives — lsblk -J -b parse + D-31 filter chain.

    Each test monkeypatches mlpstorage_py.cluster_collector.subprocess.run with
    a lambda returning a canned `_lsblk_cp(payload)`. The production code is:
        cp = subprocess.run([...], capture_output=True, text=True, timeout=10)
        if cp.returncode != 0: return []
        payload = json.loads(cp.stdout)
        ... per-row filter ...
    """

    def test_pure_nvme_array_emits_all(self, monkeypatch):
        """Happy path: two NVMe drives, both pass every filter, both emit."""
        from mlpstorage_py import cluster_collector as cc
        payload = {
            "blockdevices": [
                {"name": "nvme0n1", "model": "INTEL SSDPF2NV307TZ",
                 "vendor": "INTEL", "size": "3072000000000",
                 "rota": "0", "tran": "nvme", "rm": "0"},
                {"name": "nvme1n1", "model": "INTEL SSDPF2NV307TZ",
                 "vendor": "INTEL", "size": "3072000000000",
                 "rota": "0", "tran": "nvme", "rm": "0"},
            ]
        }
        monkeypatch.setattr(cc.subprocess, "run",
                            lambda *a, **k: _lsblk_cp(payload))
        out = cc.collect_drives()
        assert out == [
            {"vendor_name": "INTEL", "model_name": "INTEL SSDPF2NV307TZ",
             "interface": "nvme", "capacity_in_GB": 3072},
            {"vendor_name": "INTEL", "model_name": "INTEL SSDPF2NV307TZ",
             "interface": "nvme", "capacity_in_GB": 3072},
        ]

    def test_removable_rm_string_skipped(self, monkeypatch):
        """D-31 rule 1 — util-linux <2.37 emits RM as string ('1'); skip."""
        from mlpstorage_py import cluster_collector as cc
        payload = {
            "blockdevices": [
                {"name": "sda", "model": "WDC", "vendor": "WD",
                 "size": "500000000000", "rota": "1", "tran": "sata",
                 "rm": "0"},
                {"name": "sdb", "model": "Cruzer", "vendor": "SanDisk",
                 "size": "32000000000", "rota": "0", "tran": "usb",
                 "rm": "1"},   # USB stick, RM as string
            ]
        }
        monkeypatch.setattr(cc.subprocess, "run",
                            lambda *a, **k: _lsblk_cp(payload))
        out = cc.collect_drives()
        names = [d["model_name"] for d in out]
        assert "Cruzer" not in names
        assert "WDC" in names

    def test_removable_rm_int_skipped(self, monkeypatch):
        """D-31 rule 1 — util-linux >=2.37 emits RM as int (1); skip.
        Tests the str() coercion handles both string and int variants
        per RESEARCH Q1."""
        from mlpstorage_py import cluster_collector as cc
        payload = {
            "blockdevices": [
                {"name": "sdb", "model": "USB", "vendor": "Generic",
                 "size": "8000000000", "rota": "0", "tran": "usb",
                 "rm": 1},   # int (≥2.37)
            ]
        }
        monkeypatch.setattr(cc.subprocess, "run",
                            lambda *a, **k: _lsblk_cp(payload))
        out = cc.collect_drives()
        assert out == []

    def test_removable_rm_bool_skipped(self, monkeypatch):
        """D-31 rule 1 — newer util-linux JSON output (observed on the
        WSL2 dev shell during Plan 04-03 smoke verification) emits RM as
        a JSON boolean (`true`/`false`). The coercion must skip rows
        where rm is Python True even though str(True) != '1'."""
        from mlpstorage_py import cluster_collector as cc
        payload = {
            "blockdevices": [
                # Surviving accepted-TRAN row: rm=False, must emit.
                {"name": "nvme0n1", "model": "X", "vendor": "Y",
                 "size": "500000000000", "rota": False, "tran": "nvme",
                 "rm": False},
                # Removable row: rm=True, must skip.
                {"name": "sdb", "model": "USB", "vendor": "SanDisk",
                 "size": "32000000000", "rota": False, "tran": "sata",
                 "rm": True},
            ]
        }
        monkeypatch.setattr(cc.subprocess, "run",
                            lambda *a, **k: _lsblk_cp(payload))
        out = cc.collect_drives()
        assert len(out) == 1
        assert out[0]["model_name"] == "X"

    def test_loop_zram_dm_prefixes_skipped(self, monkeypatch):
        """D-31 rule 2 — virtual NAME prefixes {loop, dm-, zram, ram, sr, fd}
        all dropped. One real nvme0n1 interleaved must survive."""
        from mlpstorage_py import cluster_collector as cc
        payload = {
            "blockdevices": [
                {"name": "loop0", "model": "", "vendor": "",
                 "size": "100000000", "rota": "0", "tran": "loop",
                 "rm": "0"},
                {"name": "dm-0", "model": "", "vendor": "",
                 "size": "500000000000", "rota": "0", "tran": "",
                 "rm": "0"},
                {"name": "zram0", "model": "", "vendor": "",
                 "size": "1000000000", "rota": "0", "tran": "zram",
                 "rm": "0"},
                {"name": "ram0", "model": "", "vendor": "",
                 "size": "1000000", "rota": "0", "tran": "",
                 "rm": "0"},
                {"name": "sr0", "model": "DVD-RW", "vendor": "ATAPI",
                 "size": "1000000000", "rota": "1", "tran": "ata",
                 "rm": "0"},
                {"name": "fd0", "model": "", "vendor": "",
                 "size": "1474560", "rota": "1", "tran": "",
                 "rm": "0"},
                {"name": "nvme0n1", "model": "INTEL SSDPF2NV307TZ",
                 "vendor": "INTEL", "size": "3072000000000",
                 "rota": "0", "tran": "nvme", "rm": "0"},
            ]
        }
        monkeypatch.setattr(cc.subprocess, "run",
                            lambda *a, **k: _lsblk_cp(payload))
        out = cc.collect_drives()
        assert len(out) == 1
        assert out[0]["interface"] == "nvme"
        assert out[0]["model_name"] == "INTEL SSDPF2NV307TZ"

    def test_unknown_tran_dropped_not_other(self, monkeypatch):
        """D-31 rule 3 — TRAN not in {nvme, sata, sas} is DROPPED, not mapped
        to 'other'. Diverges from REQUIREMENTS.md COLL-07 wording per
        04-CONTEXT.md §specifics; the DriveInterface.other enum value remains
        in the schema for submitter hand-fills but the collector never emits
        it."""
        from mlpstorage_py import cluster_collector as cc
        payload = {
            "blockdevices": [
                {"name": "sda", "model": "USB-Stick", "vendor": "SanDisk",
                 "size": "32000000000", "rota": "0", "tran": "usb",
                 "rm": "0"},
                {"name": "vda", "model": "Virtio", "vendor": "Red Hat",
                 "size": "500000000000", "rota": "0", "tran": "virtio",
                 "rm": "0"},
            ]
        }
        monkeypatch.setattr(cc.subprocess, "run",
                            lambda *a, **k: _lsblk_cp(payload))
        out = cc.collect_drives()
        assert out == []

    def test_empty_tran_with_nvme_name_rescued(self, monkeypatch):
        """RESEARCH Q1 quirk (a) — older kernels emit TRAN='' for NVMe drives.
        When NAME starts with 'nvme', TRAN is rescued to 'nvme'."""
        from mlpstorage_py import cluster_collector as cc
        payload = {
            "blockdevices": [
                {"name": "nvme0n1", "model": "X", "vendor": "Y",
                 "size": "500000000000", "rota": "0", "tran": "",
                 "rm": "0"},
            ]
        }
        monkeypatch.setattr(cc.subprocess, "run",
                            lambda *a, **k: _lsblk_cp(payload))
        out = cc.collect_drives()
        assert len(out) == 1
        assert out[0]["interface"] == "nvme"

    def test_empty_tran_with_non_nvme_name_dropped(self, monkeypatch):
        """Empty TRAN with non-nvme NAME is dropped (rescue only fires for
        NVMe NAME prefix per D-31 rule 3 + RESEARCH Q1 quirk a)."""
        from mlpstorage_py import cluster_collector as cc
        payload = {
            "blockdevices": [
                {"name": "sda", "model": "X", "vendor": "Y",
                 "size": "500000000000", "rota": "1", "tran": "",
                 "rm": "0"},
            ]
        }
        monkeypatch.setattr(cc.subprocess, "run",
                            lambda *a, **k: _lsblk_cp(payload))
        out = cc.collect_drives()
        assert out == []

    def test_size_decimal_gb_floor(self, monkeypatch):
        """RESEARCH Q1 — capacity_in_GB = int(size) // 10**9 (decimal GB,
        nameplate convention). A 1 TB drive (1_000_204_886_016 bytes) emits
        as 1000, not 1024 binary GiB."""
        from mlpstorage_py import cluster_collector as cc
        payload = {
            "blockdevices": [
                {"name": "sda", "model": "WD",
                 "vendor": "WD", "size": "1000204886016",
                 "rota": "1", "tran": "sata", "rm": "0"},
            ]
        }
        monkeypatch.setattr(cc.subprocess, "run",
                            lambda *a, **k: _lsblk_cp(payload))
        out = cc.collect_drives()
        assert len(out) == 1
        assert out[0]["capacity_in_GB"] == 1000

    def test_vendor_model_stripped(self, monkeypatch):
        """D-30 emit — vendor and model strings are .strip()'d to drop
        lsblk's left/right padding."""
        from mlpstorage_py import cluster_collector as cc
        payload = {
            "blockdevices": [
                {"name": "sda", "model": "  WDC WD5003ABYX-01WERA1  ",
                 "vendor": "  ATA  ", "size": "500000000000",
                 "rota": "1", "tran": "sata", "rm": "0"},
            ]
        }
        monkeypatch.setattr(cc.subprocess, "run",
                            lambda *a, **k: _lsblk_cp(payload))
        out = cc.collect_drives()
        assert len(out) == 1
        assert out[0]["vendor_name"] == "ATA"
        assert out[0]["model_name"] == "WDC WD5003ABYX-01WERA1"

    def test_missing_vendor_emits_empty(self, monkeypatch):
        """D-30 + D-2 — missing 'vendor' key in row emits vendor_name=''
        (defensive .get('vendor') or '' shape)."""
        from mlpstorage_py import cluster_collector as cc
        payload = {
            "blockdevices": [
                {"name": "sda", "model": "X", "size": "500000000000",
                 "rota": "1", "tran": "sata", "rm": "0"},
            ]
        }
        monkeypatch.setattr(cc.subprocess, "run",
                            lambda *a, **k: _lsblk_cp(payload))
        out = cc.collect_drives()
        assert len(out) == 1
        assert out[0]["vendor_name"] == ""
        assert out[0]["model_name"] == "X"

    def test_lsblk_absent_returns_empty(self, monkeypatch):
        """D-33 — lsblk binary absent (busybox/Alpine, container without
        util-linux). FileNotFoundError from subprocess.run → []."""
        from mlpstorage_py import cluster_collector as cc
        monkeypatch.setattr(cc.subprocess, "run", _raises_filenotfound)
        assert cc.collect_drives() == []

    def test_lsblk_timeout_returns_empty(self, monkeypatch):
        """D-33 — lsblk hangs (stuck I/O, dying disk). TimeoutExpired
        from subprocess.run → []."""
        from mlpstorage_py import cluster_collector as cc
        monkeypatch.setattr(cc.subprocess, "run", _raises_timeout)
        assert cc.collect_drives() == []

    def test_lsblk_returns_invalid_json_returns_empty(self, monkeypatch):
        """D-33 — JSON parse failure (lsblk too old to support -J, or
        stdout corruption) → []."""
        from mlpstorage_py import cluster_collector as cc
        cp = MagicMock()
        cp.stdout = "not json"
        cp.returncode = 0
        monkeypatch.setattr(cc.subprocess, "run", lambda *a, **k: cp)
        assert cc.collect_drives() == []

    def test_lsblk_returns_empty_blockdevices(self, monkeypatch):
        """D-33 — lsblk runs cleanly but reports no devices (heavily
        restricted container, all-removable host). Empty blockdevices → []."""
        from mlpstorage_py import cluster_collector as cc
        payload = {"blockdevices": []}
        monkeypatch.setattr(cc.subprocess, "run",
                            lambda *a, **k: _lsblk_cp(payload))
        assert cc.collect_drives() == []

    def test_lsblk_returns_nonzero_exit_returns_empty(self, monkeypatch):
        """D-33 — lsblk exits non-zero (permission, OOM, kernel error).
        returncode != 0 → []."""
        from mlpstorage_py import cluster_collector as cc
        cp = MagicMock()
        cp.stdout = ""
        cp.returncode = 1
        monkeypatch.setattr(cc.subprocess, "run", lambda *a, **k: cp)
        assert cc.collect_drives() == []


class TestDrivesMPIScriptParity:
    """Pattern B (D-36): collect_drives + _LSBLK_ARGS + filter constants are
    duplicated inline in MPI_COLLECTOR_SCRIPT. Drift between the script and
    the module copy produces divergent per-host data shapes; this parity test
    catches drift on the same monkeypatched subprocess.run."""

    def test_drives_functions_match_module(self, monkeypatch):
        """exec the script body; assert collect_drives landed; inject a
        mock subprocess into the script namespace so the script's
        collect_drives sees the same canned payload as the module copy."""
        from mlpstorage_py.cluster_collector import collect_drives
        from mlpstorage_py import cluster_collector as cc

        ns = {}
        try:
            exec(MPI_COLLECTOR_SCRIPT, ns)
        except BaseException:
            # SystemExit / ImportError from the MPI-only top-level; DEFs
            # landed before the raise.
            pass
        assert "collect_drives" in ns, (
            "MPI_COLLECTOR_SCRIPT must define collect_drives inline (D-36)."
        )
        assert "_LSBLK_ARGS" in ns, (
            "MPI_COLLECTOR_SCRIPT must define _LSBLK_ARGS inline (D-36)."
        )

        payload = {
            "blockdevices": [
                {"name": "nvme0n1", "model": "Samsung SSD 980 PRO",
                 "vendor": "Samsung", "size": "1000204886016",
                 "rota": "0", "tran": "nvme", "rm": "0"},
                {"name": "sda", "model": "WDC WD5003ABYX-01WERA1",
                 "vendor": "ATA", "size": "500107862016",
                 "rota": "1", "tran": "sata", "rm": "0"},
                # Skipped: removable
                {"name": "sdb", "model": "Cruzer", "vendor": "SanDisk",
                 "size": "32000000000", "rota": "0", "tran": "usb",
                 "rm": "1"},
                # Skipped: virtual prefix
                {"name": "loop0", "model": "", "vendor": "",
                 "size": "1000000", "rota": "0", "tran": "loop",
                 "rm": "0"},
            ]
        }

        # Inject a mock subprocess module into the script namespace AND
        # patch the module-side subprocess.run. The script's collect_drives
        # body references the bare `subprocess` name (Pattern B duplicates
        # the import inside the script body), so we replace the symbol the
        # script body resolves at call time.
        mock_subprocess = MagicMock()
        mock_subprocess.run = lambda *a, **k: _lsblk_cp(payload)
        # Preserve real TimeoutExpired/SubprocessError so script-side except
        # tuples that reference these still type-check (the script can't
        # import subprocess if we replaced the whole module without keeping
        # the exception types).
        mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired
        mock_subprocess.SubprocessError = subprocess.SubprocessError
        ns["subprocess"] = mock_subprocess

        monkeypatch.setattr(cc.subprocess, "run",
                            lambda *a, **k: _lsblk_cp(payload))

        a = ns["collect_drives"]()
        b = collect_drives()
        assert a == b, (
            f"MPI-script collect_drives diverged from module: "
            f"script={a!r} module={b!r}"
        )


class TestDrivesWiring:
    """collect_local_system_info wires `drives` into result via the same
    try/except + default-list shape as chassis_model, networking, sysctl, and
    environment. D-2 universal-rule contract: the key is always present and
    always a list (the D-33 omit-when-empty behavior fires at the
    auto_generator transform layer, not at the collector wiring)."""

    def test_collect_local_system_info_drives_present(self):
        """Universal-rule contract: result['drives'] is always present and
        is a list (possibly empty). Smokes the WSL2 dev shell where lsblk
        may or may not be installed."""
        from mlpstorage_py import cluster_collector as cc

        result = cc.collect_local_system_info()
        assert "drives" in result
        assert isinstance(result["drives"], list)

    def test_collect_local_system_info_drives_uses_collect_drives(
        self, monkeypatch
    ):
        """Wiring contract: result['drives'] is exactly what collect_drives
        returns on the happy path."""
        from mlpstorage_py import cluster_collector as cc

        monkeypatch.setattr(
            cc,
            "collect_drives",
            lambda *a, **kw: [
                {"vendor_name": "INTEL", "model_name": "X",
                 "interface": "nvme", "capacity_in_GB": 3072}
            ],
        )
        result = cc.collect_local_system_info()
        assert result["drives"] == [
            {"vendor_name": "INTEL", "model_name": "X",
             "interface": "nvme", "capacity_in_GB": 3072}
        ]
        assert "drives" not in result.get("errors", {})

    def test_collect_local_system_info_drives_failure_isolated(
        self, monkeypatch
    ):
        """Wiring contract: a raise inside collect_drives is caught at the
        wiring layer; result['drives'] defaults to [] and errors['drives']
        captures the message."""
        from mlpstorage_py import cluster_collector as cc

        def boom(*a, **kw):
            raise RuntimeError("drives boom")

        monkeypatch.setattr(cc, "collect_drives", boom)
        result = cc.collect_local_system_info()
        assert result["drives"] == []
        assert result.get("errors", {}).get("drives") == "drives boom"


class TestSharedFsProbeNonRank0Silence:
    """HARDEN-02 D-55.3 structural defense: only rank 0 writes to stdout.

    The CAP-02 launcher's stdout-marker parsing (D-54/D-55) assumes
    non-rank-0 ranks are silent on stdout. A future probe-script edit
    that adds `print()` on other ranks without a `if rank == 0:` guard
    would corrupt the parsed JSON without this test. The class is
    parametrized over rank 0 (must emit markers) AND ranks 1/2/3 (must
    be silent) so the rank-0-only contract is locked from both sides.

    Mechanism: stub mpi4py at module level so the probe heredoc's
    `from mpi4py import MPI` succeeds inside exec'd namespace; mock
    MPI.COMM_WORLD.Get_rank() to return the parametrized rank; capture
    sys.stdout via contextlib.redirect_stdout; assert the per-rank
    contract on captured.getvalue().

    Today (pre-HARDEN-02, file-based transport): rank=0 FAILS this test
    because the current probe writes JSON to a file, not stdout — no
    markers ever appear on stdout. This is the RED proof. Non-rank-0
    ranks already pass GREEN today (they only write to stderr/files),
    so the silence side of the test serves as the GREEN-state structural
    defense once Task 2 lands the stdout transport.
    """

    @pytest.mark.parametrize("rank", [0, 1, 2, 3])
    def test_rank0_emits_markers_and_non_rank0_silent(
        self, tmp_path, monkeypatch, rank
    ):
        """rank 0 MUST emit __CAP02_RESULT_BEGIN__/END markers on stdout;
        ranks 1, 2, 3 MUST emit zero bytes to stdout."""
        import contextlib
        import io
        import re
        import sys
        from unittest.mock import MagicMock

        # Stub mpi4py: the probe heredoc does `from mpi4py import MPI`.
        fake_comm = MagicMock()
        fake_comm.Get_rank.return_value = rank
        fake_comm.Get_size.return_value = 4
        # gather: root receives a list of payloads; non-root receives None.
        if rank == 0:
            # Build a successful all_payloads with cardinality 1.
            fake_comm.gather.return_value = [
                {"hostname": "h0", "rank": 0, "failure": None, "st_dev": 100, "st_ino": 200},
                {"hostname": "h1", "rank": 1, "failure": None, "st_dev": 100, "st_ino": 200},
                {"hostname": "h2", "rank": 2, "failure": None, "st_dev": 100, "st_ino": 200},
                {"hostname": "h3", "rank": 3, "failure": None, "st_dev": 100, "st_ino": 200},
            ]
        else:
            fake_comm.gather.return_value = None
        fake_comm.bcast.return_value = "ok"
        fake_comm.Barrier.return_value = None

        fake_mpi_module = MagicMock()
        fake_mpi_module.COMM_WORLD = fake_comm

        # mpi4py is a package — provide both the top-level and the MPI submodule.
        fake_mpi4py_pkg = MagicMock()
        fake_mpi4py_pkg.MPI = fake_mpi_module
        monkeypatch.setitem(sys.modules, "mpi4py", fake_mpi4py_pkg)
        monkeypatch.setitem(sys.modules, "mpi4py.MPI", fake_mpi_module)

        # NEW 2-positional argv signature per D-54: data_dir, run_uuid (no output_file).
        monkeypatch.setattr(
            sys, "argv",
            ["probe", str(tmp_path), "silence-test-uuid"],
        )

        from mlpstorage_py.cluster_collector import SHARED_FS_PROBE_SCRIPT

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            try:
                exec(SHARED_FS_PROBE_SCRIPT, {"__name__": "__main__"})
            except SystemExit:
                pass  # probe calls sys.exit(0) on ok, sys.exit(1) on fail.
            except Exception:
                # Any other exception is OK on this code path — we are only
                # locking the stdout contract; per-rank failure modes are
                # exercised in tests/unit/test_shared_fs_probe.py.
                pass

        stdout_content = captured.getvalue()

        if rank == 0:
            assert "__CAP02_RESULT_BEGIN__" in stdout_content, (
                f"rank 0 MUST emit __CAP02_RESULT_BEGIN__ on stdout (HARDEN-02 D-54). "
                f"Got: {stdout_content!r}"
            )
            assert "__CAP02_RESULT_END__" in stdout_content, (
                f"rank 0 MUST emit __CAP02_RESULT_END__ on stdout (HARDEN-02 D-54). "
                f"Got: {stdout_content!r}"
            )
            # Verify the framed payload parses as JSON.
            m = re.search(
                r"__CAP02_RESULT_BEGIN__\s*\n(.*?)\n.*?__CAP02_RESULT_END__",
                stdout_content,
                re.DOTALL,
            )
            assert m is not None, (
                f"rank 0 markers present but no payload extractable. Got: {stdout_content!r}"
            )
            payload = m.group(1).strip()
            json.loads(payload)  # raises if invalid
        else:
            assert stdout_content == "", (
                f"rank {rank} emitted {len(stdout_content)} stdout bytes; "
                f"expected zero (HARDEN-02 D-55.3). Content: {stdout_content!r}"
            )


class TestTagOutputRegexParsesOpenMpi4xPrefix:
    """HARDEN-04 regression guard: the CAP-02 launcher's tag-strip regex
    must consume the OpenMPI 4.x --tag-output prefix format
    [rank,jobid]<channel>: (verified on OpenMPI 4.1.6 per the debug
    session at .planning/debug/cap02-stdout-empty-payload-tag-output-multihost.md).

    Today (pre-HARDEN-04) FAILS RED: the regex r'^\\[[^\\]]+\\]\\s*' was
    written assuming OpenMPI emits [host:rank] (space-separated) prefixes.
    The actual OpenMPI 4.x format is [rank,jobid]<channel>: with no
    whitespace between the bracketed identifier and the channel marker.
    The current regex strips [1,0] but leaves <stdout>: glued in front
    of the JSON; json.loads('<stdout>:{...}') raises JSONDecodeError.

    This test locks the contract WITHOUT requiring mpirun — the fixture
    is the literal repr() output captured from a real OpenMPI 4.1.6
    subprocess.run during the Phase 5.1 UAT (timestamp 2026-06-24T18:24:00Z
    in the debug session). Even when mpirun is absent (CI env, locked-down
    container), this regression is caught.
    """

    # Literal byte-string captured from `mpirun -n 2 --allow-run-as-root
    # --tag-output --host 127.0.0.1:1,127.0.0.1:1 .venv/bin/python <probe>`
    # on OpenMPI 4.1.6 + mpi4py 4.1.2 dev box (debug session).
    _OPENMPI_4X_STDOUT_FIXTURE = (
        '[1,0]<stdout>:__CAP02_RESULT_BEGIN__\n'
        '[1,0]<stdout>:{"status":"ok","ranks":[{"hostname":"h0","rank":0,'
        '"failure":null,"st_dev":2096,"st_ino":148891},{"hostname":"h1",'
        '"rank":1,"failure":null,"st_dev":2096,"st_ino":148891}],'
        '"failure_summary":null,"unlink_warning":null}\n'
        '[1,0]<stdout>:__CAP02_RESULT_END__\n'
    )

    # Backward-compat legacy format (what the original regex assumed).
    _LEGACY_HOST_RANK_FIXTURE = (
        '[host:0] __CAP02_RESULT_BEGIN__\n'
        '[host:0] {"status":"ok","ranks":[],'
        '"failure_summary":null,"unlink_warning":null}\n'
        '[host:0] __CAP02_RESULT_END__\n'
    )

    def test_openmpi_4x_tag_output_format_parses(self):
        """HARDEN-04 primary RED: OpenMPI 4.x [rank,jobid]<channel>: prefix
        must be fully stripped so json.loads succeeds."""
        import json
        import re

        # Step 1: marker regex extracts payload (mirrors launcher line 3515).
        marker_re = re.compile(
            r"__CAP02_RESULT_BEGIN__\s*\n(?P<payload>.*?)\n.*?__CAP02_RESULT_END__",
            re.DOTALL,
        )
        m = marker_re.search(self._OPENMPI_4X_STDOUT_FIXTURE)
        assert m is not None, "marker regex must find payload in OpenMPI 4.x fixture"

        # Step 2: tag-strip the payload via the production helper
        # (HARDEN-04 REFACTOR: tests consume the production single source
        # of truth, not a duplicated regex pattern).
        payload_raw = m.group("payload").strip()
        stripped = _strip_tag_output_prefix(payload_raw)

        # Step 3: json.loads must succeed (this is what fails RED in production today).
        parsed = json.loads(stripped)
        assert parsed.get("status") == "ok", (
            f"expected status='ok', got {parsed!r}"
        )

        # Step 4: independently verify the OLD regex fails on this fixture,
        # proving the test fixture captures the actual regression shape.
        old_stripped = re.sub(r"^\[[^\]]+\]\s*", "", payload_raw)
        assert old_stripped.startswith("<stdout>:"), (
            f"old regex must leave <stdout>: residual to prove HARDEN-04 regression "
            f"shape; got {old_stripped!r}"
        )
        with pytest.raises(json.JSONDecodeError):
            json.loads(old_stripped)

    @pytest.mark.parametrize("fixture_name,fixture", [
        ("openmpi_4x", _OPENMPI_4X_STDOUT_FIXTURE),
        ("legacy_host_rank", _LEGACY_HOST_RANK_FIXTURE),
    ])
    def test_tag_strip_handles_both_formats(self, fixture_name, fixture):
        """HARDEN-04 backward-compat lock: the new regex must handle BOTH
        the OpenMPI 4.x [rank,jobid]<channel>: format AND the assumed
        legacy [host:rank] format. Today (a) fails RED; (b) passes."""
        import json
        import re

        marker_re = re.compile(
            r"__CAP02_RESULT_BEGIN__\s*\n(?P<payload>.*?)\n.*?__CAP02_RESULT_END__",
            re.DOTALL,
        )
        m = marker_re.search(fixture)
        assert m is not None, f"{fixture_name}: marker regex must find payload"

        payload_raw = m.group("payload").strip()
        # HARDEN-04 REFACTOR: call production helper, not duplicated regex.
        stripped = _strip_tag_output_prefix(payload_raw)
        parsed = json.loads(stripped)
        assert parsed.get("status") == "ok"
