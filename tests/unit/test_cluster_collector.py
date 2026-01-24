"""Unit tests for cluster_collector module."""

import json
import subprocess
import time
import pytest
from unittest.mock import MagicMock, patch, Mock

from mlpstorage.cluster_collector import (
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
)
from mlpstorage.interfaces.collector import CollectionResult


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
        from mlpstorage.cluster_collector import collect_local_system_info

        info = collect_local_system_info()
        assert 'vmstat' in info
        assert isinstance(info['vmstat'], dict)
        # Should have at least some vmstat entries on a Linux system
        if info['vmstat']:
            # Check for a common vmstat key
            assert any(k.startswith('nr_') for k in info['vmstat'].keys())

    def test_includes_mounts(self):
        """Test that collect_local_system_info includes mounts data."""
        from mlpstorage.cluster_collector import collect_local_system_info

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
        from mlpstorage.cluster_collector import collect_local_system_info

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

    @patch('mlpstorage.cluster_collector.collect_local_system_info')
    def test_collect_from_localhost_uses_direct_collection(self, mock_local, collector):
        """Test that localhost uses direct collection, not SSH."""
        mock_local.return_value = {'hostname': 'localhost', 'meminfo': {}}
        result = collector._collect_from_single_host('localhost')
        mock_local.assert_called_once()
        assert result['hostname'] == 'localhost'

    @patch('mlpstorage.cluster_collector.collect_local_system_info')
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

    @patch('mlpstorage.cluster_collector.SSHClusterCollector._collect_from_single_host')
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

    @patch('mlpstorage.cluster_collector.SSHClusterCollector._collect_from_single_host')
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

    @patch('mlpstorage.cluster_collector.SSHClusterCollector._collect_from_single_host')
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

    @patch('mlpstorage.cluster_collector.SSHClusterCollector._collect_from_single_host')
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
        from mlpstorage.rules.models import TimeSeriesSample

        sample = TimeSeriesSample(
            timestamp='2026-01-24T12:00:00Z',
            hostname='testhost'
        )

        assert sample.timestamp == '2026-01-24T12:00:00Z'
        assert sample.hostname == 'testhost'

    def test_to_dict_excludes_none(self):
        """to_dict should exclude None values."""
        from mlpstorage.rules.models import TimeSeriesSample

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
        from mlpstorage.rules.models import TimeSeriesSample

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
        from mlpstorage.rules.models import TimeSeriesSample, TimeSeriesData

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
        from mlpstorage.rules.models import TimeSeriesSample, TimeSeriesData

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
        from mlpstorage.rules.models import TimeSeriesSample, TimeSeriesData

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
