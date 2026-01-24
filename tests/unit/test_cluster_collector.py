"""Unit tests for cluster_collector module."""

import pytest

from mlpstorage.cluster_collector import (
    parse_proc_vmstat,
    parse_proc_mounts,
    parse_proc_cgroups,
    MountInfo,
    CgroupInfo,
)


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
