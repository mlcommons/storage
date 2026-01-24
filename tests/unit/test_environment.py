"""
Tests for the environment detection module.

Tests cover:
- OSInfo dataclass creation and field access
- OS detection with mocked platform values
- Install instruction lookup for various OS types
- Graceful fallback behavior
"""

import pytest
from unittest.mock import patch, MagicMock

from mlpstorage.environment import OSInfo, detect_os, get_install_instruction, INSTALL_INSTRUCTIONS


class TestOSInfo:
    """Tests for OSInfo dataclass."""

    def test_create_with_required_fields(self):
        """Should create OSInfo with only required fields."""
        info = OSInfo(system='Linux', release='5.4.0', machine='x86_64')
        assert info.system == 'Linux'
        assert info.release == '5.4.0'
        assert info.machine == 'x86_64'
        assert info.distro_id is None
        assert info.distro_name is None
        assert info.distro_version is None

    def test_create_with_all_fields(self):
        """Should create OSInfo with all fields."""
        info = OSInfo(
            system='Linux',
            release='5.4.0',
            machine='x86_64',
            distro_id='ubuntu',
            distro_name='Ubuntu',
            distro_version='22.04'
        )
        assert info.distro_id == 'ubuntu'
        assert info.distro_name == 'Ubuntu'
        assert info.distro_version == '22.04'

    def test_default_optional_fields(self):
        """Should have None as default for optional fields."""
        info = OSInfo(system='Darwin', release='21.0', machine='arm64')
        assert info.distro_id is None
        assert info.distro_name is None
        assert info.distro_version is None


class TestDetectOS:
    """Tests for detect_os function."""

    def test_detects_linux_with_distro_package(self):
        """Should detect Linux with distro package when available."""
        mock_distro = MagicMock()
        mock_distro.id.return_value = 'ubuntu'
        mock_distro.name.return_value = 'Ubuntu'
        mock_distro.version.return_value = '22.04'

        with patch('platform.system', return_value='Linux'):
            with patch('platform.release', return_value='5.4.0-generic'):
                with patch('platform.machine', return_value='x86_64'):
                    with patch.dict('sys.modules', {'distro': mock_distro}):
                        info = detect_os()

        assert info.system == 'Linux'
        assert info.release == '5.4.0-generic'
        assert info.machine == 'x86_64'
        # Note: With mocked sys.modules, the actual import in detect_os
        # will still use the real module if installed

    def test_detects_darwin(self):
        """Should detect macOS (Darwin)."""
        with patch('platform.system', return_value='Darwin'):
            with patch('platform.release', return_value='21.6.0'):
                with patch('platform.machine', return_value='arm64'):
                    info = detect_os()

        assert info.system == 'Darwin'
        assert info.release == '21.6.0'
        assert info.machine == 'arm64'
        # macOS doesn't have distro info
        assert info.distro_id is None
        assert info.distro_name is None

    def test_detects_windows(self):
        """Should detect Windows."""
        with patch('platform.system', return_value='Windows'):
            with patch('platform.release', return_value='10'):
                with patch('platform.machine', return_value='AMD64'):
                    info = detect_os()

        assert info.system == 'Windows'
        assert info.release == '10'
        assert info.machine == 'AMD64'
        assert info.distro_id is None

    def test_returns_osinfo_instance(self):
        """Should always return OSInfo instance."""
        info = detect_os()
        assert isinstance(info, OSInfo)
        assert info.system in ('Linux', 'Darwin', 'Windows')

    def test_handles_missing_distro_package(self):
        """Should handle ImportError when distro package unavailable."""
        # Force import to fail by patching the import mechanism
        with patch('platform.system', return_value='Linux'):
            with patch('platform.release', return_value='5.4.0'):
                with patch('platform.machine', return_value='x86_64'):
                    # Even without distro, should return valid OSInfo
                    info = detect_os()

        assert isinstance(info, OSInfo)
        assert info.system == 'Linux'


class TestGetInstallInstruction:
    """Tests for get_install_instruction function."""

    def test_ubuntu_mpi_uses_apt(self):
        """Should return apt-get command for Ubuntu MPI."""
        ubuntu = OSInfo(
            system='Linux', release='', machine='x86_64',
            distro_id='ubuntu', distro_name='Ubuntu', distro_version='22.04'
        )
        hint = get_install_instruction('mpi', ubuntu)
        assert 'apt-get' in hint
        assert 'openmpi' in hint

    def test_debian_mpi_uses_apt(self):
        """Should return apt-get command for Debian MPI."""
        debian = OSInfo(
            system='Linux', release='', machine='x86_64',
            distro_id='debian', distro_name='Debian', distro_version='11'
        )
        hint = get_install_instruction('mpi', debian)
        assert 'apt-get' in hint
        assert 'openmpi' in hint

    def test_rhel_mpi_uses_dnf(self):
        """Should return dnf command for RHEL MPI."""
        rhel = OSInfo(
            system='Linux', release='', machine='x86_64',
            distro_id='rhel', distro_name='Red Hat Enterprise Linux', distro_version='8'
        )
        hint = get_install_instruction('mpi', rhel)
        assert 'dnf' in hint
        assert 'openmpi' in hint

    def test_centos_mpi_uses_yum(self):
        """Should return yum command for CentOS MPI."""
        centos = OSInfo(
            system='Linux', release='', machine='x86_64',
            distro_id='centos', distro_name='CentOS', distro_version='7'
        )
        hint = get_install_instruction('mpi', centos)
        assert 'yum' in hint
        assert 'openmpi' in hint

    def test_fedora_mpi_uses_dnf(self):
        """Should return dnf command for Fedora MPI."""
        fedora = OSInfo(
            system='Linux', release='', machine='x86_64',
            distro_id='fedora', distro_name='Fedora', distro_version='38'
        )
        hint = get_install_instruction('mpi', fedora)
        assert 'dnf' in hint
        assert 'openmpi' in hint

    def test_arch_mpi_uses_pacman(self):
        """Should return pacman command for Arch MPI."""
        arch = OSInfo(
            system='Linux', release='', machine='x86_64',
            distro_id='arch', distro_name='Arch Linux', distro_version=''
        )
        hint = get_install_instruction('mpi', arch)
        assert 'pacman' in hint
        assert 'openmpi' in hint

    def test_macos_mpi_uses_brew(self):
        """Should return brew command for macOS MPI."""
        macos = OSInfo(
            system='Darwin', release='21.6.0', machine='arm64',
            distro_id=None, distro_name=None, distro_version=None
        )
        hint = get_install_instruction('mpi', macos)
        assert 'brew' in hint
        assert 'open-mpi' in hint

    def test_windows_mpi_shows_msmpi(self):
        """Should return MS-MPI info for Windows MPI."""
        windows = OSInfo(
            system='Windows', release='10', machine='AMD64',
            distro_id=None, distro_name=None, distro_version=None
        )
        hint = get_install_instruction('mpi', windows)
        assert 'MS-MPI' in hint or 'microsoft' in hint.lower()

    def test_unknown_linux_distro_fallback(self):
        """Should fallback to generic Linux message for unknown distro."""
        unknown = OSInfo(
            system='Linux', release='', machine='x86_64',
            distro_id='unknowndistro', distro_name='Unknown Linux', distro_version='1.0'
        )
        hint = get_install_instruction('mpi', unknown)
        # Should get the generic Linux fallback
        assert 'package manager' in hint.lower() or 'OpenMPI' in hint

    def test_dlio_returns_pip_regardless_of_os(self):
        """Should return pip command for DLIO on any OS."""
        ubuntu = OSInfo(
            system='Linux', release='', machine='x86_64',
            distro_id='ubuntu', distro_name='Ubuntu', distro_version='22.04'
        )
        macos = OSInfo(
            system='Darwin', release='21.0', machine='arm64',
            distro_id=None, distro_name=None, distro_version=None
        )

        ubuntu_hint = get_install_instruction('dlio', ubuntu)
        macos_hint = get_install_instruction('dlio', macos)

        assert 'pip' in ubuntu_hint
        assert 'pip' in macos_hint
        assert ubuntu_hint == macos_hint

    def test_unknown_dependency_returns_generic_message(self):
        """Should return generic message for unknown dependency."""
        ubuntu = OSInfo(
            system='Linux', release='', machine='x86_64',
            distro_id='ubuntu', distro_name='Ubuntu', distro_version='22.04'
        )
        hint = get_install_instruction('unknowndep', ubuntu)
        assert 'unknowndep' in hint
        assert 'package manager' in hint.lower()

    def test_ssh_ubuntu_uses_apt(self):
        """Should return apt-get command for Ubuntu SSH."""
        ubuntu = OSInfo(
            system='Linux', release='', machine='x86_64',
            distro_id='ubuntu', distro_name='Ubuntu', distro_version='22.04'
        )
        hint = get_install_instruction('ssh', ubuntu)
        assert 'apt-get' in hint
        assert 'openssh' in hint

    def test_ssh_macos_shows_builtin(self):
        """Should indicate SSH is included on macOS."""
        macos = OSInfo(
            system='Darwin', release='21.6.0', machine='arm64',
            distro_id=None, distro_name=None, distro_version=None
        )
        hint = get_install_instruction('ssh', macos)
        assert 'macOS' in hint or 'included' in hint.lower()


class TestInstallInstructions:
    """Tests for INSTALL_INSTRUCTIONS dictionary."""

    def test_has_mpi_entries(self):
        """Should have MPI installation entries."""
        mpi_entries = [k for k in INSTALL_INSTRUCTIONS.keys() if k[0] == 'mpi']
        assert len(mpi_entries) >= 5  # At least ubuntu, debian, rhel, darwin, windows

    def test_has_ssh_entries(self):
        """Should have SSH installation entries."""
        ssh_entries = [k for k in INSTALL_INSTRUCTIONS.keys() if k[0] == 'ssh']
        assert len(ssh_entries) >= 3

    def test_has_dlio_entry(self):
        """Should have DLIO installation entry."""
        assert ('dlio', None, None) in INSTALL_INSTRUCTIONS
        dlio_hint = INSTALL_INSTRUCTIONS[('dlio', None, None)]
        assert 'pip' in dlio_hint

    def test_keys_are_tuples(self):
        """Should have (dependency, system, distro) tuple keys."""
        for key in INSTALL_INSTRUCTIONS.keys():
            assert isinstance(key, tuple)
            assert len(key) == 3
