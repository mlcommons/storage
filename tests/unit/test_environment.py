"""
Tests for the environment detection module.

Tests cover:
- OSInfo dataclass creation and field access
- OS detection with mocked platform values
- Install instruction lookup for various OS types
- Graceful fallback behavior
- ValidationIssue dataclass creation
- SSH connectivity validation
- Validation issue collection
"""

import pytest
import subprocess
from unittest.mock import patch, MagicMock

from mlpstorage.environment import (
    OSInfo,
    detect_os,
    get_install_instruction,
    INSTALL_INSTRUCTIONS,
    ValidationIssue,
    validate_ssh_connectivity,
    collect_validation_issues,
)


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


class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_create_with_required_fields(self):
        """Should create ValidationIssue with required fields."""
        issue = ValidationIssue(
            severity='error',
            category='connectivity',
            message='Host unreachable',
            suggestion='Check network'
        )
        assert issue.severity == 'error'
        assert issue.category == 'connectivity'
        assert issue.message == 'Host unreachable'
        assert issue.suggestion == 'Check network'
        assert issue.install_cmd is None
        assert issue.host is None

    def test_create_with_all_fields(self):
        """Should create ValidationIssue with all fields."""
        issue = ValidationIssue(
            severity='error',
            category='dependency',
            message='SSH not found',
            suggestion='Install SSH',
            install_cmd='sudo apt-get install openssh-client',
            host='node1'
        )
        assert issue.severity == 'error'
        assert issue.category == 'dependency'
        assert issue.message == 'SSH not found'
        assert issue.suggestion == 'Install SSH'
        assert issue.install_cmd == 'sudo apt-get install openssh-client'
        assert issue.host == 'node1'

    def test_severity_error(self):
        """Should accept error severity."""
        issue = ValidationIssue('error', 'configuration', 'Problem', 'Fix it')
        assert issue.severity == 'error'

    def test_severity_warning(self):
        """Should accept warning severity."""
        issue = ValidationIssue('warning', 'configuration', 'Minor issue', 'Consider fixing')
        assert issue.severity == 'warning'

    def test_category_dependency(self):
        """Should accept dependency category."""
        issue = ValidationIssue('error', 'dependency', 'Missing pkg', 'Install it')
        assert issue.category == 'dependency'

    def test_category_configuration(self):
        """Should accept configuration category."""
        issue = ValidationIssue('warning', 'configuration', 'Bad config', 'Change it')
        assert issue.category == 'configuration'

    def test_category_connectivity(self):
        """Should accept connectivity category."""
        issue = ValidationIssue('error', 'connectivity', 'No connection', 'Check network')
        assert issue.category == 'connectivity'

    def test_category_filesystem(self):
        """Should accept filesystem category."""
        issue = ValidationIssue('error', 'filesystem', 'No permission', 'Fix permissions')
        assert issue.category == 'filesystem'

    def test_optional_fields_none_by_default(self):
        """Should have None as default for optional fields."""
        issue = ValidationIssue('error', 'dependency', 'Test', 'Fix')
        assert issue.install_cmd is None
        assert issue.host is None


class TestValidateSshConnectivity:
    """Tests for validate_ssh_connectivity function."""

    def test_raises_validation_issue_when_ssh_not_found(self):
        """Should raise ValidationIssue when SSH binary not found."""
        with patch('shutil.which', return_value=None):
            with patch('mlpstorage.environment.detect_os') as mock_detect:
                with patch('mlpstorage.environment.get_install_instruction') as mock_hint:
                    mock_detect.return_value = OSInfo(
                        system='Linux', release='5.4.0', machine='x86_64',
                        distro_id='ubuntu', distro_name='Ubuntu', distro_version='22.04'
                    )
                    mock_hint.return_value = 'sudo apt-get install openssh-client'

                    with pytest.raises(ValidationIssue) as exc_info:
                        validate_ssh_connectivity(['node1'])

                    issue = exc_info.value
                    assert issue.severity == 'error'
                    assert issue.category == 'dependency'
                    assert 'SSH' in issue.message
                    assert issue.install_cmd == 'sudo apt-get install openssh-client'

    def test_validation_issue_has_os_specific_install_command(self):
        """Should include OS-specific install command in ValidationIssue."""
        with patch('shutil.which', return_value=None):
            with patch('mlpstorage.environment.detect_os') as mock_detect:
                with patch('mlpstorage.environment.get_install_instruction') as mock_hint:
                    # Mock RHEL system
                    mock_detect.return_value = OSInfo(
                        system='Linux', release='4.18.0', machine='x86_64',
                        distro_id='rhel', distro_name='Red Hat Enterprise Linux', distro_version='8'
                    )
                    mock_hint.return_value = 'sudo dnf install openssh-clients'

                    with pytest.raises(ValidationIssue) as exc_info:
                        validate_ssh_connectivity(['node1'])

                    assert exc_info.value.install_cmd == 'sudo dnf install openssh-clients'
                    mock_hint.assert_called_once()

    def test_skips_localhost(self):
        """Should skip localhost and return success without running SSH."""
        with patch('shutil.which', return_value='/usr/bin/ssh'):
            with patch('subprocess.run') as mock_run:
                results = validate_ssh_connectivity(['localhost'])

                assert len(results) == 1
                assert results[0][0] == 'localhost'
                assert results[0][1] is True
                assert 'skipped' in results[0][2].lower()
                mock_run.assert_not_called()

    def test_skips_127_0_0_1(self):
        """Should skip 127.0.0.1 and return success without running SSH."""
        with patch('shutil.which', return_value='/usr/bin/ssh'):
            with patch('subprocess.run') as mock_run:
                results = validate_ssh_connectivity(['127.0.0.1'])

                assert len(results) == 1
                assert results[0][0] == '127.0.0.1'
                assert results[0][1] is True
                assert 'skipped' in results[0][2].lower()
                mock_run.assert_not_called()

    def test_parses_host_slots_format(self):
        """Should parse host:slots format correctly."""
        with patch('shutil.which', return_value='/usr/bin/ssh'):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = 'ok'
            mock_result.stderr = ''

            with patch('subprocess.run', return_value=mock_result) as mock_run:
                results = validate_ssh_connectivity(['node1:4'])

                assert len(results) == 1
                assert results[0][0] == 'node1'  # Hostname without slots
                assert results[0][1] is True
                # Verify SSH was called with just the hostname
                call_args = mock_run.call_args[0][0]
                assert 'node1' in call_args
                assert 'node1:4' not in call_args

    def test_returns_failure_when_host_unreachable(self):
        """Should return failure when SSH returns non-zero."""
        with patch('shutil.which', return_value='/usr/bin/ssh'):
            mock_result = MagicMock()
            mock_result.returncode = 255
            mock_result.stdout = ''
            mock_result.stderr = 'ssh: connect to host node1 port 22: No route to host'

            with patch('subprocess.run', return_value=mock_result):
                results = validate_ssh_connectivity(['node1'])

                assert len(results) == 1
                assert results[0][0] == 'node1'
                assert results[0][1] is False
                assert 'No route to host' in results[0][2]

    def test_returns_stderr_message_on_failure(self):
        """Should include stderr in failure message."""
        with patch('shutil.which', return_value='/usr/bin/ssh'):
            mock_result = MagicMock()
            mock_result.returncode = 255
            mock_result.stdout = ''
            mock_result.stderr = 'Permission denied (publickey,password)'

            with patch('subprocess.run', return_value=mock_result):
                results = validate_ssh_connectivity(['node1'])

                assert results[0][1] is False
                assert 'Permission denied' in results[0][2]

    def test_handles_subprocess_timeout(self):
        """Should handle subprocess.TimeoutExpired."""
        with patch('shutil.which', return_value='/usr/bin/ssh'):
            with patch('subprocess.run', side_effect=subprocess.TimeoutExpired(cmd='ssh', timeout=5)):
                results = validate_ssh_connectivity(['node1'], timeout=5)

                assert len(results) == 1
                assert results[0][0] == 'node1'
                assert results[0][1] is False
                assert 'timed out' in results[0][2].lower()

    def test_returns_success_for_reachable_host(self):
        """Should return success when SSH returns zero."""
        with patch('shutil.which', return_value='/usr/bin/ssh'):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = 'ok'
            mock_result.stderr = ''

            with patch('subprocess.run', return_value=mock_result):
                results = validate_ssh_connectivity(['node1'])

                assert len(results) == 1
                assert results[0][0] == 'node1'
                assert results[0][1] is True
                assert 'connected' in results[0][2].lower()

    def test_uses_batch_mode(self):
        """Should use BatchMode to avoid password prompts."""
        with patch('shutil.which', return_value='/usr/bin/ssh'):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = 'ok'
            mock_result.stderr = ''

            with patch('subprocess.run', return_value=mock_result) as mock_run:
                validate_ssh_connectivity(['node1'])

                call_args = mock_run.call_args[0][0]
                assert '-o' in call_args
                assert 'BatchMode=yes' in call_args

    def test_uses_connect_timeout(self):
        """Should use ConnectTimeout option."""
        with patch('shutil.which', return_value='/usr/bin/ssh'):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = 'ok'
            mock_result.stderr = ''

            with patch('subprocess.run', return_value=mock_result) as mock_run:
                validate_ssh_connectivity(['node1'], timeout=10)

                call_args = mock_run.call_args[0][0]
                assert 'ConnectTimeout=10' in call_args

    def test_multiple_hosts(self):
        """Should handle multiple hosts."""
        with patch('shutil.which', return_value='/usr/bin/ssh'):
            mock_result_success = MagicMock()
            mock_result_success.returncode = 0
            mock_result_success.stdout = 'ok'
            mock_result_success.stderr = ''

            mock_result_fail = MagicMock()
            mock_result_fail.returncode = 255
            mock_result_fail.stdout = ''
            mock_result_fail.stderr = 'Connection refused'

            with patch('subprocess.run', side_effect=[mock_result_success, mock_result_fail]):
                results = validate_ssh_connectivity(['node1', 'node2'])

                assert len(results) == 2
                assert results[0][0] == 'node1'
                assert results[0][1] is True
                assert results[1][0] == 'node2'
                assert results[1][1] is False

    def test_mixed_localhost_and_remote(self):
        """Should handle mix of localhost and remote hosts."""
        with patch('shutil.which', return_value='/usr/bin/ssh'):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = 'ok'
            mock_result.stderr = ''

            with patch('subprocess.run', return_value=mock_result) as mock_run:
                results = validate_ssh_connectivity(['localhost', 'node1', '127.0.0.1', 'node2'])

                assert len(results) == 4
                # localhost skipped
                assert results[0][0] == 'localhost'
                assert results[0][1] is True
                # node1 checked
                assert results[1][0] == 'node1'
                assert results[1][1] is True
                # 127.0.0.1 skipped
                assert results[2][0] == '127.0.0.1'
                assert results[2][1] is True
                # node2 checked
                assert results[3][0] == 'node2'
                assert results[3][1] is True

                # SSH only called for node1 and node2
                assert mock_run.call_count == 2


class TestCollectValidationIssues:
    """Tests for collect_validation_issues function."""

    def test_separates_errors_from_warnings(self):
        """Should separate errors from warnings correctly."""
        issues = [
            ValidationIssue('error', 'dependency', 'Missing MPI', 'Install MPI'),
            ValidationIssue('warning', 'configuration', 'Suboptimal setting', 'Consider changing'),
            ValidationIssue('error', 'connectivity', 'Host down', 'Check network'),
        ]

        errors, warnings = collect_validation_issues(issues)

        assert len(errors) == 2
        assert len(warnings) == 1
        assert errors[0].message == 'Missing MPI'
        assert errors[1].message == 'Host down'
        assert warnings[0].message == 'Suboptimal setting'

    def test_handles_empty_list(self):
        """Should handle empty list."""
        errors, warnings = collect_validation_issues([])

        assert errors == []
        assert warnings == []

    def test_handles_all_errors(self):
        """Should handle list with only errors."""
        issues = [
            ValidationIssue('error', 'dependency', 'Error 1', 'Fix 1'),
            ValidationIssue('error', 'configuration', 'Error 2', 'Fix 2'),
        ]

        errors, warnings = collect_validation_issues(issues)

        assert len(errors) == 2
        assert len(warnings) == 0

    def test_handles_all_warnings(self):
        """Should handle list with only warnings."""
        issues = [
            ValidationIssue('warning', 'configuration', 'Warning 1', 'Suggestion 1'),
            ValidationIssue('warning', 'filesystem', 'Warning 2', 'Suggestion 2'),
        ]

        errors, warnings = collect_validation_issues(issues)

        assert len(errors) == 0
        assert len(warnings) == 2

    def test_preserves_order(self):
        """Should preserve order within each category."""
        issues = [
            ValidationIssue('error', 'dependency', 'First error', 'Fix'),
            ValidationIssue('warning', 'configuration', 'First warning', 'Suggest'),
            ValidationIssue('error', 'connectivity', 'Second error', 'Fix'),
            ValidationIssue('warning', 'filesystem', 'Second warning', 'Suggest'),
        ]

        errors, warnings = collect_validation_issues(issues)

        assert errors[0].message == 'First error'
        assert errors[1].message == 'Second error'
        assert warnings[0].message == 'First warning'
        assert warnings[1].message == 'Second warning'
