"""
Tests for dependency validation in mlpstorage.dependency_check module.

Tests cover:
- Executable availability checking
- MPI runtime validation
- DLIO benchmark validation
- Combined dependency validation
- OS-aware dependency checking with install hints
- SSH availability checking
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from mlpstorage.dependency_check import (
    check_executable_available,
    check_mpi_available,
    check_dlio_available,
    validate_benchmark_dependencies,
    check_mpi_with_hints,
    check_dlio_with_hints,
    check_ssh_available,
)
from mlpstorage.environment import OSInfo
from mlpstorage.errors import DependencyError


class TestCheckExecutableAvailable:
    """Tests for check_executable_available function."""

    def test_finds_executable_in_path(self):
        """Should find executable that exists in PATH."""
        # 'python' or 'python3' should be available in any test environment
        with patch('shutil.which', return_value='/usr/bin/python'):
            path = check_executable_available(
                executable='python',
                friendly_name='Python',
                install_suggestion='Install Python'
            )
            assert path == '/usr/bin/python'

    def test_finds_executable_in_search_paths(self, tmp_path):
        """Should find executable in additional search paths."""
        # Create a fake executable
        fake_exe = tmp_path / "fake_benchmark"
        fake_exe.touch()
        fake_exe.chmod(0o755)

        with patch('shutil.which', return_value=None):
            path = check_executable_available(
                executable='fake_benchmark',
                friendly_name='Fake Benchmark',
                install_suggestion='Install it',
                search_paths=[str(tmp_path)]
            )
            assert path == str(fake_exe)

    def test_raises_when_not_found(self):
        """Should raise DependencyError when executable not found."""
        with patch('shutil.which', return_value=None):
            with pytest.raises(DependencyError) as exc_info:
                check_executable_available(
                    executable='nonexistent_tool',
                    friendly_name='Nonexistent Tool',
                    install_suggestion='Cannot install - it does not exist'
                )

            assert 'Nonexistent Tool not found' in str(exc_info.value)

    def test_error_includes_suggestion(self):
        """Should include install suggestion in error."""
        with patch('shutil.which', return_value=None):
            with pytest.raises(DependencyError) as exc_info:
                check_executable_available(
                    executable='missing',
                    friendly_name='Missing Tool',
                    install_suggestion='Run: pip install missing-tool'
                )

            # Check the suggestion is in the error
            assert exc_info.value.suggestion == 'Run: pip install missing-tool'


class TestCheckMpiAvailable:
    """Tests for check_mpi_available function."""

    def test_finds_mpirun(self):
        """Should find mpirun when available."""
        with patch('shutil.which', return_value='/usr/bin/mpirun'):
            path = check_mpi_available('mpirun')
            assert path == '/usr/bin/mpirun'

    def test_finds_mpiexec(self):
        """Should find mpiexec when specified."""
        with patch('shutil.which', return_value='/usr/bin/mpiexec'):
            path = check_mpi_available('mpiexec')
            assert path == '/usr/bin/mpiexec'

    def test_raises_with_helpful_message(self):
        """Should raise DependencyError with installation instructions."""
        with patch('shutil.which', return_value=None):
            with pytest.raises(DependencyError) as exc_info:
                check_mpi_available('mpirun')

            error = exc_info.value
            assert 'MPI runtime' in str(error)
            assert 'apt-get install openmpi-bin' in error.suggestion


class TestCheckDlioAvailable:
    """Tests for check_dlio_available function."""

    def test_finds_dlio_in_path(self):
        """Should find dlio_benchmark in PATH."""
        with patch('shutil.which', return_value='/usr/local/bin/dlio_benchmark'):
            path = check_dlio_available()
            assert path == '/usr/local/bin/dlio_benchmark'

    def test_finds_dlio_in_custom_path(self, tmp_path):
        """Should find dlio_benchmark in custom path."""
        # Create fake dlio executable
        dlio_exe = tmp_path / "dlio_benchmark"
        dlio_exe.touch()
        dlio_exe.chmod(0o755)

        with patch('shutil.which', return_value=None):
            path = check_dlio_available(dlio_bin_path=str(tmp_path))
            assert path == str(dlio_exe)

    def test_raises_with_helpful_message(self):
        """Should raise DependencyError with installation instructions."""
        with patch('shutil.which', return_value=None):
            with pytest.raises(DependencyError) as exc_info:
                check_dlio_available()

            error = exc_info.value
            assert 'DLIO benchmark' in str(error)
            assert "pip install -e '.[full]'" in error.suggestion


class TestValidateBenchmarkDependencies:
    """Tests for validate_benchmark_dependencies function."""

    def test_validates_all_dependencies(self):
        """Should validate both MPI and DLIO when required."""
        with patch('shutil.which') as mock_which:
            mock_which.side_effect = lambda x: f'/usr/bin/{x}'

            mpi_path, dlio_path = validate_benchmark_dependencies(
                requires_mpi=True,
                requires_dlio=True,
                mpi_bin='mpirun'
            )

            assert mpi_path == '/usr/bin/mpirun'
            assert dlio_path == '/usr/bin/dlio_benchmark'

    def test_skips_mpi_when_not_required(self):
        """Should skip MPI check when not required."""
        with patch('shutil.which', return_value='/usr/bin/dlio_benchmark'):
            mpi_path, dlio_path = validate_benchmark_dependencies(
                requires_mpi=False,
                requires_dlio=True
            )

            assert mpi_path is None
            assert dlio_path == '/usr/bin/dlio_benchmark'

    def test_skips_dlio_when_not_required(self):
        """Should skip DLIO check when not required."""
        with patch('shutil.which', return_value='/usr/bin/mpirun'):
            mpi_path, dlio_path = validate_benchmark_dependencies(
                requires_mpi=True,
                requires_dlio=False
            )

            assert mpi_path == '/usr/bin/mpirun'
            assert dlio_path is None

    def test_logs_debug_messages(self):
        """Should log debug messages when logger provided."""
        mock_logger = MagicMock()

        with patch('shutil.which', return_value='/usr/bin/test'):
            validate_benchmark_dependencies(
                requires_mpi=True,
                requires_dlio=True,
                logger=mock_logger
            )

        # Should have called debug at least twice (once for MPI, once for DLIO)
        assert mock_logger.debug.call_count >= 2

    def test_fails_fast_on_missing_mpi(self):
        """Should fail immediately when MPI is missing."""
        def mock_which(cmd):
            if cmd == 'mpirun':
                return None
            return f'/usr/bin/{cmd}'

        with patch('shutil.which', side_effect=mock_which):
            with pytest.raises(DependencyError) as exc_info:
                validate_benchmark_dependencies(
                    requires_mpi=True,
                    requires_dlio=True
                )

            assert 'MPI' in str(exc_info.value)

    def test_fails_fast_on_missing_dlio(self):
        """Should fail immediately when DLIO is missing."""
        def mock_which(cmd):
            if cmd == 'dlio_benchmark':
                return None
            return f'/usr/bin/{cmd}'

        with patch('shutil.which', side_effect=mock_which):
            with pytest.raises(DependencyError) as exc_info:
                validate_benchmark_dependencies(
                    requires_mpi=True,
                    requires_dlio=True
                )

            assert 'DLIO' in str(exc_info.value)


class TestCheckMpiWithHints:
    """Tests for check_mpi_with_hints function with OS-aware error messages."""

    def test_finds_mpirun_when_available(self):
        """Should return path when mpirun is available."""
        with patch('mlpstorage.dependency_check.shutil.which', return_value='/usr/bin/mpirun'):
            path = check_mpi_with_hints('mpirun')
            assert path == '/usr/bin/mpirun'

    def test_finds_mpiexec_when_available(self):
        """Should return path when mpiexec is available."""
        with patch('mlpstorage.dependency_check.shutil.which', return_value='/usr/local/bin/mpiexec'):
            path = check_mpi_with_hints('mpiexec')
            assert path == '/usr/local/bin/mpiexec'

    def test_raises_dependency_error_when_missing(self):
        """Should raise DependencyError when MPI is not found."""
        mock_os_info = OSInfo(
            system='Linux',
            release='5.15.0',
            machine='x86_64',
            distro_id='ubuntu',
            distro_name='Ubuntu',
            distro_version='22.04'
        )

        with patch('mlpstorage.dependency_check.shutil.which', return_value=None):
            with patch('mlpstorage.dependency_check.detect_os', return_value=mock_os_info):
                with pytest.raises(DependencyError) as exc_info:
                    check_mpi_with_hints('mpirun')

                error = exc_info.value
                assert 'mpirun' in str(error) or 'MPI' in str(error)

    def test_error_contains_ubuntu_install_command(self):
        """Error message should contain apt-get for Ubuntu."""
        mock_os_info = OSInfo(
            system='Linux',
            release='5.15.0',
            machine='x86_64',
            distro_id='ubuntu',
            distro_name='Ubuntu',
            distro_version='22.04'
        )

        with patch('mlpstorage.dependency_check.shutil.which', return_value=None):
            with patch('mlpstorage.dependency_check.detect_os', return_value=mock_os_info):
                with pytest.raises(DependencyError) as exc_info:
                    check_mpi_with_hints('mpirun')

                error = exc_info.value
                # Check suggestion contains apt-get command
                assert 'apt-get' in error.suggestion or 'apt-get' in str(error)

    def test_error_contains_macos_install_command(self):
        """Error message should contain brew for macOS."""
        mock_os_info = OSInfo(
            system='Darwin',
            release='21.6.0',
            machine='x86_64'
        )

        with patch('mlpstorage.dependency_check.shutil.which', return_value=None):
            with patch('mlpstorage.dependency_check.detect_os', return_value=mock_os_info):
                with pytest.raises(DependencyError) as exc_info:
                    check_mpi_with_hints('mpirun')

                error = exc_info.value
                # Check suggestion contains brew command
                assert 'brew' in error.suggestion or 'brew' in str(error)

    def test_error_contains_rhel_install_command(self):
        """Error message should contain dnf for RHEL."""
        mock_os_info = OSInfo(
            system='Linux',
            release='5.14.0',
            machine='x86_64',
            distro_id='rhel',
            distro_name='Red Hat Enterprise Linux',
            distro_version='8.5'
        )

        with patch('mlpstorage.dependency_check.shutil.which', return_value=None):
            with patch('mlpstorage.dependency_check.detect_os', return_value=mock_os_info):
                with pytest.raises(DependencyError) as exc_info:
                    check_mpi_with_hints('mpirun')

                error = exc_info.value
                # Check suggestion contains dnf command for RHEL
                assert 'dnf' in error.suggestion or 'dnf' in str(error)


class TestCheckDlioWithHints:
    """Tests for check_dlio_with_hints function."""

    def test_finds_dlio_in_path(self):
        """Should find dlio_benchmark when available in PATH."""
        with patch('mlpstorage.dependency_check.shutil.which', return_value='/usr/local/bin/dlio_benchmark'):
            path = check_dlio_with_hints()
            assert path == '/usr/local/bin/dlio_benchmark'

    def test_finds_dlio_in_custom_path(self, tmp_path):
        """Should find dlio_benchmark in custom path."""
        # Create fake dlio executable
        dlio_exe = tmp_path / "dlio_benchmark"
        dlio_exe.touch()
        dlio_exe.chmod(0o755)

        with patch('mlpstorage.dependency_check.shutil.which', return_value=None):
            path = check_dlio_with_hints(dlio_bin_path=str(tmp_path))
            assert path == str(dlio_exe)

    def test_raises_dependency_error_when_missing(self):
        """Should raise DependencyError when DLIO is not found."""
        with patch('mlpstorage.dependency_check.shutil.which', return_value=None):
            with pytest.raises(DependencyError) as exc_info:
                check_dlio_with_hints()

            error = exc_info.value
            assert 'dlio' in str(error).lower() or 'DLIO' in str(error)

    def test_error_contains_pip_install_suggestion(self):
        """Error message should contain pip install command."""
        with patch('mlpstorage.dependency_check.shutil.which', return_value=None):
            with pytest.raises(DependencyError) as exc_info:
                check_dlio_with_hints()

            error = exc_info.value
            # Check suggestion contains pip install options
            assert 'pip' in error.suggestion or 'pip' in str(error)

    def test_prefers_path_over_custom_path(self):
        """Should prefer PATH location over custom path."""
        with patch('mlpstorage.dependency_check.shutil.which', return_value='/usr/bin/dlio_benchmark'):
            path = check_dlio_with_hints(dlio_bin_path='/custom/path')
            # Should use PATH, not custom path
            assert path == '/usr/bin/dlio_benchmark'


class TestCheckSshAvailable:
    """Tests for check_ssh_available function."""

    def test_finds_ssh_when_available(self):
        """Should find ssh when available in PATH."""
        with patch('mlpstorage.dependency_check.shutil.which', return_value='/usr/bin/ssh'):
            path = check_ssh_available()
            assert path == '/usr/bin/ssh'

    def test_raises_dependency_error_when_missing(self):
        """Should raise DependencyError when SSH is not found."""
        mock_os_info = OSInfo(
            system='Linux',
            release='5.15.0',
            machine='x86_64',
            distro_id='ubuntu',
            distro_name='Ubuntu',
            distro_version='22.04'
        )

        with patch('mlpstorage.dependency_check.shutil.which', return_value=None):
            with patch('mlpstorage.dependency_check.detect_os', return_value=mock_os_info):
                with pytest.raises(DependencyError) as exc_info:
                    check_ssh_available()

                error = exc_info.value
                assert 'ssh' in str(error).lower() or 'SSH' in str(error)

    def test_error_contains_ubuntu_install_command(self):
        """Error message should contain apt-get for Ubuntu."""
        mock_os_info = OSInfo(
            system='Linux',
            release='5.15.0',
            machine='x86_64',
            distro_id='ubuntu',
            distro_name='Ubuntu',
            distro_version='22.04'
        )

        with patch('mlpstorage.dependency_check.shutil.which', return_value=None):
            with patch('mlpstorage.dependency_check.detect_os', return_value=mock_os_info):
                with pytest.raises(DependencyError) as exc_info:
                    check_ssh_available()

                error = exc_info.value
                assert 'apt-get' in error.suggestion or 'apt-get' in str(error)

    def test_error_contains_rhel_install_command(self):
        """Error message should contain dnf for RHEL."""
        mock_os_info = OSInfo(
            system='Linux',
            release='5.14.0',
            machine='x86_64',
            distro_id='rhel',
            distro_name='Red Hat Enterprise Linux',
            distro_version='8.5'
        )

        with patch('mlpstorage.dependency_check.shutil.which', return_value=None):
            with patch('mlpstorage.dependency_check.detect_os', return_value=mock_os_info):
                with pytest.raises(DependencyError) as exc_info:
                    check_ssh_available()

                error = exc_info.value
                # RHEL uses openssh-clients (not openssh-client)
                assert 'dnf' in error.suggestion or 'dnf' in str(error)

    def test_error_contains_macos_message(self):
        """Error message should mention macOS includes SSH."""
        mock_os_info = OSInfo(
            system='Darwin',
            release='21.6.0',
            machine='x86_64'
        )

        with patch('mlpstorage.dependency_check.shutil.which', return_value=None):
            with patch('mlpstorage.dependency_check.detect_os', return_value=mock_os_info):
                with pytest.raises(DependencyError) as exc_info:
                    check_ssh_available()

                error = exc_info.value
                # macOS should say SSH is included
                assert 'macOS' in error.suggestion or 'included' in error.suggestion.lower()
