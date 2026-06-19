"""
Tests for mlpstorage_py.environment.systemd_ipc (#447 advisory).

The module is a pure-information advisory: given host state, it returns either
None (no risk) or a remediation string. These tests pin both branches of every
gate so the advisory stays accurate as we patch around it.
"""

import os
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from mlpstorage_py.environment import systemd_ipc as sut


# ---------------------------------------------------------------------------
# read_logind_remove_ipc — file parsing
# ---------------------------------------------------------------------------

class TestReadLogindRemoveIPC:
    """RemoveIPC parsing: distro default is 'yes', explicit 'no' flips it,
    drop-ins override the main config in lexical order."""

    def test_returns_true_when_no_config_files_exist(self, tmp_path):
        # No main file, no drop-ins → systemd default 'yes' applies.
        assert sut.read_logind_remove_ipc(
            main_conf=str(tmp_path / "absent.conf"),
            drop_in_glob=str(tmp_path / "absent.d" / "*.conf"),
        ) is True

    def test_explicit_yes_in_main_conf(self, tmp_path):
        main = tmp_path / "logind.conf"
        main.write_text("[Login]\nRemoveIPC=yes\n")
        assert sut.read_logind_remove_ipc(
            main_conf=str(main),
            drop_in_glob=str(tmp_path / "d" / "*.conf"),
        ) is True

    def test_explicit_no_in_main_conf(self, tmp_path):
        main = tmp_path / "logind.conf"
        main.write_text("[Login]\nRemoveIPC=no\n")
        assert sut.read_logind_remove_ipc(
            main_conf=str(main),
            drop_in_glob=str(tmp_path / "d" / "*.conf"),
        ) is False

    def test_commented_setting_treated_as_absent(self, tmp_path):
        # systemd ships the line commented; treat as default (yes).
        main = tmp_path / "logind.conf"
        main.write_text("[Login]\n#RemoveIPC=no\n")
        assert sut.read_logind_remove_ipc(
            main_conf=str(main),
            drop_in_glob=str(tmp_path / "d" / "*.conf"),
        ) is True

    def test_drop_in_overrides_main(self, tmp_path):
        main = tmp_path / "logind.conf"
        main.write_text("[Login]\nRemoveIPC=yes\n")
        drop_in_dir = tmp_path / "logind.conf.d"
        drop_in_dir.mkdir()
        (drop_in_dir / "10-no-reap.conf").write_text("[Login]\nRemoveIPC=no\n")
        assert sut.read_logind_remove_ipc(
            main_conf=str(main),
            drop_in_glob=str(drop_in_dir / "*.conf"),
        ) is False

    def test_later_drop_in_wins_over_earlier(self, tmp_path):
        # Lexical order: 90 > 10. Final value should be 'yes'.
        drop_in_dir = tmp_path / "logind.conf.d"
        drop_in_dir.mkdir()
        (drop_in_dir / "10-policy.conf").write_text("[Login]\nRemoveIPC=no\n")
        (drop_in_dir / "90-override.conf").write_text("[Login]\nRemoveIPC=yes\n")
        assert sut.read_logind_remove_ipc(
            main_conf=str(tmp_path / "absent.conf"),
            drop_in_glob=str(drop_in_dir / "*.conf"),
        ) is True

    def test_value_truthy_strings(self, tmp_path):
        main = tmp_path / "logind.conf"
        for truthy in ("yes", "YES", "true", "on", "1"):
            main.write_text(f"[Login]\nRemoveIPC={truthy}\n")
            assert sut.read_logind_remove_ipc(
                main_conf=str(main),
                drop_in_glob=str(tmp_path / "d" / "*.conf"),
            ) is True, truthy

    def test_value_falsy_strings(self, tmp_path):
        main = tmp_path / "logind.conf"
        for falsy in ("no", "NO", "false", "off", "0"):
            main.write_text(f"[Login]\nRemoveIPC={falsy}\n")
            assert sut.read_logind_remove_ipc(
                main_conf=str(main),
                drop_in_glob=str(tmp_path / "d" / "*.conf"),
            ) is False, falsy


# ---------------------------------------------------------------------------
# is_user_lingering — loginctl wrapper
# ---------------------------------------------------------------------------

class TestIsUserLingering:

    def test_returns_none_when_loginctl_missing(self, monkeypatch):
        monkeypatch.setattr(sut.shutil, "which", lambda _: None)
        assert sut.is_user_lingering(username="alice") is None

    def test_returns_true_when_loginctl_reports_yes(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Linger=yes\n", stderr=""
        )
        with patch.object(sut.subprocess, "run", return_value=completed):
            assert sut.is_user_lingering(
                username="alice", loginctl_path="/usr/bin/loginctl"
            ) is True

    def test_returns_false_when_loginctl_reports_no(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Linger=no\n", stderr=""
        )
        with patch.object(sut.subprocess, "run", return_value=completed):
            assert sut.is_user_lingering(
                username="alice", loginctl_path="/usr/bin/loginctl"
            ) is False

    def test_returns_none_on_nonzero_exit(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="user not found"
        )
        with patch.object(sut.subprocess, "run", return_value=completed):
            assert sut.is_user_lingering(
                username="ghost", loginctl_path="/usr/bin/loginctl"
            ) is None

    def test_returns_none_on_oserror(self):
        with patch.object(sut.subprocess, "run", side_effect=OSError("boom")):
            assert sut.is_user_lingering(
                username="alice", loginctl_path="/usr/bin/loginctl"
            ) is None

    def test_returns_none_on_timeout(self):
        with patch.object(
            sut.subprocess, "run",
            side_effect=subprocess.TimeoutExpired(cmd="loginctl", timeout=5),
        ):
            assert sut.is_user_lingering(
                username="alice", loginctl_path="/usr/bin/loginctl"
            ) is None

    def test_picks_up_username_from_env(self, monkeypatch):
        monkeypatch.setenv("USER", "bob")
        monkeypatch.delenv("LOGNAME", raising=False)
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Linger=yes\n", stderr=""
        )
        with patch.object(sut.subprocess, "run", return_value=completed) as mock_run:
            assert sut.is_user_lingering(loginctl_path="/usr/bin/loginctl") is True
            args, _ = mock_run.call_args
            assert "bob" in args[0]

    def test_returns_none_when_no_username_available(self, monkeypatch):
        monkeypatch.delenv("USER", raising=False)
        monkeypatch.delenv("LOGNAME", raising=False)
        assert sut.is_user_lingering(loginctl_path="/usr/bin/loginctl") is None


# ---------------------------------------------------------------------------
# check_removeipc_risk — full gating logic
# ---------------------------------------------------------------------------

class TestCheckRemoveIPCRisk:
    """Risk = systemd active AND RemoveIPC=yes AND user is not lingering.
    Any other combination returns None."""

    def test_returns_none_when_systemd_not_active(self, tmp_path):
        # Even with RemoveIPC=yes and no linger, non-systemd hosts can't reap.
        assert sut.check_removeipc_risk(
            username="alice",
            main_conf=str(tmp_path / "absent.conf"),
            drop_in_glob=str(tmp_path / "d" / "*.conf"),
            systemd_active=False,
            loginctl_path="/usr/bin/loginctl",
        ) is None

    def test_returns_none_when_remove_ipc_disabled(self, tmp_path):
        main = tmp_path / "logind.conf"
        main.write_text("[Login]\nRemoveIPC=no\n")
        assert sut.check_removeipc_risk(
            username="alice",
            main_conf=str(main),
            drop_in_glob=str(tmp_path / "d" / "*.conf"),
            systemd_active=True,
            loginctl_path="/usr/bin/loginctl",
        ) is None

    def test_returns_none_when_user_is_lingering(self, tmp_path):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Linger=yes\n", stderr=""
        )
        with patch.object(sut.subprocess, "run", return_value=completed):
            result = sut.check_removeipc_risk(
                username="alice",
                main_conf=str(tmp_path / "absent.conf"),  # default 'yes'
                drop_in_glob=str(tmp_path / "d" / "*.conf"),
                systemd_active=True,
                loginctl_path="/usr/bin/loginctl",
            )
        assert result is None

    def test_returns_advisory_when_at_risk(self, tmp_path):
        # systemd active + RemoveIPC=yes default + Linger=no → warn.
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Linger=no\n", stderr=""
        )
        with patch.object(sut.subprocess, "run", return_value=completed):
            result = sut.check_removeipc_risk(
                username="alice",
                main_conf=str(tmp_path / "absent.conf"),
                drop_in_glob=str(tmp_path / "d" / "*.conf"),
                systemd_active=True,
                loginctl_path="/usr/bin/loginctl",
            )
        assert result is not None
        assert "loginctl enable-linger alice" in result
        assert "#447" in result
        assert "RemoveIPC" in result

    def test_advisory_uses_env_user_when_username_omitted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("USER", "ci-runner")
        monkeypatch.delenv("LOGNAME", raising=False)
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Linger=no\n", stderr=""
        )
        with patch.object(sut.subprocess, "run", return_value=completed):
            result = sut.check_removeipc_risk(
                main_conf=str(tmp_path / "absent.conf"),
                drop_in_glob=str(tmp_path / "d" / "*.conf"),
                systemd_active=True,
                loginctl_path="/usr/bin/loginctl",
            )
        assert result is not None
        assert "enable-linger ci-runner" in result

    def test_returns_advisory_when_linger_unknown(self, tmp_path):
        # loginctl missing → Linger status is None (unknown). We still warn,
        # because the safer default for the user is "tell them how to fix it."
        with patch.object(sut.shutil, "which", return_value=None):
            result = sut.check_removeipc_risk(
                username="alice",
                main_conf=str(tmp_path / "absent.conf"),
                drop_in_glob=str(tmp_path / "d" / "*.conf"),
                systemd_active=True,
            )
        assert result is not None
        assert "loginctl enable-linger alice" in result


# ---------------------------------------------------------------------------
# validate_benchmark_environment integration — the advisory is logged
# ---------------------------------------------------------------------------

class TestValidationHelperIntegration:
    """The advisory must reach the user via logger.warning when a benchmark
    that uses Python multiprocessing is being validated on Linux."""

    def _run_validator(self, args, advisory_return):
        from argparse import Namespace
        from unittest.mock import MagicMock as MM
        import mlpstorage_py.validation_helpers as vh

        logger = MM()
        with patch.object(vh, "check_removeipc_risk", return_value=advisory_return), \
             patch.object(vh, "_validate_paths", return_value=[]), \
             patch.object(vh, "_validate_required_params", return_value=[]), \
             patch.object(vh, "check_mpi_with_hints"), \
             patch.object(vh, "check_dlio_with_hints"), \
             patch.object(vh, "detect_os",
                          return_value=Namespace(system="Linux", distro="ubuntu")):
            vh.validate_benchmark_environment(args, logger=logger)
        return logger

    def test_advisory_logged_for_training_run(self):
        from argparse import Namespace
        args = Namespace(
            program="training", command="run", model="unet3d",
            mpi_bin="mpirun", dlio_bin_path=None,
            hosts=[], exec_type=None,
        )
        logger = self._run_validator(args, advisory_return="ADVISORY-TEXT")
        warned = [c for c in logger.warning.call_args_list
                  if "ADVISORY-TEXT" in str(c)]
        assert warned, "expected the #447 advisory to be logger.warning'd"

    def test_advisory_skipped_for_vectordb(self):
        # vectordb is not a DLIO program → no risk of fork-after-MPI_Init
        # and we don't want to spam the warning.
        from argparse import Namespace
        args = Namespace(
            program="vectordb", command="run", model=None,
            mpi_bin="mpirun", dlio_bin_path=None,
            hosts=[], exec_type=None,
        )
        logger = self._run_validator(args, advisory_return="ADVISORY-TEXT")
        warned = [c for c in logger.warning.call_args_list
                  if "ADVISORY-TEXT" in str(c)]
        assert not warned, "vectordb runs should not see the #447 advisory"

    def test_no_warning_when_advisory_returns_none(self):
        from argparse import Namespace
        args = Namespace(
            program="training", command="run", model="unet3d",
            mpi_bin="mpirun", dlio_bin_path=None,
            hosts=[], exec_type=None,
        )
        logger = self._run_validator(args, advisory_return=None)
        # No warning containing the advisory phrase.
        warned = [c for c in logger.warning.call_args_list
                  if "#447" in str(c)]
        assert not warned
