"""Tests for the SSH-connectivity preflight bypass.

The SSH reachability probe in ``validate_benchmark_environment`` is only a
prerequisite for SSH-bootstrapped MPI. Scheduler launchers (HPE/Cray PALS
``mpiexec``, Slurm ``srun``) spawn ranks via the batch daemon, so the probe
must be bypassable — both automatically (launcher/env detection) and via the
``--skip-ssh-check`` flag (``skip_remote_checks``).
"""
import types

from mlpstorage_py.validation_helpers import _launcher_bypasses_ssh


def _args(**kw):
    return types.SimpleNamespace(**kw)


def test_srun_under_slurm_bypasses(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    assert _launcher_bypasses_ssh(_args(mpi_bin="srun")) is True


def test_pals_mpiexec_bypasses(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("PALS_NODEFILE", "/var/spool/pals/nodes")
    # Full path is fine — only the basename matters.
    assert _launcher_bypasses_ssh(_args(mpi_bin="/opt/cray/pals/1.8/bin/mpiexec")) is True


def test_plain_mpirun_still_uses_ssh(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("PALS_NODEFILE", raising=False)
    assert _launcher_bypasses_ssh(_args(mpi_bin="mpirun")) is False


def test_hydra_mpiexec_without_pals_still_uses_ssh(monkeypatch):
    # MPICH-Hydra mpiexec on a generic cluster bootstraps over SSH by default;
    # without PALS_* env we must NOT bypass.
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("PALS_NODEFILE", raising=False)
    assert _launcher_bypasses_ssh(_args(mpi_bin="mpiexec")) is False


def test_srun_without_slurm_env_does_not_bypass(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("PALS_NODEFILE", raising=False)
    assert _launcher_bypasses_ssh(_args(mpi_bin="srun")) is False


def test_missing_mpi_bin_does_not_bypass(monkeypatch):
    monkeypatch.setenv("PALS_NODEFILE", "/var/spool/pals/nodes")
    # No mpi_bin attribute at all -> safe default (do not bypass).
    assert _launcher_bypasses_ssh(_args()) is False
