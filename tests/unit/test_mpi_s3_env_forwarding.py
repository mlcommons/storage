"""Tests for S3/storage env-var forwarding to remote MPI ranks via -x flags.

Bug: mlcommons/storage #592 (Problem A).

OpenMPI does not propagate arbitrary env vars to remote ranks.  For multi-host
object-storage runs, every S3/storage env var (AWS_*, S3DLIO_*, STORAGE_LIBRARY,
BUCKET) must be explicitly opted-in via `mpirun -x VARNAME`.  Before the fix,
only DLIO_DROP_CACHES_TIMEOUT was forwarded; remote ranks had no S3 credentials
and their first S3 op hung or failed.

Test strategy: call generate_dlio_command() on a minimally-stubbed
TrainingBenchmark with generate_mpi_prefix_cmd patched to a fixed string, then
assert the -x flags appear in the output.  Tests FAIL before the fix (bug
confirmed) and PASS after.
"""

import os
from argparse import Namespace
from unittest.mock import MagicMock, patch

from mlpstorage_py.benchmarks.dlio import TrainingBenchmark
from mlpstorage_py.config import EXEC_TYPE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXED_MPI_PREFIX = "mpirun -n 3 -host h1,h2,h3 --bind-to none --map-by node"


def _make_dlio_for_cmd():
    """Minimal TrainingBenchmark stub sufficient to call generate_dlio_command()."""
    obj = object.__new__(TrainingBenchmark)
    obj.logger = MagicMock()
    obj.base_command_path = "python -m dlio_benchmark"
    obj._config_name = "unet3d_b200"
    obj.run_result_output = "/tmp/run"
    obj.params_dict = {}
    obj.config_path = "/tmp/config"
    obj.args = Namespace(
        exec_type=EXEC_TYPE.MPI,
        mpi_bin="mpirun",
        hosts=["h1", "h2", "h3"],
        num_processes=3,
        oversubscribe=False,
        allow_run_as_root=False,
        mpi_params=None,
        mpi_btl="auto",
    )
    return obj


_MPI_PREFIX_PATCH = "mlpstorage_py.benchmarks.dlio.generate_mpi_prefix_cmd"


# ---------------------------------------------------------------------------
# BUG: S3 vars not forwarded (tests FAIL before fix, PASS after fix)
# ---------------------------------------------------------------------------

class TestS3EnvForwardingMissing:
    """Each test asserts a required -x flag IS present in the mpirun command.
    All tests FAIL before the fix because the current code only adds
    -x DLIO_DROP_CACHES_TIMEOUT.
    """

    @patch(_MPI_PREFIX_PATCH, return_value=_FIXED_MPI_PREFIX)
    def test_aws_endpoint_url_forwarded(self, _mock, monkeypatch):
        """AWS_ENDPOINT_URL must reach remote ranks so they can connect to the S3 backend."""
        monkeypatch.setenv("AWS_ENDPOINT_URL", "http://aistore:51080/s3/")
        monkeypatch.delenv("DLIO_DROP_CACHES_TIMEOUT", raising=False)
        cmd = _make_dlio_for_cmd().generate_dlio_command()
        assert "-x AWS_ENDPOINT_URL" in cmd, (
            "BUG #592: AWS_ENDPOINT_URL not forwarded to remote MPI ranks"
        )

    @patch(_MPI_PREFIX_PATCH, return_value=_FIXED_MPI_PREFIX)
    def test_aws_access_key_id_forwarded(self, _mock, monkeypatch):
        """AWS_ACCESS_KEY_ID must reach remote ranks so they can authenticate."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testkey")
        monkeypatch.delenv("DLIO_DROP_CACHES_TIMEOUT", raising=False)
        cmd = _make_dlio_for_cmd().generate_dlio_command()
        assert "-x AWS_ACCESS_KEY_ID" in cmd, (
            "BUG #592: AWS_ACCESS_KEY_ID not forwarded to remote MPI ranks"
        )

    @patch(_MPI_PREFIX_PATCH, return_value=_FIXED_MPI_PREFIX)
    def test_aws_secret_access_key_forwarded(self, _mock, monkeypatch):
        """AWS_SECRET_ACCESS_KEY must reach remote ranks for S3 signing."""
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testsecret")
        monkeypatch.delenv("DLIO_DROP_CACHES_TIMEOUT", raising=False)
        cmd = _make_dlio_for_cmd().generate_dlio_command()
        assert "-x AWS_SECRET_ACCESS_KEY" in cmd, (
            "BUG #592: AWS_SECRET_ACCESS_KEY not forwarded to remote MPI ranks"
        )

    @patch(_MPI_PREFIX_PATCH, return_value=_FIXED_MPI_PREFIX)
    def test_s3dlio_follow_redirects_forwarded(self, _mock, monkeypatch):
        """S3DLIO_FOLLOW_REDIRECTS=1 must reach remote ranks (required for AIStore)."""
        monkeypatch.setenv("S3DLIO_FOLLOW_REDIRECTS", "1")
        monkeypatch.delenv("DLIO_DROP_CACHES_TIMEOUT", raising=False)
        cmd = _make_dlio_for_cmd().generate_dlio_command()
        assert "-x S3DLIO_FOLLOW_REDIRECTS" in cmd, (
            "BUG #592: S3DLIO_FOLLOW_REDIRECTS not forwarded to remote MPI ranks"
        )

    @patch(_MPI_PREFIX_PATCH, return_value=_FIXED_MPI_PREFIX)
    def test_storage_library_forwarded(self, _mock, monkeypatch):
        """STORAGE_LIBRARY must reach remote ranks so they use s3dlio, not minio."""
        monkeypatch.setenv("STORAGE_LIBRARY", "s3dlio")
        monkeypatch.delenv("DLIO_DROP_CACHES_TIMEOUT", raising=False)
        cmd = _make_dlio_for_cmd().generate_dlio_command()
        assert "-x STORAGE_LIBRARY" in cmd, (
            "BUG #592: STORAGE_LIBRARY not forwarded to remote MPI ranks"
        )

    @patch(_MPI_PREFIX_PATCH, return_value=_FIXED_MPI_PREFIX)
    def test_bucket_forwarded(self, _mock, monkeypatch):
        """BUCKET must reach remote ranks so they know which S3 bucket to access."""
        monkeypatch.setenv("BUCKET", "mlp-s3dlio")
        monkeypatch.delenv("DLIO_DROP_CACHES_TIMEOUT", raising=False)
        cmd = _make_dlio_for_cmd().generate_dlio_command()
        assert "-x BUCKET" in cmd, (
            "BUG #592: BUCKET not forwarded to remote MPI ranks"
        )


# ---------------------------------------------------------------------------
# Invariant: absent vars must NOT produce -x flags (no spurious forwarding)
# ---------------------------------------------------------------------------

class TestAbsentVarsNotForwarded:
    """Vars not present in the launcher env must not appear as -x flags."""

    @patch(_MPI_PREFIX_PATCH, return_value=_FIXED_MPI_PREFIX)
    def test_absent_aws_endpoint_not_forwarded(self, _mock, monkeypatch):
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("DLIO_DROP_CACHES_TIMEOUT", raising=False)
        cmd = _make_dlio_for_cmd().generate_dlio_command()
        assert "-x AWS_ENDPOINT_URL" not in cmd

    @patch(_MPI_PREFIX_PATCH, return_value=_FIXED_MPI_PREFIX)
    def test_absent_s3dlio_var_not_forwarded(self, _mock, monkeypatch):
        monkeypatch.delenv("S3DLIO_CONNECT_TIMEOUT_SECS", raising=False)
        monkeypatch.delenv("DLIO_DROP_CACHES_TIMEOUT", raising=False)
        cmd = _make_dlio_for_cmd().generate_dlio_command()
        assert "-x S3DLIO_CONNECT_TIMEOUT_SECS" not in cmd


# ---------------------------------------------------------------------------
# Existing DLIO_DROP_CACHES_TIMEOUT forwarding must be preserved
# ---------------------------------------------------------------------------

class TestDropCachesTimeoutStillForwarded:
    """The pre-existing -x DLIO_DROP_CACHES_TIMEOUT forwarding must not regress."""

    @patch(_MPI_PREFIX_PATCH, return_value=_FIXED_MPI_PREFIX)
    def test_drop_caches_still_forwarded(self, _mock, monkeypatch):
        monkeypatch.setenv("DLIO_DROP_CACHES_TIMEOUT", "300")
        cmd = _make_dlio_for_cmd().generate_dlio_command()
        assert "-x DLIO_DROP_CACHES_TIMEOUT" in cmd

    @patch(_MPI_PREFIX_PATCH, return_value=_FIXED_MPI_PREFIX)
    def test_drop_caches_and_s3_forwarded_together(self, _mock, monkeypatch):
        """Both DLIO_DROP_CACHES_TIMEOUT and S3 vars must appear when all are set."""
        monkeypatch.setenv("DLIO_DROP_CACHES_TIMEOUT", "120")
        monkeypatch.setenv("AWS_ENDPOINT_URL", "http://aistore:51080/s3/")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
        monkeypatch.setenv("BUCKET", "mlp-s3dlio")
        cmd = _make_dlio_for_cmd().generate_dlio_command()
        assert "-x DLIO_DROP_CACHES_TIMEOUT" in cmd
        assert "-x AWS_ENDPOINT_URL" in cmd
        assert "-x AWS_ACCESS_KEY_ID" in cmd
        assert "-x BUCKET" in cmd


# ---------------------------------------------------------------------------
# Non-MPI runs must not have -x flags (no exec_type=MPI)
# ---------------------------------------------------------------------------

class TestNonMpiRunsUnchanged:
    """When exec_type is not MPI, no -x flags should appear."""

    def test_no_x_flags_when_not_mpi(self, monkeypatch):
        monkeypatch.setenv("AWS_ENDPOINT_URL", "http://aistore:51080/s3/")
        obj = _make_dlio_for_cmd()
        obj.args.exec_type = None  # not MPI
        cmd = obj.generate_dlio_command()
        assert "-x " not in cmd
