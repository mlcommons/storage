"""Tests for Config.get_model_parallelism and Config.get_closed_mpi_processes.

Config.get_model_parallelism lazy-loads configs/dlio/workload/llama3_{key}.yaml
and returns (tensor_parallelism, pipeline_parallelism) as (tp, pp) tuples,
caching per key so the YAML is read only once per model size.

Config.get_closed_mpi_processes delegates to CLOSED_MPI_PROCESSES[model_size]
(Rules.md Table 2 — total CLOSED processes per model: TP*PP*DP).

References:
  - D-C1 (get_model_parallelism lazy-load + cache)
  - D-C2 (model keys lowercase: '8b', '70b', '405b', '1t')
  - D-C4 (CLOSED_MPI_PROCESSES constant; get_closed_mpi_processes delegate)
  - Rules.md Table 2 (§4.3.4 area) for CLOSED process counts
"""

import pytest
from unittest.mock import patch

from mlpstorage_py.submission_checker.constants import CLOSED_MPI_PROCESSES
from mlpstorage_py.submission_checker.configuration.configuration import Config


@pytest.fixture
def config():
    """Return a Config instance suitable for parallelism / mpi-process tests."""
    return Config(version="v2.0", submitters=["Acme"], skip_output_file=True)


# ---------------------------------------------------------------------------
# CLOSED_MPI_PROCESSES constant
# ---------------------------------------------------------------------------


class TestClosedMpiProcessesConstant:
    """Verify the constant shape and values match Rules.md Table 2."""

    def test_constant_matches_rules_table2(self):
        """CLOSED_MPI_PROCESSES must exactly match the four model keys and their values."""
        assert CLOSED_MPI_PROCESSES == {"8b": 8, "70b": 64, "405b": 512, "1t": 1024}

    def test_constant_keys_are_lowercase(self):
        """Keys must be lowercase per D-C2."""
        for key in CLOSED_MPI_PROCESSES:
            assert key == key.lower(), f"Key {key!r} is not lowercase"

    def test_constant_values_are_ints(self):
        """Values must be plain Python ints."""
        for key, val in CLOSED_MPI_PROCESSES.items():
            assert isinstance(val, int), f"CLOSED_MPI_PROCESSES[{key!r}] is not an int: {val!r}"


# ---------------------------------------------------------------------------
# Config.get_model_parallelism
# ---------------------------------------------------------------------------


class TestGetModelParallelism:
    """Unit tests for Config.get_model_parallelism — four keys + caching + error path."""

    def test_8b_returns_tp1_pp1(self, config):
        """llama3-8b: tensor=1, pipeline=1 per the workload YAML."""
        assert config.get_model_parallelism("8b") == (1, 1)

    def test_70b_returns_tp8_pp1(self, config):
        """llama3-70b: tensor=8, pipeline=1 per the workload YAML."""
        assert config.get_model_parallelism("70b") == (8, 1)

    def test_405b_returns_tp8_pp32(self, config):
        """llama3-405b: tensor=8, pipeline=32 per the workload YAML."""
        assert config.get_model_parallelism("405b") == (8, 32)

    def test_1t_returns_tp8_pp64(self, config):
        """llama3-1t: tensor=8, pipeline=64 per the workload YAML."""
        assert config.get_model_parallelism("1t") == (8, 64)

    def test_caching_reads_yaml_only_once(self, config):
        """Calling get_model_parallelism twice with the same key reads YAML only once."""
        call_count = 0
        real_open = open

        def counting_open(path, *args, **kwargs):
            nonlocal call_count
            if "llama3_8b.yaml" in str(path):
                call_count += 1
            return real_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=counting_open):
            config.get_model_parallelism("8b")
            config.get_model_parallelism("8b")

        assert call_count <= 1, (
            f"Expected the YAML to be read at most once for cached key '8b', "
            f"but 'open' was called {call_count} times"
        )

    def test_unknown_model_raises_file_not_found(self, config):
        """An unrecognized model key should raise FileNotFoundError (no YAML exists)."""
        with pytest.raises(FileNotFoundError):
            config.get_model_parallelism("99x")

    def test_cache_populated_after_first_call(self, config):
        """After one call, the result is in _parallelism_cache."""
        config.get_model_parallelism("70b")
        assert "70b" in config._parallelism_cache
        assert config._parallelism_cache["70b"] == (8, 1)

    def test_different_keys_cached_independently(self, config):
        """Each model key caches independently."""
        config.get_model_parallelism("8b")
        config.get_model_parallelism("1t")
        assert config._parallelism_cache["8b"] == (1, 1)
        assert config._parallelism_cache["1t"] == (8, 64)


# ---------------------------------------------------------------------------
# Config.get_closed_mpi_processes
# ---------------------------------------------------------------------------


class TestGetClosedMpiProcesses:
    """Unit tests for Config.get_closed_mpi_processes — four keys + error path."""

    def test_8b_returns_8(self, config):
        assert config.get_closed_mpi_processes("8b") == 8

    def test_70b_returns_64(self, config):
        assert config.get_closed_mpi_processes("70b") == 64

    def test_405b_returns_512(self, config):
        assert config.get_closed_mpi_processes("405b") == 512

    def test_1t_returns_1024(self, config):
        assert config.get_closed_mpi_processes("1t") == 1024

    def test_unknown_model_raises_key_error(self, config):
        """An unrecognized key must raise KeyError — no silent fallback (D-C4)."""
        with pytest.raises(KeyError):
            config.get_closed_mpi_processes("99x")


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------


class TestConfigBackwardsCompat:
    """Phase 1's Config behavior must be unchanged."""

    def test_default_construction_unchanged(self):
        """Config without new kwargs constructs successfully."""
        c = Config(version="v2.0", submitters=["Acme"])
        assert c.version == "v2.0"
        assert c.submitters == ["Acme"]

    def test_get_reference_checksum_still_works(self):
        """Phase 1's get_reference_checksum is unaffected."""
        c = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
        assert c.get_reference_checksum() is None  # REFERENCE_CHECKSUMS["v2.0"] is None

    def test_parallelism_cache_starts_empty(self):
        """New _parallelism_cache starts empty — no pre-loaded values."""
        c = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
        assert c._parallelism_cache == {}
