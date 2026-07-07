"""Tests for issue #689 — dgen-py generator under-threaded on multi-node runs.

``StreamingCheckpointing._init_generator`` sizes dgen-py's thread pool as
``os.cpu_count() // <mpi divisor>``. ``os.cpu_count()`` is a per-node figure
(the CPUs on THIS host). Prior to this fix the divisor was the GLOBAL MPI
world size (``OMPI_COMM_WORLD_SIZE`` et al.) — a units mismatch whenever a
run spans more than one node, since it divides a per-node quantity by a
total-across-all-nodes quantity.

Reported symptom (mlcommons/storage#689): a `llama3-70b` checkpoint run on
16 nodes x 4 ranks/node (64 global ranks), 192 vCPUs/node, computed
``max_threads = 192 // 64 = 3``. With only 3 generator threads, CPU-side
data generation ran close to the storage write rate and was profiled as the
throughput bottleneck in ~35% of chunks — so the reported number partly
measured client CPU generation speed rather than storage I/O.

The fix divides by the ranks sharing THIS node instead (``_local_ranks_per_node``,
via the launcher's LOCAL size env var), giving ``192 // 4 = 48`` threads/rank
for the same run — a 16x increase, matching what the single-node case already
computed correctly (world size == local size when there's only one node,
so this fix does not change single-node behavior).

Mirrors DLIO_local_changes' storage#671 fix, which hit the identical
global-vs-local mismatch for per-node accounting and resolved it with the
same technique: prefer the launcher's LOCAL size env var over any
global/world figure.

These tests lock the contract at two boundaries so a future refactor cannot
silently regress either:
  * the pure policy helper (``_local_ranks_per_node``)
  * the call site that actually feeds dgen-py (``_init_generator`` ->
    ``dgen_py.Generator(max_threads=...)``)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mlpstorage_py.checkpointing.streaming_checkpoint import (
    StreamingCheckpointing,
    _local_ranks_per_node,
)


class TestLocalRanksPerNode:
    """``_local_ranks_per_node`` is the platform-independent policy: given an
    env mapping, return the number of MPI ranks sharing this node."""

    # --- no MPI info: safe single-rank default -------------------------

    def test_no_env_defaults_to_one(self):
        """No launcher env vars set (single-node / non-MPI run): 1."""
        assert _local_ranks_per_node(env={}) == 1

    # --- each supported launcher is read -------------------------------

    def test_openmpi_local_size_used(self):
        assert _local_ranks_per_node(env={'OMPI_COMM_WORLD_LOCAL_SIZE': '4'}) == 4

    def test_mpich_localnranks_used(self):
        assert _local_ranks_per_node(env={'MPI_LOCALNRANKS': '8'}) == 8

    def test_mvapich2_local_size_used(self):
        assert _local_ranks_per_node(env={'MV2_COMM_WORLD_LOCAL_SIZE': '2'}) == 2

    # --- precedence: first-listed launcher wins when several are set ---

    def test_openmpi_takes_precedence_over_others(self):
        env = {
            'OMPI_COMM_WORLD_LOCAL_SIZE': '4',
            'MPI_LOCALNRANKS': '99',
            'MV2_COMM_WORLD_LOCAL_SIZE': '99',
        }
        assert _local_ranks_per_node(env=env) == 4

    def test_mpich_takes_precedence_over_mvapich2(self):
        env = {'MPI_LOCALNRANKS': '8', 'MV2_COMM_WORLD_LOCAL_SIZE': '99'}
        assert _local_ranks_per_node(env=env) == 8

    # --- global world-size vars must NOT be consulted (the #689 bug) ---

    def test_global_world_size_vars_are_ignored(self):
        """The whole point of the fix: global size vars must never be read
        as a stand-in for local size, even if they're the only thing set."""
        env = {
            'OMPI_COMM_WORLD_SIZE': '64',
            'PMI_SIZE': '64',
            'MV2_COMM_WORLD_SIZE': '64',
        }
        assert _local_ranks_per_node(env=env) == 1

    # --- malformed / non-positive values are skipped, not fatal --------

    def test_malformed_value_falls_through_to_next_var(self):
        env = {'OMPI_COMM_WORLD_LOCAL_SIZE': 'not-a-number', 'MPI_LOCALNRANKS': '6'}
        assert _local_ranks_per_node(env=env) == 6

    def test_all_malformed_defaults_to_one(self):
        env = {
            'OMPI_COMM_WORLD_LOCAL_SIZE': 'x',
            'MPI_LOCALNRANKS': '',
            'MV2_COMM_WORLD_LOCAL_SIZE': 'nope',
        }
        assert _local_ranks_per_node(env=env) == 1

    def test_zero_value_skipped(self):
        """A local size of 0 is nonsensical (a rank always shares its own
        node with at least itself) — treat as malformed, not as a divide-by-zero."""
        env = {'OMPI_COMM_WORLD_LOCAL_SIZE': '0', 'MPI_LOCALNRANKS': '4'}
        assert _local_ranks_per_node(env=env) == 4

    def test_negative_value_skipped(self):
        env = {'OMPI_COMM_WORLD_LOCAL_SIZE': '-1', 'MPI_LOCALNRANKS': '4'}
        assert _local_ranks_per_node(env=env) == 4

    def test_default_env_reads_os_environ(self):
        """With env=None (the default), os.environ is consulted."""
        with patch.dict('os.environ', {'OMPI_COMM_WORLD_LOCAL_SIZE': '4'}, clear=False):
            assert _local_ranks_per_node() == 4


class TestInitGeneratorThreadCount:
    """``_init_generator`` is the actual call site that feeds dgen-py.
    These tests patch ``dgen_py.Generator`` and ``os.cpu_count`` to lock the
    real max_threads value passed to the generator — not just the pure
    policy helper — so a future refactor of the wiring cannot silently
    regress the fix even if ``_local_ranks_per_node`` itself stays correct.
    """

    def _make_checkpoint(self):
        return StreamingCheckpointing(
            chunk_size=1024 * 1024, num_buffers=2, use_dgen=True, backend='file',
        )

    def test_689_regression_16_nodes_4_ranks_per_node(self):
        """The exact scenario from issue #689: 192 vCPUs/node, 16 nodes x 4
        ranks/node (64 global ranks). Prior to the fix this computed
        max_threads = 192 // 64 = 3 (the reported symptom). Fixed:
        max_threads = 192 // 4 = 48."""
        env = {
            'OMPI_COMM_WORLD_LOCAL_SIZE': '4',   # ranks sharing this node
            'OMPI_COMM_WORLD_SIZE': '64',        # global — must be ignored
        }
        ckpt = self._make_checkpoint()
        with patch('mlpstorage_py.checkpointing.streaming_checkpoint.HAS_DGEN', True), \
             patch('mlpstorage_py.checkpointing.streaming_checkpoint.dgen_py') as mock_dgen, \
             patch('os.cpu_count', return_value=192), \
             patch.dict('os.environ', env, clear=True):
            mock_dgen.Generator.return_value = MagicMock()
            ckpt._init_generator(total_size_bytes=1024 * 1024 * 1024)

        assert mock_dgen.Generator.call_count == 1
        kwargs = mock_dgen.Generator.call_args.kwargs
        assert kwargs['max_threads'] == 48, (
            f"max_threads={kwargs['max_threads']}; expected 48 (192 cpus / 4 "
            f"local ranks). Got the pre-fix value (3 = 192 / 64 global ranks) "
            f"if this regresses (#689)."
        )

    def test_single_node_no_mpi_uses_all_cpus(self):
        """No MPI env at all (single-process run): use every CPU — this
        must match pre-fix behavior exactly (world size defaulted to 1
        too), so the fix must not change single-node/non-MPI behavior."""
        ckpt = self._make_checkpoint()
        with patch('mlpstorage_py.checkpointing.streaming_checkpoint.HAS_DGEN', True), \
             patch('mlpstorage_py.checkpointing.streaming_checkpoint.dgen_py') as mock_dgen, \
             patch('os.cpu_count', return_value=16), \
             patch.dict('os.environ', {}, clear=True):
            mock_dgen.Generator.return_value = MagicMock()
            ckpt._init_generator(total_size_bytes=1024 * 1024)

        assert mock_dgen.Generator.call_args.kwargs['max_threads'] == 16

    def test_single_node_multi_rank_divides_by_local_size(self):
        """Single node, 8 ranks on it, no global-size env set at all (a
        launcher that only exposes local info): still throttles correctly."""
        ckpt = self._make_checkpoint()
        env = {'OMPI_COMM_WORLD_LOCAL_SIZE': '8'}
        with patch('mlpstorage_py.checkpointing.streaming_checkpoint.HAS_DGEN', True), \
             patch('mlpstorage_py.checkpointing.streaming_checkpoint.dgen_py') as mock_dgen, \
             patch('os.cpu_count', return_value=64), \
             patch.dict('os.environ', env, clear=True):
            mock_dgen.Generator.return_value = MagicMock()
            ckpt._init_generator(total_size_bytes=1024 * 1024)

        assert mock_dgen.Generator.call_args.kwargs['max_threads'] == 8

    def test_at_least_one_thread_even_with_more_ranks_than_cpus(self):
        """Pathological oversubscription (more local ranks than CPUs) must
        floor at 1 thread, never 0 or negative."""
        ckpt = self._make_checkpoint()
        env = {'OMPI_COMM_WORLD_LOCAL_SIZE': '32'}
        with patch('mlpstorage_py.checkpointing.streaming_checkpoint.HAS_DGEN', True), \
             patch('mlpstorage_py.checkpointing.streaming_checkpoint.dgen_py') as mock_dgen, \
             patch('os.cpu_count', return_value=4), \
             patch.dict('os.environ', env, clear=True):
            mock_dgen.Generator.return_value = MagicMock()
            ckpt._init_generator(total_size_bytes=1024)

        assert mock_dgen.Generator.call_args.kwargs['max_threads'] == 1
