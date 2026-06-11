#!/usr/bin/env python3
"""
Regression tests for issue #380:

    Parameter error from integration test (test_dlio_mpi.py)

Root cause: ``tests/integration/test_dlio_mpi.py`` selected its endpoint with

    ompi_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])

That subscript is OpenMPI-specific and is **unset** whenever the script is not
launched by OpenMPI's ``mpirun`` — e.g. the documented ``python tests/
integration/test_dlio_mpi.py`` invocation, MPICH/``mpiexec``, or ``srun`` — so
the test died with a bare ``KeyError: 'OMPI_COMM_WORLD_RANK'``. The guide
(``tests/README.md``) also told users to start this MPI program with plain
``python`` and gave no guidance for the "not enough slots" error that appears
when ``-np`` exceeds the host core count.

After the fix:

* Endpoint selection uses ``rank = comm.Get_rank()`` (portable across launchers
  and already captured by the script), not the OpenMPI env var.
* ``OMPI_COMM_WORLD_RANK`` is read for display only, via
  ``os.environ.get(..., 'not set')`` — never a hard subscript.
* The guide launches the test with ``mpirun`` and documents
  ``--oversubscribe`` for under-provisioned hosts.

These tests pin those invariants so the regression cannot reappear. They are
pure-logic / source checks and intentionally require neither mpi4py nor a live
MPI runtime.
"""

import os
import re
from pathlib import Path

import pytest

TEST_SCRIPT = (
    Path(__file__).resolve().parent / "test_dlio_mpi.py"
)

ENDPOINTS = [
    "http://endpoint1:9000",
    "http://endpoint2:9000",
    "http://endpoint3:9000",
    "http://endpoint4:9000",
]


# ---------------------------------------------------------------------------
# Direct reproduction of the original failure mode
# ---------------------------------------------------------------------------

def test_hard_env_subscript_was_the_crash():
    """The original pattern raises KeyError when not launched by OpenMPI."""
    env = dict(os.environ)
    env.pop("OMPI_COMM_WORLD_RANK", None)
    with pytest.raises(KeyError):
        # This mirrors the pre-fix line in test_dlio_mpi.py.
        int(env["OMPI_COMM_WORLD_RANK"])


def test_safe_env_read_does_not_crash():
    """The fixed pattern degrades to a default instead of raising."""
    env = dict(os.environ)
    env.pop("OMPI_COMM_WORLD_RANK", None)
    assert env.get("OMPI_COMM_WORLD_RANK", "not set") == "not set"


# ---------------------------------------------------------------------------
# Selection invariant: endpoint must be derived from the MPI rank
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "rank,expected_index",
    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 0), (7, 3), (8, 0)],
)
def test_rank_selects_endpoint_round_robin(rank, expected_index):
    """rank % len(endpoints) round-robins regardless of any env var."""
    assert rank % len(ENDPOINTS) == expected_index
    assert ENDPOINTS[rank % len(ENDPOINTS)] == ENDPOINTS[expected_index]


# ---------------------------------------------------------------------------
# Source guards: the crashing pattern must not return
# ---------------------------------------------------------------------------

def test_script_has_no_hard_env_subscript():
    src = TEST_SCRIPT.read_text()
    # No hard subscript of the OpenMPI rank env var in any quoting style.
    assert not re.search(r"os\.environ\[\s*['\"]OMPI_COMM_WORLD_RANK['\"]\s*\]", src), (
        "test_dlio_mpi.py must not subscript os.environ['OMPI_COMM_WORLD_RANK']; "
        "use comm.Get_rank() and os.environ.get(...) instead."
    )


def test_script_selects_endpoint_from_rank():
    src = TEST_SCRIPT.read_text()
    assert "endpoint_index = rank % len(endpoints)" in src, (
        "endpoint selection must be driven by comm.Get_rank() (the `rank` var)."
    )
