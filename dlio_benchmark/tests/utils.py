"""
Copyright (c) 2022, UChicago Argonne, LLC
All Rights Reserved

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Test Utilities
==============

Shared utility functions for DLIO benchmark tests.
"""

import sys
import shutil
import subprocess

# Check if mpirun or flux is available
ENABLE_FLUX = False
HAS_MPIRUN = shutil.which("mpirun") is not None
HAS_FLUX = shutil.which("flux") is not None and ENABLE_FLUX
HAS_MPI_RUNNER = HAS_MPIRUN or HAS_FLUX
NUM_PROCS = 2 if HAS_MPI_RUNNER else 1
TEST_TIMEOUT_SECONDS = 600  # 10 minutes

def delete_folder(path):
    """Delete a folder and all its contents, ignoring errors."""
    shutil.rmtree(path, ignore_errors=True)


def run_mpi_benchmark(overrides, num_procs=NUM_PROCS, expect_failure=False, timeout=TEST_TIMEOUT_SECONDS):
    """
    Run the benchmark as a subprocess using DLIO's main entry point.
    Uses flux or mpirun if available, otherwise falls back to single process.

    Args:
        overrides: List of Hydra config overrides
        num_procs: Number of MPI processes (default: NUM_PROCS, only used if flux/mpirun is available)
        expect_failure: If True, return result even on non-zero exit code (default: False)
        timeout: Timeout in seconds for the subprocess (default: TEST_TIMEOUT_SECONDS)

    Returns:
        subprocess.CompletedProcess instance
    """
    # Build command to call DLIO's main module
    if HAS_MPI_RUNNER and num_procs > 1:
        # Prefer flux if available, otherwise use mpirun
        if HAS_FLUX:
            cmd = [
                "flux", "run",
                "-n", str(num_procs),
                "--queue=pdebug",
                "--time-limit", "10m",
                sys.executable,
                "-m", "dlio_benchmark.main"
            ] + overrides
            print(f"Running with Flux ({num_procs} processes, queue=pdebug, time-limit=10m): {' '.join(cmd)}")
        else:  # HAS_MPIRUN
            cmd = [
                "mpirun",
                "-np", str(num_procs),
                sys.executable,
                "-m", "dlio_benchmark.main"
            ] + overrides
            print(f"Running with MPI ({num_procs} processes): {' '.join(cmd)}")
    else:
        # Fall back to single process
        if not HAS_MPI_RUNNER:
            print(f"Warning: neither flux nor mpirun found, falling back to single process")
        cmd = [
            sys.executable,
            "-m", "dlio_benchmark.main"
        ] + overrides
        print(f"Running single process: {' '.join(cmd)}")

    # Run the subprocess and wait for completion
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        print(f"ERROR: Command timed out after {timeout} seconds")
        print(f"Command: {' '.join(cmd)}")
        print(f"STDOUT:\n{e.stdout if e.stdout else 'N/A'}")
        print(f"STDERR:\n{e.stderr if e.stderr else 'N/A'}")
        raise RuntimeError(f"Benchmark timed out after {timeout} seconds") from e

    if result.returncode != 0:
        if expect_failure:
            # Expected failure - return the result for inspection
            print(f"Command failed as expected with return code {result.returncode}")
            return result
        else:
            # Unexpected failure - raise error
            print(f"ERROR: Command failed with return code {result.returncode}")
            print(f"Command: {' '.join(cmd)}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            raise RuntimeError(f"Benchmark failed with return code {result.returncode}")

    return result
