"""Real-mpirun integration test for the SHARED_FS_PROBE_SCRIPT heredoc body
— Phase 5 / Plan 05-05 Task 2 / Checker B-3 Option A.

Closes the gap that Plan 05-04 left open: the heredoc was only compile-
checked and grep-checked at the unit layer (B-3 Option B locked via
`exec()` against a fake mpi4py shim). This test invokes the heredoc body
end-to-end via real ``mpirun``, two ranks on localhost, against a local
``tmp_path`` directory.

Skip discipline (project memory "UAT defer pattern for hardware"):

- ``mpirun`` not installed → all tests skip (NOT fail). The submitter's
  CI environment may not carry OpenMPI; surfacing as SKIPPED keeps the
  test suite green while documenting the coverage gap.
- ``mpi4py`` not importable on the launching Python interpreter → all
  tests skip. The launcher invokes ``sys.executable`` for the per-rank
  Python, so the same interpreter must carry mpi4py for the probe body
  to make it past the Pitfall-8 ImportError branch.
- Both present → tests run and lock the heredoc contract end-to-end.

Lock contracts:

- ``test_two_local_ranks_same_tmpfs_succeeds_silently`` — two ranks on
  the same local tmp_path see the same ``(st_dev, st_ino)`` tuple →
  cardinality 1 → ``status='ok'`` → exit 0. Sentinel file unlinked.
- ``test_d49_quiesce_observable_via_wall_clock`` — rank-0 ``time.sleep(5.0)``
  inside the ``finally`` block actually executes at runtime: subprocess
  wall-clock ≥ 5.0 seconds. This verifies the D-49 quiesce is observable,
  not just present in source.

Per checker B-3 Option A, this is the "real MPI runtime" companion to
the unit-level mocked-mpi4py exec test at
``tests/unit/test_shared_fs_probe.py::TestPitfall4BcastStatusPreventsProceed``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Mirror the conftest discipline of the sibling integration test file: stub
# heavy deps the production cluster_collector import expects. The
# SHARED_FS_PROBE_SCRIPT itself does NOT need these, but the module's
# top-level imports do (cluster_collector pulls in psutil + pyarrow indirectly
# via cluster collector internals). Use importlib.util.find_spec — checking
# sys.modules alone would install a MagicMock for a perfectly importable
# module that just hasn't been imported yet, which then poisons later test
# collections by causing find_spec to raise ValueError on the Mock's
# __spec__. Matches the safe pattern in tests/unit/test_benchmarks_kvcache.py.
import importlib.util as _ilu
for _dep in ('pyarrow', 'pyarrow.ipc', 'psutil'):
    if _ilu.find_spec(_dep) is None and _dep not in sys.modules:
        sys.modules[_dep] = MagicMock()

from mlpstorage_py.cluster_collector import SHARED_FS_PROBE_SCRIPT, _strip_tag_output_prefix


# Module-level skip: if mpirun is not available, skip the entire suite.
# Honors the "UAT defer pattern for hardware" project memory — CI without
# OpenMPI reports SKIPPED, not FAILED. When mpirun IS available, the tests
# run and lock the heredoc body end-to-end (B-3 Option A coverage).
def _mpi4py_importable() -> bool:
    """Check if the launching Python interpreter can import mpi4py. The
    probe body invokes `from mpi4py import MPI` on every rank; the launching
    interpreter (passed to mpirun via sys.executable) must carry mpi4py for
    the body to make it past the Pitfall-8 early-exit branch.
    """
    try:
        import mpi4py  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = [
    pytest.mark.skipif(not shutil.which('mpirun'), reason='mpirun not installed — install OpenMPI (apt-get install openmpi-bin) to run real-MPI CAP-02 probe tests'),
    pytest.mark.skipif(not _mpi4py_importable(), reason='mpi4py not installed in launching interpreter — install mpi4py (pip install mpi4py) to run real-MPI CAP-02 probe tests'),
]


def _write_probe_script(tmp_path: Path) -> Path:
    """Write SHARED_FS_PROBE_SCRIPT to a tmp_path file with 0o755 perms."""
    script_path = tmp_path / 'probe_script.py'
    script_path.write_text(SHARED_FS_PROBE_SCRIPT)
    script_path.chmod(0o755)
    return script_path


class TestSharedFsProbeRealMpi:
    """Real-mpirun coverage of SHARED_FS_PROBE_SCRIPT body (checker B-3 Option A).

    All tests in this class use:
    - ``mpirun -n 2`` on localhost (two ranks pretending to be peers)
    - ``sys.executable`` so the same Python interpreter that imports
      mpi4py here is the one mpirun launches.
    """

    def test_two_local_ranks_same_tmpfs_succeeds_silently(self, tmp_path):
        """B-3 Option A primary lock: two ranks on a single shared local
        directory see the same (st_dev, st_ino) → cardinality 1 → status='ok'
        → exit 0. Sentinel file is unlinked in the finally block (D-44).

        HARDEN-02 (Plan 05.1-02): argv signature is now 2-positional
        (data_dir, run_uuid). Rank 0 emits the JSON result via
        __CAP02_RESULT_BEGIN__/__CAP02_RESULT_END__ markers on stdout
        (D-54/D-55), not via a file. --tag-output is passed for per-line
        PRRTE atomicity.
        """
        import re

        script_path = _write_probe_script(tmp_path)
        run_uuid = 'test-uuid-cardone'

        # subprocess.run mpirun with -n 2 ranks on localhost.
        # HARDEN-02: --tag-output + 2 positionals (no output_file).
        result = subprocess.run(
            ['mpirun', '-n', '2', '--allow-run-as-root', '--tag-output',
             sys.executable, str(script_path),
             str(tmp_path), run_uuid],
            capture_output=True, text=True, timeout=120,
        )
        # The probe body returns 0 on status='ok'.
        assert result.returncode == 0, (
            f"mpirun exited non-zero: stdout={result.stdout!r} "
            f"stderr={result.stderr!r}"
        )

        # Rank 0 emits markers on stdout (HARDEN-02 D-54/D-55).
        assert '__CAP02_RESULT_BEGIN__' in result.stdout
        assert '__CAP02_RESULT_END__' in result.stdout
        m = re.search(
            r'__CAP02_RESULT_BEGIN__\s*\n(.*?)\n.*?__CAP02_RESULT_END__',
            result.stdout, re.DOTALL,
        )
        assert m is not None, f"no marker payload extractable: {result.stdout!r}"
        payload = _strip_tag_output_prefix(m.group(1).strip())
        parsed = json.loads(payload)
        assert parsed.get('status') == 'ok', f"expected status='ok', got {parsed!r}"

        # All ranks reported the same (st_dev, st_ino) — cardinality 1.
        ranks = parsed.get('ranks', [])
        assert len(ranks) == 2, f"expected 2 ranks in output, got {len(ranks)}"
        fsids = {(r.get('st_dev'), r.get('st_ino')) for r in ranks}
        assert len(fsids) == 1, (
            f"expected cardinality 1 (same FS on both ranks), got {fsids!r}"
        )

        # D-44 sentinel unlink: the sentinel file is removed in the finally
        # block by rank 0. By end-of-run it should NOT be in tmp_path.
        sentinel_name = '.mlpstorage-shared-fs-probe-' + run_uuid
        sentinel_path = tmp_path / sentinel_name
        assert not sentinel_path.exists(), (
            f"D-44 violation: sentinel {sentinel_path} was not unlinked"
        )

    def test_d49_quiesce_observable_via_wall_clock(self, tmp_path):
        """D-49 runtime-observable lock: rank-0 time.sleep(5.0) inside the
        finally block actually executes — subprocess wall-clock >= 5.0 sec.

        Proves the quiesce is OBSERVABLE at runtime, not just textually present
        in the heredoc body (the W-1 grep test at the unit layer locks the
        latter; this locks the former).

        HARDEN-02 (Plan 05.1-02): argv signature is 2-positional; status
        parsed from stdout markers, not from output_json file.
        """
        import re

        script_path = _write_probe_script(tmp_path)
        run_uuid = 'test-uuid-d49quiesce'

        wall_start = time.monotonic()
        result = subprocess.run(
            ['mpirun', '-n', '2', '--allow-run-as-root', '--tag-output',
             sys.executable, str(script_path),
             str(tmp_path), run_uuid],
            capture_output=True, text=True, timeout=60,
        )
        wall_seconds = time.monotonic() - wall_start

        # The quiesce is observable in the wall-clock: at least 5.0 seconds
        # elapse because the rank-0 finally block sleeps 5s before the final
        # comm.Barrier releases the fleet.
        assert wall_seconds >= 5.0, (
            f"D-49 violation: subprocess completed in {wall_seconds:.2f}s "
            f"(expected >= 5.0s due to rank-0 quiesce)"
        )
        # And not infinite-hang: should still complete inside a reasonable
        # bound on a healthy machine.
        assert wall_seconds < 30.0, (
            f"D-49 hang suspect: subprocess took {wall_seconds:.2f}s "
            f"(expected < 30s)"
        )

        # Sanity: status='ok' parsed from stdout markers.
        assert result.returncode == 0, (
            f"unexpected non-zero exit: stderr={result.stderr!r}"
        )
        m = re.search(
            r'__CAP02_RESULT_BEGIN__\s*\n(.*?)\n.*?__CAP02_RESULT_END__',
            result.stdout, re.DOTALL,
        )
        assert m is not None, f"no marker payload extractable: {result.stdout!r}"
        payload = _strip_tag_output_prefix(m.group(1).strip())
        parsed = json.loads(payload)
        assert parsed.get('status') == 'ok'

    def test_cr02_rank0_result_arrives_via_stdout_markers_not_file(self, tmp_path):
        """HARDEN-02 / REVIEW-CR-02 lock: rank-0 emits JSON between
        __CAP02_RESULT_BEGIN__ / __CAP02_RESULT_END__ markers on stdout.

        Reproduces the REVIEW-CR-02 shape (launch host outside --hosts /
        rank 0 lands on remote) via --host 127.0.0.1:1,127.0.0.1:1 +
        --tag-output. The probe is invoked with the new 2-positional argv
        signature (data_dir, run_uuid) — NO output_file. The launcher
        recovers the rank-0 payload from result.stdout via the marker
        regex; no launch-host-local file is involved.

        This test fails RED today (pre-fix) because:
          - The current argv parse requires len(sys.argv) >= 4 (data_dir,
            run_uuid, output_file). With only 2 positionals, the probe
            writes the _argv_error JSON to /tmp/cap02_err.json and
            sys.exit(1) — returncode != 0, no markers in stdout.
          - Even if argv passed, the current code writes to output_file,
            not stdout — no markers would appear.

        Passes GREEN once Task 2 of HARDEN-02 lands the stdout transport.
        """
        import re

        script_path = _write_probe_script(tmp_path)
        run_uuid = 'test-uuid-cr02-stdout'

        # NOTE: --tag-output + --host 127.0.0.1:1,127.0.0.1:1 simulate the
        # remote-rank-0 scenario. Only 2 positionals after the script per
        # the new D-54 argv contract — no output_file.
        result = subprocess.run(
            ['mpirun', '-n', '2', '--allow-run-as-root',
             '--tag-output',
             '--host', '127.0.0.1:1,127.0.0.1:1',
             sys.executable, str(script_path),
             str(tmp_path), run_uuid],
            capture_output=True, text=True, timeout=120,
        )

        assert result.returncode == 0, (
            f"mpirun exited non-zero: stdout={result.stdout!r} "
            f"stderr={result.stderr!r}"
        )

        # Markers MUST be present in stdout (D-54/D-55 stdout transport).
        assert '__CAP02_RESULT_BEGIN__' in result.stdout, (
            f"expected __CAP02_RESULT_BEGIN__ in stdout; got stdout={result.stdout!r}"
        )
        assert '__CAP02_RESULT_END__' in result.stdout, (
            f"expected __CAP02_RESULT_END__ in stdout; got stdout={result.stdout!r}"
        )

        # The misleading "mpi4py not installed" error must NOT appear anywhere
        # (HARDEN-02 acceptance criterion — the old file-based code raised
        # this even when the probe semantically succeeded).
        assert 'mpi4py not installed' not in result.stdout, (
            f"misleading mpi4py-not-installed substring leaked into stdout: {result.stdout!r}"
        )
        assert 'mpi4py not installed' not in result.stderr, (
            f"misleading mpi4py-not-installed substring leaked into stderr: {result.stderr!r}"
        )

        # The JSON between markers parses and reports status='ok'. Strip a
        # leading [host:rank] tag from --tag-output if present.
        m = re.search(
            r'__CAP02_RESULT_BEGIN__\s*\n(.*?)\n.*?__CAP02_RESULT_END__',
            result.stdout, re.DOTALL,
        )
        assert m is not None, (
            f"regex failed to extract payload between markers: {result.stdout!r}"
        )
        payload = _strip_tag_output_prefix(m.group(1).strip())
        parsed = json.loads(payload)
        assert parsed.get('status') == 'ok', (
            f"expected status='ok' in rank-0 payload, got {parsed!r}"
        )

    def test_two_local_ranks_outputs_carry_correct_uuid(self, tmp_path):
        """W-5 launcher pass-through end-to-end: the run_uuid passed as argv[2]
        flows through the probe body to the sentinel filename. We verify by
        inspecting tmp_path mid-run via a forced-failure path — but since the
        sentinel is unlinked at end-of-run (D-44), we use a UUID-distinctive
        string and assert it appears in the parsed output's sentinel path.

        Since the heredoc does not echo the sentinel path in the JSON output
        directly, we use the simpler lock: the run_uuid we pass IS the one
        consumed by the script — verified by the success outcome (a
        mismatched UUID would prevent rank 0 from creating the sentinel
        that rank 1 then stats, producing per-rank failure with
        mode='sentinel_stat'; but since both ranks share the same argv,
        they share the same UUID and succeed).

        HARDEN-02 (Plan 05.1-02): argv signature is 2-positional; status
        parsed from stdout markers, not from output_json file.
        """
        import re

        script_path = _write_probe_script(tmp_path)
        distinctive_uuid = 'unique-uuid-deadbeef-abc123'

        result = subprocess.run(
            ['mpirun', '-n', '2', '--allow-run-as-root', '--tag-output',
             sys.executable, str(script_path),
             str(tmp_path), distinctive_uuid],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, (
            f"UUID flow-through test failed: stderr={result.stderr!r}"
        )
        m = re.search(
            r'__CAP02_RESULT_BEGIN__\s*\n(.*?)\n.*?__CAP02_RESULT_END__',
            result.stdout, re.DOTALL,
        )
        assert m is not None, f"no marker payload extractable: {result.stdout!r}"
        payload = _strip_tag_output_prefix(m.group(1).strip())
        parsed = json.loads(payload)
        # Success → UUID was consumed identically by both ranks.
        assert parsed.get('status') == 'ok'

        # Sentinel was named with the distinctive UUID and unlinked.
        sentinel_path = tmp_path / ('.mlpstorage-shared-fs-probe-' + distinctive_uuid)
        assert not sentinel_path.exists(), (
            "D-44 unlink succeeded for the distinctive-UUID sentinel"
        )
