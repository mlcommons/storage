"""Tests for warmup-run labelling in reportgen output (GH#616).

Training workloads produce 6 disk directories = 1 throwaway warmup + 5
submission runs. DLIO stamps the warmup's ``summary.start`` with the FIRST
real run's start time (not the warmup's own directory time), so the two
runs collide on ``run_id``. Only the 5 real runs enter the JSON/CSV
aggregate (dict dedup by ``run_id`` handles that), but the workload
printer iterates the walker's pre-dedup list and previously double-printed
the warmup as a normal ``[CLOSED]`` run — which reads as duplication.

These tests pin the fix: detect the collision, tag the lex-earlier
result_dir basename as the warmup, and label it in stdout as
``[WARMUP, not aggregated]``.
"""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from mlpstorage_py.report_generator import ReportGenerator, Result
from mlpstorage_py.config import BENCHMARK_TYPES, PARAM_VALIDATION
from mlpstorage_py.rules.models import RunID


def _make_bare_generator(tmp_path):
    """Instantiate ReportGenerator with accumulate/print patched out so tests
    can populate state manually and exercise individual methods."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    with patch.object(ReportGenerator, 'accumulate_results'):
        with patch.object(ReportGenerator, 'print_results'):
            return ReportGenerator(str(results_dir), validate_structure=False)


def _make_run(run_id, result_dir, benchmark_type=BENCHMARK_TYPES.training,
              model='resnet50', accelerator='h100'):
    """Build a minimal BenchmarkRun-like mock with the attributes used by
    _process_single_run and _print_workload_details."""
    m = MagicMock()
    m.run_id = run_id
    m.result_dir = result_dir
    m.benchmark_type = benchmark_type
    m.model = model
    m.accelerator = accelerator
    m.command = 'run'
    m.metrics = {}
    m.issues = []
    return m


class TestWarmupDetection:
    """Collision-based warmup detection at accumulation time."""

    def test_collision_marks_earlier_basename_as_warmup(self, tmp_path):
        gen = _make_bare_generator(tmp_path)
        shared_id = RunID('training', 'run', 'resnet50',
                          '2025-07-10T14:22:24.203210')

        # Warmup disk-time 14:12:19; summary.start mis-stamped to 14:22:24.
        warmup = _make_run(shared_id,
                           '/results/training/resnet50/20250710_141219')
        # First real run: disk-time 14:22:19, matching summary.start.
        real = _make_run(shared_id,
                         '/results/training/resnet50/20250710_142219')

        with patch(
            'mlpstorage_py.report_generator.BenchmarkVerifier'
        ) as mv:
            mv.return_value.verify.return_value = PARAM_VALIDATION.CLOSED
            mv.return_value.issues = []
            gen._process_single_run(warmup)
            gen._process_single_run(real)

        assert '/results/training/resnet50/20250710_141219' in gen.warmup_result_dirs
        assert '/results/training/resnet50/20250710_142219' not in gen.warmup_result_dirs

    def test_collision_walk_order_independent(self, tmp_path):
        """Warmup detection must be independent of get_runs_files() iteration
        order (os.walk order is not guaranteed on all filesystems)."""
        gen = _make_bare_generator(tmp_path)
        shared_id = RunID('training', 'run', 'resnet50',
                          '2025-07-10T14:22:24.203210')

        warmup = _make_run(shared_id,
                           '/results/training/resnet50/20250710_141219')
        real = _make_run(shared_id,
                         '/results/training/resnet50/20250710_142219')

        with patch(
            'mlpstorage_py.report_generator.BenchmarkVerifier'
        ) as mv:
            mv.return_value.verify.return_value = PARAM_VALIDATION.CLOSED
            mv.return_value.issues = []
            # Reverse order: real first, warmup second.
            gen._process_single_run(real)
            gen._process_single_run(warmup)

        assert '/results/training/resnet50/20250710_141219' in gen.warmup_result_dirs
        assert '/results/training/resnet50/20250710_142219' not in gen.warmup_result_dirs

    def test_no_collision_no_warmup(self, tmp_path):
        """Non-colliding runs must not be flagged as warmups (checkpointing
        workloads have 1 disk dir = 1 run — no warmup on disk)."""
        gen = _make_bare_generator(tmp_path)
        rid = RunID('checkpointing', 'run', 'llama3-8b',
                    '2025-07-10T14:22:24')
        run = _make_run(
            rid,
            '/results/checkpointing/llama3-8b/20250710_142224',
            benchmark_type=BENCHMARK_TYPES.checkpointing,
            model='llama3-8b',
            accelerator=None,
        )

        with patch(
            'mlpstorage_py.report_generator.BenchmarkVerifier'
        ) as mv:
            mv.return_value.verify.return_value = PARAM_VALIDATION.CLOSED
            mv.return_value.issues = []
            gen._process_single_run(run)

        assert gen.warmup_result_dirs == set()

    def test_same_run_id_different_systems_no_collision(self, tmp_path):
        """Warmup collision detection must be scoped per-system.

        Two systems can produce identical RunIDs (RunID is
        program+command+model+run_datetime — no system field) when their
        DLIO summary.json files happen to share a start timestamp. Those
        are two independent real runs, not a warmup/real pair. Prior to
        the (system_scope, run_id) collision key, one of them would be
        mislabelled as a warmup and dropped from the aggregate.
        """
        gen = _make_bare_generator(tmp_path)
        shared_id = RunID('training', 'run', 'resnet50',
                          '2025-07-10T14:22:24.203210')

        # Same run_id, same timestamp basename, DIFFERENT <system>/ parent.
        # System scope for training walks up 4 dirs from the leaf; these
        # two paths resolve to different <system>/ folders.
        run_sysA = _make_run(
            shared_id,
            '/mount/results/sysA/training/resnet50/run/20250710_142224',
        )
        run_sysB = _make_run(
            shared_id,
            '/mount/results/sysB/training/resnet50/run/20250710_142224',
        )

        with patch(
            'mlpstorage_py.report_generator.BenchmarkVerifier'
        ) as mv:
            mv.return_value.verify.return_value = PARAM_VALIDATION.CLOSED
            mv.return_value.issues = []
            gen._process_single_run(run_sysA)
            gen._process_single_run(run_sysB)

        # Neither run is flagged as a warmup: they belong to different
        # systems, so the (system_scope, run_id) keys don't collide.
        assert gen.warmup_result_dirs == set()
        # Both runs are retained in the aggregate.
        assert len(gen.run_results) == 2
        scopes = {scope for scope, _rid in gen.run_results.keys()}
        assert scopes == {
            '/mount/results/sysA',
            '/mount/results/sysB',
        }, f"Expected one entry per system scope; got {scopes}"

    def test_same_run_id_same_system_still_collides(self, tmp_path):
        """Collision detection must still fire when two runs share a system.

        Companion to test_same_run_id_different_systems_no_collision: same
        run_id under the SAME <system>/ folder is the real DLIO-stamp
        warmup pattern and MUST be caught.
        """
        gen = _make_bare_generator(tmp_path)
        shared_id = RunID('training', 'run', 'resnet50',
                          '2025-07-10T14:22:24.203210')

        warmup = _make_run(
            shared_id,
            '/mount/results/sysA/training/resnet50/run/20250710_141219',
        )
        real = _make_run(
            shared_id,
            '/mount/results/sysA/training/resnet50/run/20250710_142224',
        )

        with patch(
            'mlpstorage_py.report_generator.BenchmarkVerifier'
        ) as mv:
            mv.return_value.verify.return_value = PARAM_VALIDATION.CLOSED
            mv.return_value.issues = []
            gen._process_single_run(warmup)
            gen._process_single_run(real)

        assert (
            '/mount/results/sysA/training/resnet50/run/20250710_141219'
            in gen.warmup_result_dirs
        )
        assert (
            '/mount/results/sysA/training/resnet50/run/20250710_142224'
            not in gen.warmup_result_dirs
        )
        # Only the real run wins the run_results slot.
        assert len(gen.run_results) == 1


class TestWarmupPrintLabel:
    """Warmup rendering in _print_workload_details."""

    def test_workload_print_labels_warmup(self, tmp_path, capsys):
        gen = _make_bare_generator(tmp_path)
        shared_id = RunID('training', 'run', 'resnet50',
                          '2025-07-10T14:22:24.203210')
        second_id = RunID('training', 'run', 'resnet50',
                          '2025-07-10T14:30:16.920770')

        warmup = _make_run(
            shared_id,
            '/results/training/resnet50/20250710_141219',
        )
        real1 = _make_run(
            shared_id,
            '/results/training/resnet50/20250710_142219',
        )
        real2 = _make_run(
            second_id,
            '/results/training/resnet50/20250710_143012',
        )

        gen.warmup_result_dirs = {'/results/training/resnet50/20250710_141219'}
        gen.run_results = {
            # Post-collision-scoping (GH#616 follow-up): run_results is
            # keyed by (system_scope, run_id). For these short mock paths
            # (/results/training/<model>/<ts>/), _system_scope_key walks
            # up 4 dirs from the leaf and lands at '/' (safe: all runs
            # in this test share that scope, so the collision key still
            # groups them together).
            ('/', shared_id): Result(
                multi=False,
                benchmark_type=BENCHMARK_TYPES.training,
                benchmark_command='run',
                benchmark_model='resnet50',
                benchmark_run=real1,
                issues=[],
                category=PARAM_VALIDATION.CLOSED,
                metrics={},
            ),
            ('/', second_id): Result(
                multi=False,
                benchmark_type=BENCHMARK_TYPES.training,
                benchmark_command='run',
                benchmark_model='resnet50',
                benchmark_run=real2,
                issues=[],
                category=PARAM_VALIDATION.CLOSED,
                metrics={},
            ),
        }
        workload_result = Result(
            multi=True,
            benchmark_type=BENCHMARK_TYPES.training,
            benchmark_command='run',
            benchmark_model='resnet50',
            benchmark_run=[warmup, real1, real2],
            issues=[],
            category=PARAM_VALIDATION.CLOSED,
            metrics={},
        )
        gen._print_workload_details(('resnet50', 'h100'), workload_result)
        out = capsys.readouterr().out

        assert '[WARMUP, not aggregated' in out
        # Warmup's disk basename must appear so submitters can locate the dir.
        assert '20250710_141219' in out

        warmup_line = next(
            line for line in out.splitlines()
            if '[WARMUP, not aggregated' in line
        )
        # Warmup line must NOT carry a CLOSED/OPEN/INVALID category badge —
        # WARMUP is a separate axis from the submission category.
        assert 'CLOSED' not in warmup_line
        assert 'OPEN' not in warmup_line
        assert 'INVALID' not in warmup_line

    def test_workload_print_sorts_runs_deterministically(self, tmp_path,
                                                         capsys):
        """Runs must render in disk-basename lex order so the warmup (always
        lex-earliest by design of the DLIO stamp mismatch) is visually the
        first entry."""
        gen = _make_bare_generator(tmp_path)
        id_a = RunID('training', 'run', 'resnet50',
                     '2025-07-10T14:22:24')
        id_b = RunID('training', 'run', 'resnet50',
                     '2025-07-10T14:30:16')
        id_c = RunID('training', 'run', 'resnet50',
                     '2025-07-10T14:38:09')

        warmup = _make_run(
            id_a, '/results/training/resnet50/20250710_141219')
        run_a = _make_run(
            id_a, '/results/training/resnet50/20250710_142219')
        run_b = _make_run(
            id_b, '/results/training/resnet50/20250710_143012')
        run_c = _make_run(
            id_c, '/results/training/resnet50/20250710_143805')

        gen.warmup_result_dirs = {'/results/training/resnet50/20250710_141219'}
        gen.run_results = {
            # See note in test_workload_print_labels_warmup: scope is '/'
            # because these mock paths are shorter than the canonical tree.
            ('/', rid): Result(
                multi=False,
                benchmark_type=BENCHMARK_TYPES.training,
                benchmark_command='run',
                benchmark_model='resnet50',
                benchmark_run=run,
                issues=[],
                category=PARAM_VALIDATION.CLOSED,
                metrics={},
            )
            for rid, run in [(id_a, run_a), (id_b, run_b), (id_c, run_c)]
        }
        # Pass runs to workload_result in a scrambled order.
        workload_result = Result(
            multi=True,
            benchmark_type=BENCHMARK_TYPES.training,
            benchmark_command='run',
            benchmark_model='resnet50',
            benchmark_run=[run_c, warmup, run_b, run_a],
            issues=[],
            category=PARAM_VALIDATION.CLOSED,
            metrics={},
        )
        gen._print_workload_details(('resnet50', 'h100'), workload_result)
        out = capsys.readouterr().out

        # Warmup (basename 141219) must render first — its disk basename
        # is the only unique-in-output token for it (its run_id is
        # mis-stamped to match run_a's).
        warmup_pos = out.find('20250710_141219')
        # Real runs are identified by their unique run_datetime timestamps.
        ts_a_pos = out.rfind('14:22:24')  # rfind: second occurrence (run_a),
                                          # first is inside the warmup line.
        ts_b_pos = out.find('14:30:16')
        ts_c_pos = out.find('14:38:09')
        positions = [warmup_pos, ts_a_pos, ts_b_pos, ts_c_pos]
        assert all(p >= 0 for p in positions), \
            f"All runs must appear in output; positions={positions}"
        assert positions == sorted(positions), \
            f"Runs must render in disk-basename lex order; got {positions}"

    def test_no_warmup_prints_all_runs_with_category_badge(self, tmp_path,
                                                           capsys):
        """When no collision occurred, the print path behaves as before."""
        gen = _make_bare_generator(tmp_path)
        rid = RunID('checkpointing', 'run', 'llama3-8b',
                    '2025-07-10T14:22:24')
        run = _make_run(
            rid,
            '/results/checkpointing/llama3-8b/20250710_142224',
            benchmark_type=BENCHMARK_TYPES.checkpointing,
            model='llama3-8b',
            accelerator=None,
        )

        gen.warmup_result_dirs = set()
        gen.run_results = {
            # Checkpointing scope walks up 3 from '/results/checkpointing/
            # llama3-8b/20250710_142224' → '/results'.
            ('/results', rid): Result(
                multi=False,
                benchmark_type=BENCHMARK_TYPES.checkpointing,
                benchmark_command='run',
                benchmark_model='llama3-8b',
                benchmark_run=run,
                issues=[],
                category=PARAM_VALIDATION.CLOSED,
                metrics={},
            ),
        }
        workload_result = Result(
            multi=True,
            benchmark_type=BENCHMARK_TYPES.checkpointing,
            benchmark_command='run',
            benchmark_model='llama3-8b',
            benchmark_run=[run],
            issues=[],
            category=PARAM_VALIDATION.CLOSED,
            metrics={},
        )
        gen._print_workload_details(('llama3-8b', None), workload_result)
        out = capsys.readouterr().out

        assert '[WARMUP, not aggregated' not in out
        assert 'CLOSED' in out


class TestOutputDirRemoved:
    """--output-dir was removed in GH#616. Summary files are hardcoded to
    <results-dir>/results.{csv,json}."""

    def test_output_dir_attribute_absent(self, tmp_path):
        """Passing an args namespace with output_dir must NOT redirect
        writes; the ReportGenerator no longer has an output_dir attribute."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        args = Namespace(debug=False, output_dir='/some/other/dir')

        with patch.object(ReportGenerator, 'accumulate_results'):
            with patch.object(ReportGenerator, 'print_results'):
                gen = ReportGenerator(
                    str(results_dir), args=args, validate_structure=False,
                )

        assert not hasattr(gen, 'output_dir')

    def test_write_paths_use_results_dir(self, tmp_path):
        """Both write_json_file and write_csv_file write inside
        results_dir even when the caller supplied args.output_dir."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        args = Namespace(debug=False, output_dir='/some/other/dir')

        with patch.object(ReportGenerator, 'accumulate_results'):
            with patch.object(ReportGenerator, 'print_results'):
                gen = ReportGenerator(
                    str(results_dir), args=args, validate_structure=False,
                )
        gen.write_json_file([{'run_id': 'x'}])
        gen.write_csv_file([{'run_id': 'x'}])

        assert (results_dir / 'results.json').exists()
        assert (results_dir / 'results.csv').exists()
