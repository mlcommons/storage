"""
Round-trip regression test for `TrainingBenchmark.generate_datagen_benchmark_command`.

The `datasize` command emits a copy-paste `mlpstorage ... datagen ...` string for
the user to run next. Issue #433 surfaced that this string had drifted from the
real CLI shape (wrong flag name for params, missing mode prefix, missing
storage-protocol positional, `--model=` instead of positional). To stop that
drift class from recurring, this test takes the emitted string, shlex-splits it,
and runs it through the real argparse parser. Any future divergence between the
emitter and the parser fails here in CI.
"""

import shlex
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mlpstorage_py.benchmarks.dlio import TrainingBenchmark
from mlpstorage_py.cli_parser import parse_arguments


def _make_stub(*, mode, model, hosts, num_processes, results_dir, data_dir,
               params_dict, exec_type='mpi'):
    """Build the minimal object shape that generate_datagen_benchmark_command reads."""
    return SimpleNamespace(
        args=SimpleNamespace(
            mode=mode,
            model=model,
            hosts=hosts,
            num_processes=num_processes,
            results_dir=results_dir,
            data_dir=data_dir,
            exec_type=exec_type,
        ),
        params_dict=dict(params_dict),
    )


@pytest.mark.parametrize("mode,model", [
    ("closed", "unet3d"),
    ("closed", "retinanet"),
    ("open", "unet3d"),
])
def test_generated_datagen_command_round_trips(mode, model):
    """The string emitted by datasize must parse via the real CLI parser."""
    stub = _make_stub(
        mode=mode,
        model=model,
        hosts=['172.16.4.101', '172.16.4.102'],
        num_processes=79,
        results_dir='/tmp/results',
        data_dir='/tmp/data',
        params_dict={},
    )

    cmd_str = TrainingBenchmark.generate_datagen_benchmark_command(
        stub, num_files_train=248207024, num_subfolders_train=24820
    )

    tokens = shlex.split(cmd_str)
    assert tokens[0] == 'mlpstorage', f"command must start with mlpstorage, got: {cmd_str!r}"

    with patch('sys.argv', tokens):
        ns = parse_arguments()

    assert ns.mode == mode
    assert ns.benchmark == 'training'
    assert ns.model == model
    assert ns.command == 'datagen'
    assert ns.data_access_protocol == 'file'
    assert ns.num_processes == 79
    assert ns.results_dir == '/tmp/results'
    assert ns.data_dir == '/tmp/data'
    assert ns.hosts == ['172.16.4.101', '172.16.4.102']

    # --params uses nargs='+' action='append' → list of lists. Flatten and check the
    # dotted-key overrides datasize computed are present.
    flattened = [kv for batch in (ns.params or []) for kv in batch]
    assert 'dataset.num_files_train=248207024' in flattened
    assert 'dataset.num_subfolders_train=24820' in flattened


def test_generated_datagen_command_omits_zero_subfolders():
    """When subfolders=0, datasize should not emit that override."""
    stub = _make_stub(
        mode='closed',
        model='unet3d',
        hosts=['127.0.0.1'],
        num_processes=8,
        results_dir='/tmp/results',
        data_dir='/tmp/data',
        params_dict={},
    )

    cmd_str = TrainingBenchmark.generate_datagen_benchmark_command(
        stub, num_files_train=1234, num_subfolders_train=0
    )

    tokens = shlex.split(cmd_str)
    with patch('sys.argv', tokens):
        ns = parse_arguments()

    flattened = [kv for batch in (ns.params or []) for kv in batch]
    assert 'dataset.num_files_train=1234' in flattened
    assert not any(kv.startswith('dataset.num_subfolders_train=') for kv in flattened)


def test_generated_datagen_command_carries_existing_params():
    """Extra dotted-key overrides already in params_dict must flow through --params."""
    stub = _make_stub(
        mode='closed',
        model='unet3d',
        hosts=['127.0.0.1'],
        num_processes=8,
        results_dir='/tmp/results',
        data_dir='/tmp/data',
        params_dict={'reader.read_threads': 4},
    )

    cmd_str = TrainingBenchmark.generate_datagen_benchmark_command(
        stub, num_files_train=1000, num_subfolders_train=0
    )

    tokens = shlex.split(cmd_str)
    with patch('sys.argv', tokens):
        ns = parse_arguments()

    flattened = [kv for batch in (ns.params or []) for kv in batch]
    assert 'reader.read_threads=4' in flattened
    assert 'dataset.num_files_train=1000' in flattened
