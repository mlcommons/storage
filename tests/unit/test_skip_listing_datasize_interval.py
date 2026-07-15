"""
Regression test for storage#795.

When the ``datasize`` command runs, ``_apply_skip_listing_params`` must derive
``dataset.listing_validation_interval`` and its INFO log from the DRAM-derived
recommendation that ``datasize()`` computes — not from the workload YAML's
``num_files_train`` default. Reading the YAML default (e.g. 1,170,301 for
retinanet_b200) produced two visible defects:

 1. The datasize INFO log said "skip_listing enabled: 1,170,301 train files"
    even though the ``Number of training files:`` result line seconds later
    said 257,173. The two lines contradicted each other.
 2. The emitted ``mlpstorage ... datagen ...`` hint carried
    ``dataset.listing_validation_interval=1000`` (correct bucket for 1.17M)
    but ``dataset.num_files_train=257173`` — inconsistent params. A user
    running the hint literally got an interval computed for a dataset 4.5x
    larger than they actually generate.

The fix threads a ``num_files_override`` argument through
``_apply_skip_listing_params`` and calls it from ``datasize()`` after the
recommendation lands in ``params_dict``. This test exercises the method
directly with the override to make sure the interval bucket is chosen from
the override, not from ``combined_params``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from mlpstorage_py.benchmarks.dlio import TrainingBenchmark


def _make_stub(*, combined_num_files_train, params_dict=None):
    """Minimal shape ``_apply_skip_listing_params`` reads off ``self``.

    ``_compute_validation_interval`` is a @staticmethod on the class, but the
    method under test calls it as ``self._compute_validation_interval(...)``,
    so the stub needs to route that lookup through to the real staticmethod.
    """
    return SimpleNamespace(
        params_dict=dict(params_dict or {}),
        combined_params={'dataset': {'num_files_train': combined_num_files_train}},
        logger=MagicMock(),
        _compute_validation_interval=TrainingBenchmark._compute_validation_interval,
    )


def test_override_takes_precedence_over_combined_params_default():
    """The retinanet_b200 scenario from issue #795.

    combined_params still holds the workload YAML default (1,170,301). If the
    method reads combined_params, the interval bucket is 1,000 (right for
    1.17M files). The datasize recommendation is 257,173 files, whose correct
    bucket is 100. The override argument must win so the emitted interval
    matches the num_files_train the user is actually going to use.
    """
    stub = _make_stub(combined_num_files_train=1_170_301)

    TrainingBenchmark._apply_skip_listing_params(stub, num_files_override=257_173)

    assert stub.params_dict['dataset.skip_listing'] == 'True'
    assert stub.params_dict['dataset.listing_validation_interval'] == '100'

    # The INFO log must also reflect the override, not the YAML default, or
    # the user sees a "1,170,301 train files" message right above a
    # "Number of training files: 257173" result line.
    logged = ' '.join(str(call.args[0]) for call in stub.logger.info.call_args_list)
    assert '257,173 train files' in logged
    assert '1,170,301' not in logged
    assert 'validation_interval=100' in logged


def test_no_override_falls_back_to_combined_params():
    """Non-datasize commands (run, datagen) still read the effective
    num_files_train off combined_params — either the YAML default, or the
    user's ``--params dataset.num_files_train=N`` override, which flows into
    combined_params before ``__init__`` calls this method."""
    stub = _make_stub(combined_num_files_train=50_000)

    TrainingBenchmark._apply_skip_listing_params(stub)

    # 50,000 files falls in the 10,000 <= n < 100,000 bucket → interval=10.
    assert stub.params_dict['dataset.listing_validation_interval'] == '10'

    logged = ' '.join(str(call.args[0]) for call in stub.logger.info.call_args_list)
    assert '50,000 train files' in logged
    assert 'validation_interval=10' in logged


def test_override_respects_user_supplied_interval():
    """If the user already set ``dataset.listing_validation_interval`` via
    ``--params`` it must not be overwritten — the whole point of the guard
    in the method. The override argument only decides what to compute *if*
    the method is going to compute anything."""
    stub = _make_stub(
        combined_num_files_train=1_170_301,
        params_dict={'dataset.listing_validation_interval': '42'},
    )

    TrainingBenchmark._apply_skip_listing_params(stub, num_files_override=257_173)

    assert stub.params_dict['dataset.listing_validation_interval'] == '42'
    # Since the guard short-circuited, no INFO log should have fired either.
    assert stub.logger.info.call_count == 0
