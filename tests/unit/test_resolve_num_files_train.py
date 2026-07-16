"""
Regression test for storage#795 (run-path half).

The workload YAML ``num_files_train`` (retinanet_b200: 1,170,301) is a reference
placeholder sized for a large system — not the count a given submitter should
read. ``datasize`` computes the real per-system minimum from the 5x-memory /
500-step rules. Because ``skip_listing`` is forced on, every rank reconstructs
filenames as ``{prefix}_{idx}_of_{num_files_train}.{ext}`` instead of listing the
directory, so if the run's ``num_files_train`` differs from the value the dataset
was generated with, the reconstructed ``_of_{total}`` names match no object and
the startup HEAD checks all miss. That is the storage#795 abort: the run defaulted
to the YAML 1,170,301 while the reporter had generated the (smaller) datasize set.

``_resolve_num_files_train`` fixes this by defaulting the run/configview
``num_files_train`` to the datasize-computed minimum when the user did not pass
``--params dataset.num_files_train`` — while leaving datagen (different host set,
deliberate over-generation) and explicit user overrides untouched. These tests
exercise the method directly against the same computation ``check_num_files_train``
uses, patched to a known value so the wiring — not the arithmetic — is asserted.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from mlpstorage_py.benchmarks.dlio import TrainingBenchmark

_MODULE = "mlpstorage_py.benchmarks.dlio.calculate_training_data_size"


def _make_stub(*, params_dict=None, cluster_information=object(),
               combined_num_files_train=1_170_301):
    """Minimal shape ``_resolve_num_files_train`` reads off ``self``."""
    return SimpleNamespace(
        args=SimpleNamespace(),
        params_dict=dict(params_dict or {}),
        cluster_information=cluster_information,
        combined_params={
            'dataset': {'num_files_train': combined_num_files_train},
            'reader': {'batch_size': 1},
        },
        logger=MagicMock(),
    )


def test_resolves_to_datasize_minimum_when_not_overridden():
    """The retinanet_b200 reporter scenario: user did not pass num_files_train,
    so the run must default to the computed minimum (257,173), not the YAML
    reference (1,170,301) — otherwise skip_listing reconstructs ``_of_1170301``
    names that miss every generated ``_of_257173`` object."""
    stub = _make_stub(combined_num_files_train=1_170_301)

    with patch(_MODULE, return_value=(257_173, 0, 83_055_820_561)):
        TrainingBenchmark._resolve_num_files_train(stub)

    # Both the DLIO override surface and the effective combined config are
    # updated so the interval calc, the run, and the run-checker all agree.
    assert stub.params_dict['dataset.num_files_train'] == 257_173
    assert stub.combined_params['dataset']['num_files_train'] == 257_173

    logged = ' '.join(str(c.args[0]) for c in stub.logger.info.call_args_list)
    assert '257,173' in logged
    assert '1,170,301' in logged  # names the YAML default it superseded
    assert 'storage#795' in logged


def test_explicit_user_override_is_respected():
    """A submitter running against a deliberately over-generated dataset passes
    ``--params dataset.num_files_train=N``; that value must win and the method
    must not recompute or log."""
    stub = _make_stub(
        params_dict={'dataset.num_files_train': 2_000_000},
        combined_num_files_train=2_000_000,
    )

    with patch(_MODULE) as calc:
        TrainingBenchmark._resolve_num_files_train(stub)
        calc.assert_not_called()

    assert stub.params_dict['dataset.num_files_train'] == 2_000_000
    assert stub.logger.info.call_count == 0


def test_no_cluster_information_is_a_noop():
    """The datagen path has no cluster_information (host-info collection is
    skipped for datagen, and its host set / memory typically differ from the
    run's anyway). With no memory basis to size against, leave the YAML default
    in place and do not touch params_dict."""
    stub = _make_stub(cluster_information=None, combined_num_files_train=1_170_301)

    with patch(_MODULE) as calc:
        TrainingBenchmark._resolve_num_files_train(stub)
        calc.assert_not_called()

    assert 'dataset.num_files_train' not in stub.params_dict
    assert stub.combined_params['dataset']['num_files_train'] == 1_170_301
    assert stub.logger.info.call_count == 0


def test_computed_equal_to_yaml_default_does_not_inject_override():
    """When the computed minimum already equals the YAML default, there is
    nothing to change — and, crucially, the tool must not manufacture a
    redundant override that would show up in the audit trail as a user tune."""
    stub = _make_stub(combined_num_files_train=1_170_301)

    with patch(_MODULE, return_value=(1_170_301, 0, 1)):
        TrainingBenchmark._resolve_num_files_train(stub)

    assert 'dataset.num_files_train' not in stub.params_dict
    assert stub.combined_params['dataset']['num_files_train'] == 1_170_301
    assert stub.logger.info.call_count == 0


def test_compute_failure_degrades_gracefully():
    """If the sizing computation cannot run (missing inputs on some path),
    swallow it at debug level and leave the YAML default — never crash the run
    over an auto-defaulting convenience."""
    stub = _make_stub(combined_num_files_train=1_170_301)

    with patch(_MODULE, side_effect=ValueError("no memory basis")):
        TrainingBenchmark._resolve_num_files_train(stub)

    assert 'dataset.num_files_train' not in stub.params_dict
    assert stub.combined_params['dataset']['num_files_train'] == 1_170_301
    assert stub.logger.info.call_count == 0
    stub.logger.debug.assert_called_once()
