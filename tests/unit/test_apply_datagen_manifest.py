"""``TrainingBenchmark._apply_datagen_manifest`` — the run-side manifest check
(storage#571 Q4 / storage#795 follow-up; datagen-manifest v1.0 PR-1).

Fires from the training pre-execution gate on ``run`` and ``configview``
(after CAP-02, so on multi-host runs the data-dir has just been proven
shared). Semantics (Curtis 2026-09-21, D2/D3):

* manifest found and ``run <= generated`` → inject
  ``dataset.num_files_generated=<generated>`` (DLIO v3.0.5 reads the first
  ``num_files_train`` files of the larger generated set) and, when the run
  did not set it, ``dataset.num_subfolders_train`` from the manifest so the
  reconstructed names match the generated layout.
* ``run > generated`` → ``MANIFEST-003`` hard error with the exact
  re-datagen command (or the smaller read count when the generated set
  still meets the CLOSED minimum). ``--skip-validation`` and whatif
  downgrade it to a warning.
* invariant mismatch (model / record_length_bytes / num_samples_per_file /
  dataset_format / explicit num_subfolders_train) → ``MANIFEST-001``, same
  escape policy.
* no manifest → ``MANIFEST-000`` warning, today's behaviour (equality).

The run count itself is never changed here: explicit ``--params`` >
storage#795 datasize minimum, and ``check_num_files_train`` still enforces
``run >= minimum``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mlpstorage_py.benchmarks.dlio import TrainingBenchmark
from mlpstorage_py.errors import ConfigurationError
from mlpstorage_py.rules.datagen_hierarchy import DatagenManifest
from mlpstorage_py.rules.run_checkers.training import TrainingRunRulesChecker

_READ = "mlpstorage_py.benchmarks.dlio.read_datagen_manifest"
_CALC = "mlpstorage_py.benchmarks.dlio.calculate_training_data_size"


def _manifest(**over):
    base = dict(
        location="/data/unet3d/.mlps-datagen-manifest.json",
        schema_version=1, model="unet3d",
        num_files_train=1_000, num_samples_per_file=1, record_length_bytes=146600628,
        created_at="2026-09-21T00:00:00Z", mlpstorage_version="3.0.0",
        source_datagen_result_dir="/results/leaf",
        num_subfolders_train=0, dataset_format="npz", rules_edition="3.0",
    )
    base.update(over)
    return DatagenManifest(**base)


def _stub(*, command="run", mode="closed", skip_validation=False, params_dict=None,
          dataset=None, data_dir="/data", model="unet3d"):
    ds = {"num_files_train": 400, "num_samples_per_file": 1,
          "record_length_bytes": 146600628, "format": "npz"}
    ds.update(dataset or {})
    return SimpleNamespace(
        args=SimpleNamespace(command=command, mode=mode, skip_validation=skip_validation,
                             data_dir=data_dir, model=model),
        params_dict=dict(params_dict or {}),
        combined_params={"dataset": ds, "reader": {"batch_size": 1}},
        cluster_information=object(),
        logger=MagicMock(),
        generate_datagen_benchmark_command=lambda n, s: f"HINT files={n} subfolders={s}",
    )


def _apply(stub, manifest, minimum=500):
    with patch(_READ, return_value=manifest), \
         patch(_CALC, return_value=(minimum, 0, 0)):
        TrainingBenchmark._apply_datagen_manifest(stub)


def _logged(stub, level):
    return " ".join(str(c.args[0]) for c in getattr(stub.logger, level).call_args_list)


class TestInjection:
    def test_fits_injects_generated_count(self):
        stub = _stub()
        _apply(stub, _manifest(num_files_train=1_000))
        assert stub.params_dict["dataset.num_files_generated"] == 1_000
        assert stub.combined_params["dataset"]["num_files_generated"] == 1_000
        info = _logged(stub, "info")
        assert "1,000" in info and "400" in info
        stub.logger.warning.assert_not_called()
        stub.logger.error.assert_not_called()

    def test_equal_counts_still_inject(self):
        stub = _stub(dataset={"num_files_train": 1_000})
        _apply(stub, _manifest(num_files_train=1_000))
        assert stub.params_dict["dataset.num_files_generated"] == 1_000

    def test_subfolders_taken_from_manifest_when_run_unset(self):
        stub = _stub()
        _apply(stub, _manifest(num_subfolders_train=8))
        assert stub.params_dict["dataset.num_subfolders_train"] == 8
        assert stub.combined_params["dataset"]["num_subfolders_train"] == 8

    def test_zero_subfolders_is_not_injected(self):
        stub = _stub()
        _apply(stub, _manifest(num_subfolders_train=0))
        assert "dataset.num_subfolders_train" not in stub.params_dict

    def test_explicit_num_files_generated_override_is_kept_with_a_warning(self):
        stub = _stub(params_dict={"dataset.num_files_generated": 900})
        _apply(stub, _manifest(num_files_train=1_000))
        assert stub.params_dict["dataset.num_files_generated"] == 900
        assert "900" in _logged(stub, "warning") and "1,000" in _logged(stub, "warning")

    def test_pre_extension_manifest_skips_optional_invariants(self):
        stub = _stub(params_dict={"dataset.num_subfolders_train": 4})
        _apply(stub, _manifest(num_subfolders_train=None, dataset_format=None, rules_edition=None))
        assert stub.params_dict["dataset.num_files_generated"] == 1_000
        assert stub.params_dict["dataset.num_subfolders_train"] == 4


class TestMissingManifest:
    def test_warns_with_location_and_does_not_inject(self):
        stub = _stub()
        _apply(stub, None)
        w = _logged(stub, "warning")
        assert "MANIFEST-000" in w
        assert "/data/unet3d/.mlps-datagen-manifest.json" in w
        assert "dataset.num_files_generated" not in stub.params_dict

    def test_read_failure_propagates(self):
        stub = _stub()
        with patch(_READ, side_effect=ConfigurationError("corrupt")):
            with pytest.raises(ConfigurationError):
                TrainingBenchmark._apply_datagen_manifest(stub)

    def test_read_failure_is_a_warning_under_skip_validation(self):
        stub = _stub(skip_validation=True)
        with patch(_READ, side_effect=ConfigurationError("corrupt")):
            TrainingBenchmark._apply_datagen_manifest(stub)
        assert "corrupt" in _logged(stub, "warning")
        assert "dataset.num_files_generated" not in stub.params_dict


class TestShortfall:
    def test_run_exceeds_generated_raises_with_hint(self):
        stub = _stub(dataset={"num_files_train": 2_000})
        with pytest.raises(ConfigurationError) as ei:
            _apply(stub, _manifest(num_files_train=1_000))
        text = str(ei.value)
        assert "MANIFEST-003" in text
        assert "2,000" in text and "1,000" in text
        assert "HINT files=2000 subfolders=0" in text
        assert "--skip-validation" in text
        assert "dataset.num_files_generated" not in stub.params_dict

    def test_hint_offers_smaller_read_count_when_generated_meets_minimum(self):
        stub = _stub(dataset={"num_files_train": 2_000})
        with pytest.raises(ConfigurationError) as ei:
            _apply(stub, _manifest(num_files_train=1_000), minimum=500)
        assert "--params dataset.num_files_train=1000" in str(ei.value)

    def test_hint_omits_smaller_read_count_when_generated_is_below_minimum(self):
        stub = _stub(dataset={"num_files_train": 2_000})
        with pytest.raises(ConfigurationError) as ei:
            _apply(stub, _manifest(num_files_train=1_000), minimum=1_500)
        assert "--params dataset.num_files_train=1000" not in str(ei.value)
        assert "HINT files=2000" in str(ei.value)

    def test_hint_uses_manifest_subfolders(self):
        stub = _stub(dataset={"num_files_train": 2_000})
        with pytest.raises(ConfigurationError) as ei:
            _apply(stub, _manifest(num_files_train=1_000, num_subfolders_train=16))
        assert "HINT files=2000 subfolders=16" in str(ei.value)

    def test_skip_validation_downgrades_to_warning(self):
        stub = _stub(dataset={"num_files_train": 2_000}, skip_validation=True)
        _apply(stub, _manifest(num_files_train=1_000))
        assert "MANIFEST-003" in _logged(stub, "warning")
        assert "dataset.num_files_generated" not in stub.params_dict

    def test_whatif_downgrades_to_warning(self):
        stub = _stub(dataset={"num_files_train": 2_000}, mode="whatif")
        _apply(stub, _manifest(num_files_train=1_000))
        assert "MANIFEST-003" in _logged(stub, "warning")


class TestInvariants:
    @pytest.mark.parametrize("field, manifest_value, run_key, run_value", [
        ("record_length_bytes", 146600628, "record_length_bytes", 322957),
        ("num_samples_per_file", 1, "num_samples_per_file", 4),
        ("dataset_format", "npz", "format", "jpeg"),
    ])
    def test_dataset_invariant_mismatch_raises(self, field, manifest_value, run_key, run_value):
        stub = _stub(dataset={run_key: run_value})
        with pytest.raises(ConfigurationError) as ei:
            _apply(stub, _manifest(**{field: manifest_value}))
        text = str(ei.value)
        assert "MANIFEST-001" in text
        assert run_key in text and str(manifest_value) in text and str(run_value) in text
        assert "dataset.num_files_generated" not in stub.params_dict

    def test_model_mismatch_raises(self):
        stub = _stub(model="retinanet", dataset={"record_length_bytes": 322957, "format": "jpeg"})
        with pytest.raises(ConfigurationError) as ei:
            _apply(stub, _manifest(model="unet3d", record_length_bytes=322957, dataset_format="jpeg"))
        assert "MANIFEST-001" in str(ei.value) and "unet3d" in str(ei.value)

    def test_explicit_subfolders_mismatch_raises(self):
        stub = _stub(params_dict={"dataset.num_subfolders_train": 4})
        with pytest.raises(ConfigurationError) as ei:
            _apply(stub, _manifest(num_subfolders_train=8))
        assert "MANIFEST-001" in str(ei.value) and "num_subfolders_train" in str(ei.value)

    def test_all_mismatches_are_reported_together(self):
        stub = _stub(dataset={"record_length_bytes": 1, "num_samples_per_file": 2})
        with pytest.raises(ConfigurationError) as ei:
            _apply(stub, _manifest())
        assert "record_length_bytes" in str(ei.value) and "num_samples_per_file" in str(ei.value)

    def test_skip_validation_downgrades_and_still_injects_count(self):
        stub = _stub(dataset={"record_length_bytes": 1}, skip_validation=True)
        _apply(stub, _manifest())
        assert "MANIFEST-001" in _logged(stub, "warning")
        # The operator took responsibility for the dataset; the count is
        # still the only way DLIO can name the files that ARE there.
        assert stub.params_dict["dataset.num_files_generated"] == 1_000


class TestScope:
    @pytest.mark.parametrize("command", ["datagen", "datasize"])
    def test_noop_for_non_run_commands(self, command):
        stub = _stub(command=command)
        with patch(_READ) as read:
            TrainingBenchmark._apply_datagen_manifest(stub)
            read.assert_not_called()

    def test_noop_without_data_dir(self):
        stub = _stub(data_dir=None)
        with patch(_READ) as read:
            TrainingBenchmark._apply_datagen_manifest(stub)
            read.assert_not_called()

    def test_configview_applies_too(self):
        stub = _stub(command="configview")
        _apply(stub, _manifest())
        assert stub.params_dict["dataset.num_files_generated"] == 1_000

    def test_injected_key_is_tool_injected_for_the_run_checker(self):
        # Otherwise check_allowed_params marks every manifest-aware CLOSED
        # run INVALID for a value the tool wrote (storage#494 precedent).
        assert "dataset.num_files_generated" in TrainingRunRulesChecker.TOOL_INJECTED_PARAMS


# --------------------------------------------------------------------------- #
# D4 linkage: the consumed manifest is snapshotted into the run leaf           #
# --------------------------------------------------------------------------- #

import json as _json
import os as _os

from mlpstorage_py.rules.datagen_hierarchy import (
    DATAGEN_MANIFEST_SNAPSHOT_FILENAME,
    METADATA_DATAGEN_MANIFEST_KEY,
    datagen_manifest_body,
)


def _leaf_stub(tmp_path, **kw):
    stub = _stub(**kw)
    stub.run_result_output = str(tmp_path)
    return stub


def _snapshot(tmp_path):
    p = tmp_path / DATAGEN_MANIFEST_SNAPSHOT_FILENAME
    return _json.loads(p.read_text()) if p.exists() else None


class TestSnapshotIntoRunLeaf:
    def test_run_writes_the_snapshot_and_declares_it(self, tmp_path):
        stub = _leaf_stub(tmp_path)
        _apply(stub, _manifest())
        snap = _snapshot(tmp_path)
        assert snap is not None
        assert snap["location"] == "/data/unet3d/.mlps-datagen-manifest.json"
        assert snap["overridden"] == []
        assert snap["manifest"] == datagen_manifest_body(_manifest())
        assert stub._datagen_manifest_snapshot == DATAGEN_MANIFEST_SNAPSHOT_FILENAME

    def test_configview_does_not_write_a_leaf_file(self, tmp_path):
        stub = _leaf_stub(tmp_path, command="configview")
        _apply(stub, _manifest())
        assert _snapshot(tmp_path) is None
        assert not hasattr(stub, "_datagen_manifest_snapshot")

    def test_no_manifest_means_no_snapshot(self, tmp_path):
        stub = _leaf_stub(tmp_path)
        _apply(stub, None)
        assert _snapshot(tmp_path) is None
        assert not hasattr(stub, "_datagen_manifest_snapshot")

    def test_hard_manifest_003_error_leaves_no_snapshot(self, tmp_path):
        stub = _leaf_stub(tmp_path, dataset={"num_files_train": 2_000})
        with pytest.raises(ConfigurationError):
            _apply(stub, _manifest(num_files_train=1_000))
        assert _snapshot(tmp_path) is None

    def test_skip_validation_over_003_records_the_override(self, tmp_path):
        stub = _leaf_stub(tmp_path, skip_validation=True, dataset={"num_files_train": 2_000})
        _apply(stub, _manifest(num_files_train=1_000))
        assert _snapshot(tmp_path)["overridden"] == ["MANIFEST-003"]
        assert stub._datagen_manifest_snapshot == DATAGEN_MANIFEST_SNAPSHOT_FILENAME

    def test_whatif_over_001_records_the_override(self, tmp_path):
        stub = _leaf_stub(tmp_path, mode="whatif", dataset={"record_length_bytes": 1})
        _apply(stub, _manifest())
        assert _snapshot(tmp_path)["overridden"] == ["MANIFEST-001"]

    def test_both_overrides_are_recorded_in_order(self, tmp_path):
        stub = _leaf_stub(tmp_path, skip_validation=True,
                          dataset={"record_length_bytes": 1, "num_files_train": 2_000})
        _apply(stub, _manifest(num_files_train=1_000))
        assert _snapshot(tmp_path)["overridden"] == ["MANIFEST-001", "MANIFEST-003"]

    def test_stub_without_a_leaf_is_a_no_op(self):
        stub = _stub()
        _apply(stub, _manifest())
        assert stub.params_dict["dataset.num_files_generated"] == 1_000
        assert not hasattr(stub, "_datagen_manifest_snapshot")

    def test_snapshot_write_failure_is_a_warning_not_an_abort(self, tmp_path):
        stub = _leaf_stub(tmp_path / "missing-leaf")
        _apply(stub, _manifest())
        assert stub.params_dict["dataset.num_files_generated"] == 1_000
        assert DATAGEN_MANIFEST_SNAPSHOT_FILENAME in _logged(stub, "warning")
        assert not hasattr(stub, "_datagen_manifest_snapshot")


class TestMetadataDeclaresSnapshot:
    """``training_<ts>_metadata.json`` names the sidecar, like ``provenance_file``."""

    def _metadata(self, tmp_path, declare):
        # A real TrainingBenchmark object without its constructor (which
        # reserves a leaf, registers the run, collects the cluster...), so the
        # inherited ``metadata`` property runs against the real helpers.
        stub = TrainingBenchmark.__new__(TrainingBenchmark)
        stub.args = SimpleNamespace(command="run", model="unet3d", data_dir="/data")
        stub.run_datetime = "20260921_180000"
        stub.run_result_output = str(tmp_path)
        stub.combined_params = {"dataset": {"num_files_train": 400}}
        stub.params_dict = {}
        stub.cluster_information = None
        stub.runtime = None
        stub.verification = None
        stub.command_output_files = []
        if declare:
            stub._datagen_manifest_snapshot = DATAGEN_MANIFEST_SNAPSHOT_FILENAME
        return stub.metadata

    def test_declared_when_snapshot_written(self, tmp_path):
        md = self._metadata(tmp_path, declare=True)
        assert md[METADATA_DATAGEN_MANIFEST_KEY] == DATAGEN_MANIFEST_SNAPSHOT_FILENAME
        assert md["parameters"]["dataset"]["num_files_train"] == 400

    def test_absent_when_no_snapshot(self, tmp_path):
        md = self._metadata(tmp_path, declare=False)
        assert METADATA_DATAGEN_MANIFEST_KEY not in md
