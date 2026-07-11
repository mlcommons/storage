"""Unit tests for mlpstorage_py.rules.datagen_hierarchy.

Covers the four helpers introduced for training datagen validation:

    * ``validate_supported_model(model, mode)`` — hard-fail on unsupported
      model for the given submission mode. Whatif skips (matches the D-29
      whatif-exemption policy in Phase 6 reportgen).

    * ``assert_data_dir_hierarchy_absent(data_dir, model)`` — refuse to
      overwrite an existing ``<data-dir>/<model>`` tree. Rationale: a
      re-datagen with different ``num_samples_per_file`` (or any other
      parameter that changes per-file byte size) would leave the old
      and new files interleaved on filename collision, silently
      corrupting the dataset.

    * ``validate_datagen_leaf(leaf_path)`` — stat-only presence check
      of the four datagen-required files plus ``dlio_config/`` and its
      three inner files. Returns a list of human-readable missing-item
      strings (empty list means the leaf is complete). Does NOT walk
      into file contents.

    * ``write_datagen_manifest(...)`` — extract
      ``num_files_train`` / ``num_samples_per_file`` /
      ``record_length_bytes`` from a nested ``dataset`` param dict,
      write ``<data-dir>/<model>/.mlps-datagen-manifest.json`` with the
      documented schema, and return the manifest path. Loud failure if
      any of the three fields is absent from the input dict.
"""
from __future__ import annotations

import json
import os
import pathlib
from unittest.mock import MagicMock, patch

import pytest

from mlpstorage_py.errors import ConfigurationError
from mlpstorage_py.rules.datagen_hierarchy import (
    DATAGEN_MANIFEST_FILENAME,
    DATAGEN_MANIFEST_SCHEMA_VERSION,
    assert_data_dir_hierarchy_absent,
    validate_checkpoint_leaf,
    validate_datagen_leaf,
    validate_run_leaf,
    validate_supported_model,
    write_datagen_manifest,
)


# --------------------------------------------------------------------------- #
# validate_supported_model                                                    #
# --------------------------------------------------------------------------- #


class TestValidateSupportedModel:
    """Mode-aware model allowlist enforcement.

    Rules.md v3.0 restricts CLOSED and OPEN to {unet3d, retinanet}.
    Whatif is simulation and skips all validation per the D-29
    reportgen policy — the helper mirrors that policy so unsupported
    models pass through in whatif.
    """

    @pytest.mark.parametrize("mode", ["closed", "open"])
    @pytest.mark.parametrize("model", ["unet3d", "retinanet"])
    def test_supported_models_accepted_in_closed_and_open(self, mode, model):
        # Must not raise.
        validate_supported_model(model, mode)

    @pytest.mark.parametrize("mode", ["closed", "open"])
    @pytest.mark.parametrize(
        "model", ["cosmoflow", "resnet50", "dlrm", "flux", "gpt2", ""]
    )
    def test_unsupported_models_rejected_in_closed_and_open(self, mode, model):
        with pytest.raises(ConfigurationError) as excinfo:
            validate_supported_model(model, mode)
        # Message should surface both the model and the mode so the
        # operator can see immediately which allowlist rejected them.
        text = str(excinfo.value)
        assert model in text or "model" in text.lower()
        assert mode in text

    @pytest.mark.parametrize(
        "model", ["unet3d", "retinanet", "cosmoflow", "resnet50", "dlrm", "flux"]
    )
    def test_whatif_accepts_any_known_model(self, model):
        # Whatif is simulation — the D-29 pattern skips validation.
        validate_supported_model(model, "whatif")

    def test_unknown_mode_rejected(self):
        # Defensive: a mode outside {closed, open, whatif} means a
        # caller wiring bug — loud failure over silent accept.
        with pytest.raises(ConfigurationError):
            validate_supported_model("unet3d", "bogus")


# --------------------------------------------------------------------------- #
# assert_data_dir_hierarchy_absent                                            #
# --------------------------------------------------------------------------- #


class TestAssertDataDirHierarchyAbsent:
    """Refuse-to-overwrite guard for ``<data-dir>/<model>``.

    Rationale: if the operator re-runs datagen against the same
    ``--data-dir`` with a different ``num_samples_per_file`` (or any
    other per-file size knob), filename collisions may only
    partially overwrite the prior tree. Rather than teach the
    generator to detect that, we refuse and force ``rm -rf`` or a
    different ``--data-dir``.
    """

    def test_absent_hierarchy_returns_cleanly(self, tmp_path):
        # <data-dir>/<model> does not exist — nothing to refuse.
        assert_data_dir_hierarchy_absent(str(tmp_path), "unet3d")

    def test_empty_model_dir_returns_cleanly(self, tmp_path):
        # A pre-created but empty directory is allowed — e.g. the
        # operator ran ``mkdir -p`` in advance. The invariant we
        # protect is "no partially-overwritten files", so an empty
        # dir is fine.
        (tmp_path / "unet3d").mkdir()
        assert_data_dir_hierarchy_absent(str(tmp_path), "unet3d")

    def test_non_empty_model_dir_raises_configuration_error(self, tmp_path):
        model_dir = tmp_path / "unet3d"
        model_dir.mkdir()
        (model_dir / "some_shard.npz").write_bytes(b"prior data")
        with pytest.raises(ConfigurationError) as excinfo:
            assert_data_dir_hierarchy_absent(str(tmp_path), "unet3d")
        text = str(excinfo.value)
        # Operator needs to see the offending path and be told how to
        # recover (rm -rf or a different --data-dir).
        assert str(model_dir) in text
        assert "unet3d" in text

    def test_model_dir_is_a_file_raises(self, tmp_path):
        # An operator laying down a file at <data-dir>/<model> is a
        # pathological state; refuse rather than silently proceed.
        (tmp_path / "unet3d").write_text("stray")
        with pytest.raises(ConfigurationError):
            assert_data_dir_hierarchy_absent(str(tmp_path), "unet3d")

    def test_sibling_model_dir_does_not_block(self, tmp_path):
        # A populated retinanet tree must not block a fresh unet3d
        # datagen — one --data-dir is allowed to hold multiple
        # datasets, as long as the per-model subdirs don't collide.
        (tmp_path / "retinanet").mkdir()
        (tmp_path / "retinanet" / "shard.jpeg").write_bytes(b"x")
        assert_data_dir_hierarchy_absent(str(tmp_path), "unet3d")


# --------------------------------------------------------------------------- #
# validate_datagen_leaf                                                       #
# --------------------------------------------------------------------------- #


def _make_healthy_leaf(root: pathlib.Path, ts: str = "20260708_120000") -> pathlib.Path:
    """Create a healthy datagen leaf directory: all required files + dlio_config."""
    leaf = root / ts
    leaf.mkdir(parents=True)
    (leaf / "training_datagen.stdout.log").write_text("")
    (leaf / "training_datagen.stderr.log").write_text("")
    (leaf / "dlio.log").write_text("")
    (leaf / f"training_{ts}_metadata.json").write_text("{}")
    dlio_cfg = leaf / "dlio_config"
    dlio_cfg.mkdir()
    (dlio_cfg / "config.yaml").write_text("")
    (dlio_cfg / "hydra.yaml").write_text("")
    (dlio_cfg / "overrides.yaml").write_text("")
    return leaf


class TestValidateDatagenLeaf:
    """Stat-only completeness check of the ``datagen/<ts>/`` leaf.

    Mirrors the file/folder list already enforced by
    ``submission_checker`` rule §2.1.14 / §2.1.15 (see
    ``mlpstorage_py/submission_checker/constants.py:67-81``) but is
    reachable both from a post-datagen run-checker hook and from
    ``reports reportgen`` — the submission_checker itself is a
    separate tool the operator may never invoke.

    Contract: return a list of human-readable missing-item strings.
    Empty list means the leaf is complete. Never raises for a
    stat-only check.
    """

    def test_healthy_leaf_returns_empty_list(self, tmp_path):
        leaf = _make_healthy_leaf(tmp_path)
        assert validate_datagen_leaf(str(leaf)) == []

    def test_missing_stdout_log_reported(self, tmp_path):
        leaf = _make_healthy_leaf(tmp_path)
        (leaf / "training_datagen.stdout.log").unlink()
        missing = validate_datagen_leaf(str(leaf))
        assert any("stdout" in m for m in missing), missing

    def test_missing_metadata_reported(self, tmp_path):
        leaf = _make_healthy_leaf(tmp_path, ts="20260708_130000")
        (leaf / "training_20260708_130000_metadata.json").unlink()
        missing = validate_datagen_leaf(str(leaf))
        assert any("metadata" in m for m in missing), missing

    def test_missing_dlio_log_reported(self, tmp_path):
        leaf = _make_healthy_leaf(tmp_path)
        (leaf / "dlio.log").unlink()
        missing = validate_datagen_leaf(str(leaf))
        assert any("dlio.log" in m for m in missing), missing

    def test_missing_dlio_config_folder_reported(self, tmp_path):
        leaf = _make_healthy_leaf(tmp_path)
        # Remove the whole folder — expects folder-level missing message,
        # not per-file complaints.
        for entry in (leaf / "dlio_config").iterdir():
            entry.unlink()
        (leaf / "dlio_config").rmdir()
        missing = validate_datagen_leaf(str(leaf))
        assert any("dlio_config" in m for m in missing), missing

    def test_missing_config_yaml_inside_dlio_config_reported(self, tmp_path):
        leaf = _make_healthy_leaf(tmp_path)
        (leaf / "dlio_config" / "config.yaml").unlink()
        missing = validate_datagen_leaf(str(leaf))
        assert any("config.yaml" in m for m in missing), missing

    def test_leaf_does_not_exist_reports_leaf_missing(self, tmp_path):
        # If the leaf itself does not exist, the helper reports that
        # as a single top-level miss rather than N per-file misses.
        missing = validate_datagen_leaf(str(tmp_path / "does_not_exist"))
        assert len(missing) == 1
        assert "does_not_exist" in missing[0]


# --------------------------------------------------------------------------- #
# validate_run_leaf                                                           #
# --------------------------------------------------------------------------- #


def _make_healthy_run_leaf(root: pathlib.Path, ts: str = "20260710_095915") -> pathlib.Path:
    """Create a healthy training-run leaf: all required files + dlio_config."""
    leaf = root / ts
    leaf.mkdir(parents=True)
    (leaf / "training_run.stdout.log").write_text("")
    (leaf / "training_run.stderr.log").write_text("")
    (leaf / "dlio.log").write_text("")
    # RUN_REQUIRED_FILES uses the `.*<name>.json` regex family — the DLIO
    # training loop prefixes these with the workload+timestamp; drop a
    # representative name here so the regex matches.
    (leaf / f"unet3d_{ts}_output.json").write_text("{}")
    (leaf / f"unet3d_{ts}_per_epoch_stats.json").write_text("{}")
    (leaf / f"unet3d_{ts}_summary.json").write_text("{}")
    dlio_cfg = leaf / "dlio_config"
    dlio_cfg.mkdir()
    (dlio_cfg / "config.yaml").write_text("")
    (dlio_cfg / "hydra.yaml").write_text("")
    (dlio_cfg / "overrides.yaml").write_text("")
    return leaf


class TestValidateRunLeaf:
    """Stat-only completeness check of the ``training/<model>/run/<ts>/`` leaf.

    Mirrors ``TestValidateDatagenLeaf`` against ``RUN_REQUIRED_FILES`` /
    ``RUN_REQUIRED_FOLDERS`` (submission_checker/constants.py:83-93).
    Wired into ``TrainingBenchmark._run`` post-run so a DLIO subprocess
    that crashes silently surfaces at run time instead of at
    ``mlpstorage validate`` time (storage#761).
    """

    def test_healthy_leaf_returns_empty_list(self, tmp_path):
        leaf = _make_healthy_run_leaf(tmp_path)
        assert validate_run_leaf(str(leaf)) == []

    def test_missing_dlio_log_reported(self, tmp_path):
        # The exact signature of storage#761: DLIO crashed before
        # writing dlio.log — the wrapper stdout log is present but the
        # DLIO-native artifacts are not.
        leaf = _make_healthy_run_leaf(tmp_path)
        (leaf / "dlio.log").unlink()
        missing = validate_run_leaf(str(leaf))
        assert any("dlio.log" in m for m in missing), missing

    def test_missing_summary_json_reported(self, tmp_path):
        leaf = _make_healthy_run_leaf(tmp_path, ts="20260710_100000")
        (leaf / "unet3d_20260710_100000_summary.json").unlink()
        missing = validate_run_leaf(str(leaf))
        assert any("summary.json" in m for m in missing), missing

    def test_missing_output_json_reported(self, tmp_path):
        leaf = _make_healthy_run_leaf(tmp_path, ts="20260710_100000")
        (leaf / "unet3d_20260710_100000_output.json").unlink()
        missing = validate_run_leaf(str(leaf))
        assert any("output.json" in m for m in missing), missing

    def test_missing_per_epoch_stats_reported(self, tmp_path):
        leaf = _make_healthy_run_leaf(tmp_path, ts="20260710_100000")
        (leaf / "unet3d_20260710_100000_per_epoch_stats.json").unlink()
        missing = validate_run_leaf(str(leaf))
        assert any("per_epoch_stats.json" in m for m in missing), missing

    def test_missing_dlio_config_folder_reported(self, tmp_path):
        leaf = _make_healthy_run_leaf(tmp_path)
        for entry in (leaf / "dlio_config").iterdir():
            entry.unlink()
        (leaf / "dlio_config").rmdir()
        missing = validate_run_leaf(str(leaf))
        assert any("dlio_config" in m for m in missing), missing

    def test_all_dlio_artifacts_missing_reports_all(self, tmp_path):
        # The exact storage#761 scenario: DLIO never wrote anything.
        # The wrapper stdout/stderr are present but no dlio.log,
        # no *.json outputs, no dlio_config/. Expect the helper to
        # enumerate every miss rather than short-circuit after the first.
        leaf = tmp_path / "20260710_095915"
        leaf.mkdir(parents=True)
        (leaf / "training_run.stdout.log").write_text("")
        (leaf / "training_run.stderr.log").write_text("")
        missing = validate_run_leaf(str(leaf))
        # Expect: dlio.log, output.json, per_epoch_stats.json, summary.json, dlio_config/
        assert any("dlio.log" in m for m in missing), missing
        assert any("output.json" in m for m in missing), missing
        assert any("per_epoch_stats.json" in m for m in missing), missing
        assert any("summary.json" in m for m in missing), missing
        assert any("dlio_config" in m for m in missing), missing

    def test_leaf_does_not_exist_reports_leaf_missing(self, tmp_path):
        missing = validate_run_leaf(str(tmp_path / "does_not_exist"))
        assert len(missing) == 1
        assert "does_not_exist" in missing[0]
        assert "run" in missing[0]


# --------------------------------------------------------------------------- #
# validate_checkpoint_leaf                                                    #
# --------------------------------------------------------------------------- #


def _make_healthy_checkpoint_leaf(root: pathlib.Path, ts: str = "20260710_120000") -> pathlib.Path:
    """Create a healthy checkpointing-run leaf per CHECKPOINT_REQUIRED_FILES."""
    leaf = root / ts
    leaf.mkdir(parents=True)
    (leaf / "checkpointing_run.stdout.log").write_text("")
    (leaf / "checkpointing_run.stderr.log").write_text("")
    (leaf / "dlio.log").write_text("")
    (leaf / f"llama3_8b_{ts}_output.json").write_text("{}")
    (leaf / f"llama3_8b_{ts}_per_epoch_stats.json").write_text("{}")
    (leaf / f"llama3_8b_{ts}_summary.json").write_text("{}")
    dlio_cfg = leaf / "dlio_config"
    dlio_cfg.mkdir()
    (dlio_cfg / "config.yaml").write_text("")
    (dlio_cfg / "hydra.yaml").write_text("")
    (dlio_cfg / "overrides.yaml").write_text("")
    return leaf


class TestValidateCheckpointLeaf:
    """Stat-only completeness check of the checkpointing leaf.

    Mirrors ``TestValidateRunLeaf`` against
    ``CHECKPOINT_REQUIRED_FILES`` / ``CHECKPOINT_REQUIRED_FOLDERS``
    (submission_checker/constants.py:98-108). Wired into
    ``CheckpointingBenchmark._run`` — sibling of storage#761.
    """

    def test_healthy_leaf_returns_empty_list(self, tmp_path):
        leaf = _make_healthy_checkpoint_leaf(tmp_path)
        assert validate_checkpoint_leaf(str(leaf)) == []

    def test_missing_dlio_log_reported(self, tmp_path):
        leaf = _make_healthy_checkpoint_leaf(tmp_path)
        (leaf / "dlio.log").unlink()
        missing = validate_checkpoint_leaf(str(leaf))
        assert any("dlio.log" in m for m in missing), missing

    def test_missing_summary_json_reported(self, tmp_path):
        leaf = _make_healthy_checkpoint_leaf(tmp_path, ts="20260710_130000")
        (leaf / "llama3_8b_20260710_130000_summary.json").unlink()
        missing = validate_checkpoint_leaf(str(leaf))
        assert any("summary.json" in m for m in missing), missing

    def test_missing_dlio_config_folder_reported(self, tmp_path):
        leaf = _make_healthy_checkpoint_leaf(tmp_path)
        for entry in (leaf / "dlio_config").iterdir():
            entry.unlink()
        (leaf / "dlio_config").rmdir()
        missing = validate_checkpoint_leaf(str(leaf))
        assert any("dlio_config" in m for m in missing), missing

    def test_leaf_does_not_exist_reports_leaf_missing(self, tmp_path):
        missing = validate_checkpoint_leaf(str(tmp_path / "does_not_exist"))
        assert len(missing) == 1
        assert "checkpoint" in missing[0]


# --------------------------------------------------------------------------- #
# write_datagen_manifest                                                      #
# --------------------------------------------------------------------------- #


def _valid_dataset_params():
    """Realistic dataset param block matching configs/dlio/workload/unet3d_datagen.yaml."""
    return {
        "num_files_train": 168,
        "num_samples_per_file": 1,
        "record_length_bytes": 146600628,
    }


class TestWriteDatagenManifest:
    """Manifest emission at ``<data-dir>/<model>/.mlps-datagen-manifest.json``.

    Fields recorded (per user direction — only what the future
    run-vs-datagen size check needs plus provenance):

        schema_version         — integer, bumps if layout changes
        model                  — e.g. "unet3d"
        num_files_train        — from merged DLIO config
        num_samples_per_file   — from merged DLIO config
        record_length_bytes    — from merged DLIO config
        created_at             — ISO 8601 UTC (Z-suffixed)
        mlpstorage_version     — from mlpstorage_py.__version__
        source_datagen_result_dir — absolute path to the --results-dir leaf

    Loud failure if any of the three DLIO fields is absent from the
    input dict — the extraction is authoritative, and a config that
    lacks any of them is a workload-YAML bug (both unet3d_datagen.yaml
    and retinanet_datagen.yaml expose all three).
    """

    def test_writes_manifest_with_all_expected_fields(self, tmp_path):
        source_dir = "/tmp/whatever/training/unet3d/datagen/20260708_120000"
        manifest_path = write_datagen_manifest(
            data_dir=str(tmp_path),
            model="unet3d",
            dataset_params=_valid_dataset_params(),
            source_datagen_result_dir=source_dir,
        )
        # Return value: absolute path we wrote.
        assert manifest_path.endswith(
            os.path.join("unet3d", DATAGEN_MANIFEST_FILENAME)
        )
        assert os.path.isabs(manifest_path)
        assert os.path.isfile(manifest_path)

        with open(manifest_path) as f:
            data = json.load(f)

        assert data["schema_version"] == DATAGEN_MANIFEST_SCHEMA_VERSION
        assert data["model"] == "unet3d"
        assert data["num_files_train"] == 168
        assert data["num_samples_per_file"] == 1
        assert data["record_length_bytes"] == 146600628
        assert data["source_datagen_result_dir"] == source_dir
        # ISO 8601 UTC — the exact format string is trivial to verify,
        # but the important invariant is "parseable + Z-suffixed".
        assert isinstance(data["created_at"], str)
        assert data["created_at"].endswith("Z")
        # mlpstorage_version is captured — value not pinned here so
        # the test doesn't break on every version bump.
        assert isinstance(data["mlpstorage_version"], str)
        assert data["mlpstorage_version"]

    def test_writes_into_model_subdir(self, tmp_path):
        # Manifest lives at <data-dir>/<model>/.mlps-datagen-manifest.json —
        # NOT at the data-dir root. Per-model nesting lets one
        # --data-dir hold multiple datasets.
        write_datagen_manifest(
            data_dir=str(tmp_path),
            model="retinanet",
            dataset_params={
                "num_files_train": 1170301,
                "num_samples_per_file": 1,
                "record_length_bytes": 322957,
            },
            source_datagen_result_dir="/some/leaf",
        )
        assert (tmp_path / "retinanet" / DATAGEN_MANIFEST_FILENAME).is_file()
        # And NOT at the top level.
        assert not (tmp_path / DATAGEN_MANIFEST_FILENAME).exists()

    def test_creates_model_subdir_if_missing(self, tmp_path):
        # The refuse-to-overwrite guard fires BEFORE this helper —
        # by the time we get here, <data-dir>/<model> may not exist
        # yet (datagen itself hasn't populated it). The manifest
        # writer must create the directory as needed.
        write_datagen_manifest(
            data_dir=str(tmp_path),
            model="unet3d",
            dataset_params=_valid_dataset_params(),
            source_datagen_result_dir="/some/leaf",
        )
        assert (tmp_path / "unet3d").is_dir()

    @pytest.mark.parametrize(
        "missing_key",
        ["num_files_train", "num_samples_per_file", "record_length_bytes"],
    )
    def test_missing_required_field_raises_loudly(self, tmp_path, missing_key):
        params = _valid_dataset_params()
        del params[missing_key]
        with pytest.raises(ConfigurationError) as excinfo:
            write_datagen_manifest(
                data_dir=str(tmp_path),
                model="unet3d",
                dataset_params=params,
                source_datagen_result_dir="/some/leaf",
            )
        # Error must name the missing key so operators can see which
        # workload YAML is broken.
        assert missing_key in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Object-storage parity — refuse-to-overwrite + manifest PUT via s3dlio       #
# --------------------------------------------------------------------------- #

# All s3dlio interaction goes through the ``s3dlio`` module attribute
# on ``mlpstorage_py.rules.datagen_hierarchy``. Patching that attribute
# is how we simulate LIST / PUT behavior in these unit tests — no real
# object-storage endpoint is required. The maintainer confirmed
# depending on s3dlio's upstream unit tests for wire-level correctness
# is acceptable given (1) s3dlio is already exercised by the
# checkpointing benchmark's actual production I/O, and (2) our sentinel
# operations are a ~1 KB PUT and a bounded LIST — orders of magnitude
# simpler than what s3dlio already carries in this codebase.


@pytest.mark.parametrize(
    "uri",
    [
        "s3://bucket/prefix",
        "s3://bucket/prefix/",
        "gs://bucket/prefix",
        "az://container/prefix",
        "direct://bucket/prefix",
    ],
)
class TestAssertDataDirHierarchyAbsentObjectStorage:
    """Refuse-to-overwrite parity for every URI scheme s3dlio abstracts.

    Semantic contract must match the local case:
        - No object under <data-dir>/<model>/ → clean return.
        - Any object present → ``ConfigurationError`` with the offending
          URI in the message.
        - Any s3dlio exception during LIST → ``ConfigurationError``
          (fail-safe abort per the design decision — a transient
          object-store error must not silently be treated as
          "assume empty, proceed").
    """

    def test_empty_prefix_returns_cleanly(self, uri):
        with patch(
            "mlpstorage_py.rules.datagen_hierarchy.s3dlio"
        ) as mock_s3dlio:
            mock_s3dlio.list.return_value = []
            # Must not raise.
            assert_data_dir_hierarchy_absent(uri, "unet3d")

            # LIST must be scoped to <data-dir>/<model>/ (with model
            # segment appended), NOT to the raw data-dir prefix —
            # otherwise a data-dir that hosts sibling datasets
            # (retinanet, etc.) would be misdetected as populated.
            (called_uri,), _ = mock_s3dlio.list.call_args
            assert "unet3d" in called_uri
            assert called_uri.startswith(uri.rstrip("/"))

    def test_bounded_list_uses_non_recursive(self, uri):
        # s3dlio.list has no MaxKeys=1 knob. To keep the check
        # bounded regardless of dataset size we pass
        # recursive=False — a delimiter-based listing that returns
        # only the small set of direct children (train/, valid/,
        # test/ + manifest). Documented in the design conversation
        # and PR discussion.
        with patch(
            "mlpstorage_py.rules.datagen_hierarchy.s3dlio"
        ) as mock_s3dlio:
            mock_s3dlio.list.return_value = []
            assert_data_dir_hierarchy_absent(uri, "unet3d")
            _, kwargs = mock_s3dlio.list.call_args
            assert kwargs.get("recursive") is False, (
                f"Expected recursive=False for bounded LIST; got kwargs={kwargs!r}"
            )

    def test_non_empty_prefix_raises_with_uri_in_message(self, uri):
        with patch(
            "mlpstorage_py.rules.datagen_hierarchy.s3dlio"
        ) as mock_s3dlio:
            mock_s3dlio.list.return_value = [
                f"{uri.rstrip('/')}/unet3d/train/shard_000.npz"
            ]
            with pytest.raises(ConfigurationError) as excinfo:
                assert_data_dir_hierarchy_absent(uri, "unet3d")
            text = str(excinfo.value)
            # Must name the offending model URI so the operator can
            # act (rm-rf equivalent or a different --data-dir).
            assert "unet3d" in text
            assert "s3://" in text or "gs://" in text or "az://" in text or "direct://" in text

    def test_s3dlio_exception_fails_safe_abort(self, uri):
        # Per the design decision: a transient LIST failure must NOT
        # be treated as "assume empty". Any s3dlio exception during
        # the refuse-to-overwrite check is a ConfigurationError.
        with patch(
            "mlpstorage_py.rules.datagen_hierarchy.s3dlio"
        ) as mock_s3dlio:
            mock_s3dlio.list.side_effect = RuntimeError(
                "network failure or credentials missing"
            )
            with pytest.raises(ConfigurationError) as excinfo:
                assert_data_dir_hierarchy_absent(uri, "unet3d")
            # The underlying error must surface (chained or in the
            # message) so the operator can diagnose.
            text = str(excinfo.value)
            assert "unet3d" in text
            # Chained exception should preserve the original.
            assert excinfo.value.__cause__ is not None


class TestWriteDatagenManifestObjectStorage:
    """Manifest PUT via s3dlio.put_bytes for object-storage --data-dir.

    Feature parity with the local ``open(..., 'w')`` path: the JSON
    body is byte-identical (schema_version, three DLIO knobs,
    provenance fields) — only the write mechanism differs.
    """

    def test_writes_manifest_via_put_bytes_at_expected_uri(self):
        with patch(
            "mlpstorage_py.rules.datagen_hierarchy.s3dlio"
        ) as mock_s3dlio:
            returned_uri = write_datagen_manifest(
                data_dir="s3://mybucket/mytraining",
                model="unet3d",
                dataset_params=_valid_dataset_params(),
                source_datagen_result_dir="/results/leaf",
            )

            # The manifest lands under <data-dir>/<model>/<filename>
            # (matches local per-model nesting so one bucket can hold
            # multiple model datasets).
            expected_uri = (
                f"s3://mybucket/mytraining/unet3d/{DATAGEN_MANIFEST_FILENAME}"
            )
            assert returned_uri == expected_uri
            mock_s3dlio.put_bytes.assert_called_once()
            (called_uri, payload), _ = mock_s3dlio.put_bytes.call_args
            assert called_uri == expected_uri
            # Payload must be bytes-typed — s3dlio.put_bytes contract.
            assert isinstance(payload, (bytes, bytearray))

    def test_manifest_body_matches_local_schema(self):
        # The wire body must be byte-identical to what the local
        # branch would write — the future run-side reader is the
        # same code path across storage backends.
        with patch(
            "mlpstorage_py.rules.datagen_hierarchy.s3dlio"
        ) as mock_s3dlio:
            write_datagen_manifest(
                data_dir="s3://mybucket/mytraining",
                model="unet3d",
                dataset_params=_valid_dataset_params(),
                source_datagen_result_dir="/results/leaf",
            )
            (_, payload), _ = mock_s3dlio.put_bytes.call_args
            data = json.loads(payload.decode("utf-8"))
            assert data["schema_version"] == DATAGEN_MANIFEST_SCHEMA_VERSION
            assert data["model"] == "unet3d"
            assert data["num_files_train"] == 168
            assert data["num_samples_per_file"] == 1
            assert data["record_length_bytes"] == 146600628
            assert data["source_datagen_result_dir"] == "/results/leaf"
            assert data["created_at"].endswith("Z")
            assert data["mlpstorage_version"]

    @pytest.mark.parametrize(
        "missing_key",
        ["num_files_train", "num_samples_per_file", "record_length_bytes"],
    )
    def test_missing_field_raises_before_put(self, missing_key):
        # The loud-failure contract applies to object storage too.
        # No PUT must fire if extraction fails — otherwise a
        # zero-valued placeholder could pollute the object store.
        with patch(
            "mlpstorage_py.rules.datagen_hierarchy.s3dlio"
        ) as mock_s3dlio:
            params = _valid_dataset_params()
            del params[missing_key]
            with pytest.raises(ConfigurationError):
                write_datagen_manifest(
                    data_dir="s3://mybucket/mytraining",
                    model="unet3d",
                    dataset_params=params,
                    source_datagen_result_dir="/results/leaf",
                )
            mock_s3dlio.put_bytes.assert_not_called()

    def test_put_bytes_exception_propagates_configuration_error(self):
        # s3dlio raises on PUT (auth failure, network hiccup, etc.).
        # Wrap into ConfigurationError so main.py's uniform error
        # rendering surfaces it — silent PUT failure would leave the
        # dataset without its manifest and break the future
        # run-vs-datagen check.
        with patch(
            "mlpstorage_py.rules.datagen_hierarchy.s3dlio"
        ) as mock_s3dlio:
            mock_s3dlio.put_bytes.side_effect = RuntimeError("PUT failed")
            with pytest.raises(ConfigurationError) as excinfo:
                write_datagen_manifest(
                    data_dir="s3://mybucket/mytraining",
                    model="unet3d",
                    dataset_params=_valid_dataset_params(),
                    source_datagen_result_dir="/results/leaf",
                )
            assert excinfo.value.__cause__ is not None
