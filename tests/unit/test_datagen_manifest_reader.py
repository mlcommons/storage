"""Datagen manifest reader + additive writer fields (storage#571 Q4, datagen-manifest v1.0 PR-1).

The writer (PR #732) emits ``<data-dir>/<model>/.mlps-datagen-manifest.json``
after a successful ``training datagen``. Until now nothing read it. These tests
pin the read side and the three additive v1 fields the writer gains so the
run-side check can reproduce the generated layout:

* ``num_subfolders_train`` — needed to reconstruct subfoldered names under
  ``skip_listing`` (DLIO ``utils/skip_listing.py``).
* ``dataset_format`` — an invariant the run must match.
* ``rules_edition`` — provenance (``config.RULES_EDITION``).

Contract (Curtis 2026-09-21, D1/D3): schema_version stays 1 — the extension is
additive and the reader tolerates pre-extension manifests (fields read back as
``None``). A missing manifest is ``None`` (the caller warns and falls back to
today's behaviour); a manifest that exists but cannot be read or parsed is a
``ConfigurationError`` — silence there would hide a real problem (VFY-06).
Object storage reads use ``s3dlio.exists`` (HEAD) to tell "not present" from
"read failed" before the single ``s3dlio.get`` (OBJ-04 / OBJ-06).
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from mlpstorage_py.config import RULES_EDITION
from mlpstorage_py.errors import ConfigurationError, ErrorCode
from mlpstorage_py.rules.datagen_hierarchy import (
    DATAGEN_MANIFEST_FILENAME,
    DATAGEN_MANIFEST_SCHEMA_VERSION,
    DatagenManifest,
    datagen_manifest_location,
    read_datagen_manifest,
    write_datagen_manifest,
)

_S3DLIO = "mlpstorage_py.rules.datagen_hierarchy.s3dlio"


def _params(**over):
    base = {
        "num_files_train": 168,
        "num_samples_per_file": 1,
        "record_length_bytes": 146600628,
        "format": "npz",
    }
    base.update(over)
    return base


def _v1_pre_extension_body(**over):
    """The exact eight-key body PR #732 wrote before the additive fields."""
    body = {
        "schema_version": 1,
        "model": "unet3d",
        "num_files_train": 168,
        "num_samples_per_file": 1,
        "record_length_bytes": 146600628,
        "created_at": "2026-07-08T00:00:00Z",
        "mlpstorage_version": "3.0.0",
        "source_datagen_result_dir": "/results/leaf",
    }
    body.update(over)
    return body


# --------------------------------------------------------------------------- #
# Writer: additive fields                                                     #
# --------------------------------------------------------------------------- #


class TestWriterAdditiveFields:
    def test_local_manifest_carries_layout_and_provenance_fields(self, tmp_path):
        path = write_datagen_manifest(
            data_dir=str(tmp_path), model="unet3d",
            dataset_params=_params(num_subfolders_train=8),
            source_datagen_result_dir="/results/leaf",
        )
        data = json.loads(open(path).read())
        # Additive: schema version does NOT bump (pre-extension readers keep working).
        assert data["schema_version"] == DATAGEN_MANIFEST_SCHEMA_VERSION == 1
        assert data["num_subfolders_train"] == 8
        assert data["dataset_format"] == "npz"
        assert data["rules_edition"] == RULES_EDITION

    def test_num_subfolders_defaults_to_zero_when_absent(self, tmp_path):
        # DLIO's default (utils/config.py num_subfolders_train: int = 0);
        # the workload YAMLs don't set it, datasize's hint may.
        path = write_datagen_manifest(
            data_dir=str(tmp_path), model="unet3d",
            dataset_params=_params(), source_datagen_result_dir="/r",
        )
        assert json.loads(open(path).read())["num_subfolders_train"] == 0

    def test_dataset_format_is_null_when_absent(self, tmp_path):
        params = _params()
        del params["format"]
        path = write_datagen_manifest(
            data_dir=str(tmp_path), model="unet3d",
            dataset_params=params, source_datagen_result_dir="/r",
        )
        assert json.loads(open(path).read())["dataset_format"] is None

    def test_object_manifest_carries_the_same_fields(self):
        with patch(_S3DLIO) as s3:
            write_datagen_manifest(
                data_dir="s3://bucket/prefix", model="unet3d",
                dataset_params=_params(num_subfolders_train=3),
                source_datagen_result_dir="/r",
            )
            (_, payload), _ = s3.put_bytes.call_args
        data = json.loads(payload.decode("utf-8"))
        assert data["num_subfolders_train"] == 3
        assert data["dataset_format"] == "npz"
        assert data["rules_edition"] == RULES_EDITION


# --------------------------------------------------------------------------- #
# Location                                                                    #
# --------------------------------------------------------------------------- #


class TestManifestLocation:
    def test_local(self, tmp_path):
        assert datagen_manifest_location(str(tmp_path), "unet3d") == os.path.join(
            str(tmp_path), "unet3d", DATAGEN_MANIFEST_FILENAME)

    @pytest.mark.parametrize("data_dir", ["s3://b/p", "s3://b/p/"])
    def test_object_normalises_trailing_slash(self, data_dir):
        assert datagen_manifest_location(data_dir, "unet3d") == (
            f"s3://b/p/unet3d/{DATAGEN_MANIFEST_FILENAME}")

    def test_manifest_is_a_sibling_of_train_and_valid(self, tmp_path):
        """MAN-04 pin: DLIO lists only ``<data_folder>/{train,valid}/`` (and
        with skip_listing forced on, nothing at all). The manifest must sit
        one level up so no lister — DLIO's or a user's ``ls train/`` — ever
        counts it as a sample."""
        model_dir = tmp_path / "unet3d"
        (model_dir / "train").mkdir(parents=True)
        (model_dir / "valid").mkdir()
        (model_dir / "train" / "img_0_of_1.npz").write_bytes(b"x")
        write_datagen_manifest(str(tmp_path), "unet3d", _params(), "/r")
        assert DATAGEN_MANIFEST_FILENAME in os.listdir(model_dir)
        assert os.listdir(model_dir / "train") == ["img_0_of_1.npz"]
        assert os.listdir(model_dir / "valid") == []


# --------------------------------------------------------------------------- #
# Reader: local filesystem                                                    #
# --------------------------------------------------------------------------- #


class TestReadLocal:
    def test_absent_data_dir_is_none(self, tmp_path):
        assert read_datagen_manifest(str(tmp_path / "nope"), "unet3d") is None

    def test_model_dir_without_manifest_is_none(self, tmp_path):
        (tmp_path / "unet3d" / "train").mkdir(parents=True)
        assert read_datagen_manifest(str(tmp_path), "unet3d") is None

    def test_round_trip_local(self, tmp_path):
        path = write_datagen_manifest(
            data_dir=str(tmp_path), model="retinanet",
            dataset_params=_params(num_files_train=257_173, record_length_bytes=322957,
                                   format="jpeg", num_subfolders_train=16),
            source_datagen_result_dir="/results/leaf",
        )
        m = read_datagen_manifest(str(tmp_path), "retinanet")
        assert isinstance(m, DatagenManifest)
        assert m.location == path
        assert m.schema_version == 1
        assert m.model == "retinanet"
        assert m.num_files_train == 257_173
        assert m.num_samples_per_file == 1
        assert m.record_length_bytes == 322957
        assert m.num_subfolders_train == 16
        assert m.dataset_format == "jpeg"
        assert m.rules_edition == RULES_EDITION
        assert m.source_datagen_result_dir == "/results/leaf"
        assert m.created_at.endswith("Z")
        assert m.mlpstorage_version

    def test_pre_extension_manifest_reads_with_none_for_new_fields(self, tmp_path):
        (tmp_path / "unet3d").mkdir()
        (tmp_path / "unet3d" / DATAGEN_MANIFEST_FILENAME).write_text(
            json.dumps(_v1_pre_extension_body()))
        m = read_datagen_manifest(str(tmp_path), "unet3d")
        assert m.num_files_train == 168
        assert m.num_subfolders_train is None
        assert m.dataset_format is None
        assert m.rules_edition is None

    def test_counts_are_coerced_to_int(self, tmp_path):
        (tmp_path / "unet3d").mkdir()
        (tmp_path / "unet3d" / DATAGEN_MANIFEST_FILENAME).write_text(
            json.dumps(_v1_pre_extension_body(num_files_train="168", num_subfolders_train="4")))
        m = read_datagen_manifest(str(tmp_path), "unet3d")
        assert m.num_files_train == 168 and isinstance(m.num_files_train, int)
        assert m.num_subfolders_train == 4

    def test_corrupt_json_raises_parse_error_naming_the_path(self, tmp_path):
        (tmp_path / "unet3d").mkdir()
        p = tmp_path / "unet3d" / DATAGEN_MANIFEST_FILENAME
        p.write_text("{not json")
        with pytest.raises(ConfigurationError) as ei:
            read_datagen_manifest(str(tmp_path), "unet3d")
        assert ei.value.code == ErrorCode.CONFIG_PARSE_ERROR
        assert str(p) in str(ei.value)

    @pytest.mark.parametrize("missing", ["num_files_train", "num_samples_per_file",
                                         "record_length_bytes", "model", "schema_version"])
    def test_missing_required_key_raises(self, tmp_path, missing):
        (tmp_path / "unet3d").mkdir()
        body = _v1_pre_extension_body()
        del body[missing]
        (tmp_path / "unet3d" / DATAGEN_MANIFEST_FILENAME).write_text(json.dumps(body))
        with pytest.raises(ConfigurationError) as ei:
            read_datagen_manifest(str(tmp_path), "unet3d")
        assert missing in str(ei.value)

    def test_non_object_json_raises(self, tmp_path):
        (tmp_path / "unet3d").mkdir()
        (tmp_path / "unet3d" / DATAGEN_MANIFEST_FILENAME).write_text("[1, 2, 3]")
        with pytest.raises(ConfigurationError):
            read_datagen_manifest(str(tmp_path), "unet3d")

    def test_unsupported_schema_version_raises(self, tmp_path):
        (tmp_path / "unet3d").mkdir()
        (tmp_path / "unet3d" / DATAGEN_MANIFEST_FILENAME).write_text(
            json.dumps(_v1_pre_extension_body(schema_version=2)))
        with pytest.raises(ConfigurationError) as ei:
            read_datagen_manifest(str(tmp_path), "unet3d")
        assert "schema_version" in str(ei.value)

    def test_unreadable_manifest_raises_not_none(self, tmp_path):
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            pytest.skip("root ignores file modes")
        (tmp_path / "unet3d").mkdir()
        p = tmp_path / "unet3d" / DATAGEN_MANIFEST_FILENAME
        p.write_text(json.dumps(_v1_pre_extension_body()))
        p.chmod(0)
        try:
            with pytest.raises(ConfigurationError) as ei:
                read_datagen_manifest(str(tmp_path), "unet3d")
            assert ei.value.__cause__ is not None
        finally:
            p.chmod(0o644)


# --------------------------------------------------------------------------- #
# Reader: object storage                                                      #
# --------------------------------------------------------------------------- #


class TestReadObject:
    URI = f"s3://bucket/prefix/unet3d/{DATAGEN_MANIFEST_FILENAME}"

    def test_absent_object_is_none_without_a_get(self):
        with patch(_S3DLIO) as s3:
            s3.exists.return_value = False
            assert read_datagen_manifest("s3://bucket/prefix", "unet3d") is None
            s3.exists.assert_called_once_with(self.URI)
            s3.get.assert_not_called()

    def test_present_object_is_read_with_a_single_get(self):
        with patch(_S3DLIO) as s3:
            s3.exists.return_value = True
            s3.get.return_value = json.dumps(_v1_pre_extension_body()).encode()
            m = read_datagen_manifest("s3://bucket/prefix/", "unet3d")
            s3.get.assert_called_once_with(self.URI)
        assert m.location == self.URI
        assert m.num_files_train == 168

    def test_get_result_is_coerced_to_bytes(self):
        # s3dlio.get returns a BytesView (buffer protocol); DLIO's own
        # callers wrap it in bytes(...). Anything buffer-like must parse.
        with patch(_S3DLIO) as s3:
            s3.exists.return_value = True
            s3.get.return_value = bytearray(json.dumps(_v1_pre_extension_body()).encode())
            assert read_datagen_manifest("s3://bucket/prefix", "unet3d").model == "unet3d"

    def test_exists_failure_is_an_error_not_absence(self):
        # Fail-safe like assert_data_dir_hierarchy_absent: a HEAD that
        # errors (credentials, network) must not read as "no manifest".
        with patch(_S3DLIO) as s3:
            s3.exists.side_effect = RuntimeError("403")
            with pytest.raises(ConfigurationError) as ei:
                read_datagen_manifest("s3://bucket/prefix", "unet3d")
            assert ei.value.__cause__ is not None
            assert self.URI in str(ei.value)

    def test_get_failure_is_an_error(self):
        with patch(_S3DLIO) as s3:
            s3.exists.return_value = True
            s3.get.side_effect = RuntimeError("timeout")
            with pytest.raises(ConfigurationError) as ei:
                read_datagen_manifest("s3://bucket/prefix", "unet3d")
            assert ei.value.__cause__ is not None

    def test_corrupt_object_body_raises_parse_error(self):
        with patch(_S3DLIO) as s3:
            s3.exists.return_value = True
            s3.get.return_value = b"\x00\xff not json"
            with pytest.raises(ConfigurationError) as ei:
                read_datagen_manifest("s3://bucket/prefix", "unet3d")
            assert ei.value.code == ErrorCode.CONFIG_PARSE_ERROR

    def test_round_trip_object(self):
        """OBJ-07 (mocked transport): the bytes put_bytes wrote come back
        through get and parse to the same manifest the local path yields."""
        with patch(_S3DLIO) as s3:
            write_datagen_manifest(
                data_dir="s3://bucket/prefix", model="unet3d",
                dataset_params=_params(num_subfolders_train=2),
                source_datagen_result_dir="/r",
            )
            (put_uri, payload), _ = s3.put_bytes.call_args
            s3.exists.return_value = True
            s3.get.return_value = payload
            m = read_datagen_manifest("s3://bucket/prefix", "unet3d")
        assert put_uri == self.URI == m.location
        assert (m.num_files_train, m.num_subfolders_train, m.dataset_format) == (168, 2, "npz")
