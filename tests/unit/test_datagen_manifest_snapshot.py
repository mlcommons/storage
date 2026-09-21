"""The run-leaf datagen-manifest snapshot (datagen-manifest v1.0, D4 linkage).

``training run`` copies the manifest it consumed from
``<data-dir>/<model>/.mlps-datagen-manifest.json`` into its own results leaf
as ``<leaf>/datagen-manifest.json`` so that ``mlpstorage validate`` can tie
the run to the ``datagen/`` leaf that produced the data (rule 3.3.1
``MANIFEST-*`` tokens) without ever touching ``--data-dir`` — the validator
never lists or reads a submitter's data directory (B-04 hard constraint).

On-disk shape (``mlps-datagen-manifest-snapshot/1``)::

    {
      "schema": "mlps-datagen-manifest-snapshot/1",
      "consumed_at": "<ISO 8601 UTC, Z suffix>",
      "location": "<where the manifest was read from>",
      "overridden": ["MANIFEST-003"],        # findings --skip-validation waived
      "manifest": { ...the v1 body verbatim... }
    }

Absent optional v1 fields (pre-extension manifests) are omitted from the
``manifest`` block, never written as ``null``. Reading a snapshot re-uses the
v1 manifest parser, so a corrupt snapshot is as loud as a corrupt manifest.
"""

from __future__ import annotations

import datetime
import json
import os

import pytest

from mlpstorage_py.errors import ConfigurationError
from mlpstorage_py.rules.datagen_hierarchy import (
    DATAGEN_MANIFEST_SNAPSHOT_FILENAME,
    DATAGEN_MANIFEST_SNAPSHOT_SCHEMA,
    METADATA_DATAGEN_MANIFEST_KEY,
    DatagenManifest,
    DatagenManifestSnapshot,
    datagen_manifest_body,
    read_datagen_manifest_snapshot,
    write_datagen_manifest_snapshot,
)


def _manifest(**over):
    base = dict(
        location="/data/unet3d/.mlps-datagen-manifest.json",
        schema_version=1, model="unet3d",
        num_files_train=1_000, num_samples_per_file=1, record_length_bytes=146600628,
        created_at="2026-09-21T00:00:00Z", mlpstorage_version="3.0.46",
        source_datagen_result_dir="/results/closed/Acme/results/sys/training/unet3d/datagen/20260921_100000",
        num_subfolders_train=0, dataset_format="npz", rules_edition="3.0",
    )
    base.update(over)
    return DatagenManifest(**base)


class TestConstants:
    def test_names_are_the_contract(self):
        assert DATAGEN_MANIFEST_SNAPSHOT_FILENAME == "datagen-manifest.json"
        assert DATAGEN_MANIFEST_SNAPSHOT_SCHEMA == "mlps-datagen-manifest-snapshot/1"
        assert METADATA_DATAGEN_MANIFEST_KEY == "datagen_manifest_file"


class TestBody:
    def test_body_is_the_v1_shape_without_location(self):
        body = datagen_manifest_body(_manifest())
        assert list(body) == [
            "schema_version", "model", "num_files_train", "num_samples_per_file",
            "record_length_bytes", "num_subfolders_train", "dataset_format",
            "rules_edition", "created_at", "mlpstorage_version",
            "source_datagen_result_dir",
        ]
        assert "location" not in body
        assert body["num_files_train"] == 1_000

    def test_pre_extension_fields_are_omitted_not_null(self):
        body = datagen_manifest_body(_manifest(
            num_subfolders_train=None, dataset_format=None, rules_edition=None,
            created_at=None, mlpstorage_version=None, source_datagen_result_dir=None))
        assert list(body) == ["schema_version", "model", "num_files_train",
                              "num_samples_per_file", "record_length_bytes"]
        assert None not in body.values()


class TestWrite:
    def test_writes_snapshot_with_fixed_key_order(self, tmp_path):
        now = datetime.datetime(2026, 9, 21, 18, 30, 0, tzinfo=datetime.timezone.utc)
        path = write_datagen_manifest_snapshot(str(tmp_path), _manifest(), now=now)
        assert path == os.path.join(str(tmp_path), DATAGEN_MANIFEST_SNAPSHOT_FILENAME)
        data = json.loads(open(path, encoding="utf-8").read())
        assert list(data) == ["schema", "consumed_at", "location", "overridden", "manifest"]
        assert data["schema"] == DATAGEN_MANIFEST_SNAPSHOT_SCHEMA
        assert data["consumed_at"] == "2026-09-21T18:30:00Z"
        assert data["location"] == "/data/unet3d/.mlps-datagen-manifest.json"
        assert data["overridden"] == []
        assert data["manifest"] == datagen_manifest_body(_manifest())

    def test_overridden_is_recorded(self, tmp_path):
        write_datagen_manifest_snapshot(str(tmp_path), _manifest(),
                                        overridden=["MANIFEST-003"])
        data = json.loads((tmp_path / DATAGEN_MANIFEST_SNAPSHOT_FILENAME).read_text())
        assert data["overridden"] == ["MANIFEST-003"]

    def test_no_tmp_file_left_behind(self, tmp_path):
        write_datagen_manifest_snapshot(str(tmp_path), _manifest())
        assert sorted(os.listdir(tmp_path)) == [DATAGEN_MANIFEST_SNAPSHOT_FILENAME]

    def test_overwrites_a_previous_snapshot(self, tmp_path):
        write_datagen_manifest_snapshot(str(tmp_path), _manifest(num_files_train=5))
        write_datagen_manifest_snapshot(str(tmp_path), _manifest(num_files_train=7))
        assert read_datagen_manifest_snapshot(str(tmp_path)).manifest.num_files_train == 7

    def test_missing_leaf_dir_raises(self, tmp_path):
        with pytest.raises(OSError):
            write_datagen_manifest_snapshot(str(tmp_path / "nope"), _manifest())


class TestRead:
    def test_round_trip(self, tmp_path):
        write_datagen_manifest_snapshot(str(tmp_path), _manifest(),
                                        overridden=["MANIFEST-001"])
        snap = read_datagen_manifest_snapshot(str(tmp_path))
        assert isinstance(snap, DatagenManifestSnapshot)
        assert snap.path == os.path.join(str(tmp_path), DATAGEN_MANIFEST_SNAPSHOT_FILENAME)
        assert snap.consumed_at.endswith("Z")
        assert snap.overridden == ["MANIFEST-001"]
        assert snap.manifest.location == "/data/unet3d/.mlps-datagen-manifest.json"
        assert snap.manifest.num_files_train == 1_000
        assert snap.manifest.source_datagen_result_dir.endswith("/datagen/20260921_100000")
        assert snap.manifest.rules_edition == "3.0"

    def test_pre_extension_manifest_round_trips_as_none(self, tmp_path):
        write_datagen_manifest_snapshot(str(tmp_path), _manifest(
            num_subfolders_train=None, dataset_format=None, rules_edition=None))
        snap = read_datagen_manifest_snapshot(str(tmp_path))
        assert snap.manifest.num_subfolders_train is None
        assert snap.manifest.dataset_format is None
        assert snap.manifest.rules_edition is None

    def test_absent_is_none(self, tmp_path):
        assert read_datagen_manifest_snapshot(str(tmp_path)) is None

    def test_corrupt_json_is_an_error(self, tmp_path):
        (tmp_path / DATAGEN_MANIFEST_SNAPSHOT_FILENAME).write_text("{not json")
        with pytest.raises(ConfigurationError, match="not valid JSON"):
            read_datagen_manifest_snapshot(str(tmp_path))

    def test_wrong_schema_is_an_error(self, tmp_path):
        (tmp_path / DATAGEN_MANIFEST_SNAPSHOT_FILENAME).write_text(json.dumps({
            "schema": "mlps-datagen-manifest-snapshot/9", "consumed_at": "x",
            "location": "y", "overridden": [], "manifest": {}}))
        with pytest.raises(ConfigurationError, match="snapshot/9"):
            read_datagen_manifest_snapshot(str(tmp_path))

    def test_bad_inner_manifest_is_an_error(self, tmp_path):
        (tmp_path / DATAGEN_MANIFEST_SNAPSHOT_FILENAME).write_text(json.dumps({
            "schema": DATAGEN_MANIFEST_SNAPSHOT_SCHEMA, "consumed_at": "x",
            "location": "y", "overridden": [], "manifest": {"schema_version": 1}}))
        with pytest.raises(ConfigurationError, match="missing required key"):
            read_datagen_manifest_snapshot(str(tmp_path))

    def test_not_an_object_is_an_error(self, tmp_path):
        (tmp_path / DATAGEN_MANIFEST_SNAPSHOT_FILENAME).write_text("[]")
        with pytest.raises(ConfigurationError):
            read_datagen_manifest_snapshot(str(tmp_path))
