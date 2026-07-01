"""Tests for VectorDB-aware results directory validation."""

import json
from pathlib import Path

from mlpstorage_py.reporting.directory_validator import (
    ResultsDirectoryValidator,
    discover_scan_roots,
)


def _write_metadata(run_dir: Path, benchmark_type: str = "vector_database") -> Path:
    """Create the minimum metadata file recognized by the validator."""
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = run_dir.name
    metadata_path = run_dir / f"{benchmark_type}_{timestamp}_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "benchmark_type": benchmark_type,
                "run_datetime": timestamp,
                "result_dir": str(run_dir),
            }
        )
    )
    return run_dir


def _validate(results_dir: Path):
    return ResultsDirectoryValidator(str(results_dir)).validate()


class TestVectorDBDirectoryLayouts:
    """The validator accepts old and new VectorDB directory layouts."""

    def test_accepts_index_aware_layout(self, tmp_path):
        _write_metadata(
            tmp_path
            / "vector_database"
            / "milvus"
            / "DISKANN"
            / "run"
            / "20250115_160000"
        )

        result = _validate(tmp_path)

        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.found_benchmark_types == {"vector_database"}
        assert result.found_runs == 1

    def test_accepts_pre_engine_legacy_layout(self, tmp_path):
        _write_metadata(
            tmp_path
            / "vector_database"
            / "run"
            / "20250115_160000"
        )

        result = _validate(tmp_path)

        assert result.errors == []
        assert result.warnings == []
        assert result.found_runs == 1

    def test_accepts_engine_only_pr442_layout(self, tmp_path):
        _write_metadata(
            tmp_path
            / "vector_database"
            / "milvus"
            / "run"
            / "20250115_160000"
        )

        result = _validate(tmp_path)

        assert result.errors == []
        assert result.warnings == []
        assert result.found_runs == 1

    def test_accepts_multiple_engines_indexes_and_commands(self, tmp_path):
        _write_metadata(
            tmp_path
            / "vector_database"
            / "milvus"
            / "DISKANN"
            / "datagen"
            / "20250115_160000"
        )
        _write_metadata(
            tmp_path
            / "vector_database"
            / "milvus"
            / "HNSW"
            / "run"
            / "20250115_160100"
        )
        _write_metadata(
            tmp_path
            / "vector_database"
            / "elasticsearch"
            / "HNSW"
            / "run"
            / "20250115_160200"
        )

        result = _validate(tmp_path)

        assert result.errors == []
        assert result.warnings == []
        assert result.found_runs == 3

    def test_missing_metadata_is_reported_in_index_aware_layout(self, tmp_path):
        run_dir = (
            tmp_path
            / "vector_database"
            / "milvus"
            / "DISKANN"
            / "run"
            / "20250115_160000"
        )
        run_dir.mkdir(parents=True)

        result = _validate(tmp_path)

        assert result.found_runs == 0
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "malformed"
        assert "Missing metadata file" in result.errors[0].message
        assert result.errors[0].path == str(run_dir)

    def test_empty_index_directory_produces_actionable_warning(self, tmp_path):
        index_dir = tmp_path / "vector_database" / "milvus" / "DISKANN"
        index_dir.mkdir(parents=True)

        result = _validate(tmp_path)

        assert result.found_runs == 0
        assert any(
            "VectorDB index directory" in warning and "is empty" in warning
            for warning in result.warnings
        )
        assert any(
            "No valid VectorDB run directories" in warning
            for warning in result.warnings
        )

    def test_unexpected_directory_below_index_is_warned(self, tmp_path):
        unexpected = (
            tmp_path
            / "vector_database"
            / "milvus"
            / "DISKANN"
            / "not-a-command"
        )
        unexpected.mkdir(parents=True)

        result = _validate(tmp_path)

        assert result.found_runs == 0
        assert any(
            "Unexpected directory in VectorDB index directory" in warning
            and "not-a-command" in warning
            for warning in result.warnings
        )


class TestDirectoryValidatorRegressionCoverage:
    """VectorDB specialization does not change generic benchmark handling."""

    def test_training_command_layout_still_validates(self, tmp_path):
        run_dir = _write_metadata(
            tmp_path
            / "training"
            / "unet3d"
            / "run"
            / "20250115_143022",
            benchmark_type="training",
        )
        (run_dir / "summary.json").write_text("{}")

        result = _validate(tmp_path)

        assert result.errors == []
        assert result.warnings == []
        assert result.found_runs == 1

    def test_help_documents_preferred_and_compatible_vdb_layouts(self, tmp_path):
        validator = ResultsDirectoryValidator(str(tmp_path))

        help_text = validator.get_expected_structure_help()

        assert "milvus" in help_text
        assert "DISKANN" in help_text
        assert "vector_database/<command>/<datetime>/" in help_text
        assert "vector_database/<engine>/<command>/<datetime>/" in help_text


class TestDiscoverScanRoots:
    """Issue #599 bug 1+3: discover_scan_roots maps a sentinel-bearing
    submission root + (orgname, systemname) to the per-mode results slices
    that actually contain runs — so reportgen validates and walks only the
    requested system's subtree under the canonical layout that
    `mlpstorage init` / `<bench> run` / `validate` produce.

    Without orgname or systemname the helper passes the input path through
    unchanged so legacy flat-layout callers continue to work."""

    def _make_canonical_slice(self, tmp_path: Path, mode: str,
                              orgname: str, systemname: str) -> Path:
        """Create <tmp_path>/<mode>/<org>/results/<system>/training/... with
        one valid run inside, mirroring what `mlpstorage <bench> run`
        actually writes."""
        slice_root = tmp_path / mode / orgname / "results" / systemname
        run_dir = slice_root / "training" / "unet3d" / "run" / "20260123_120000"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "training_unet3d_metadata.json").write_text("{}")
        (run_dir / "summary.json").write_text("{}")
        return slice_root

    def test_returns_results_dir_when_orgname_missing(self, tmp_path):
        """No orgname → no canonical probing (cannot construct the candidate
        path); flat layout passthrough."""
        roots = discover_scan_roots(str(tmp_path), orgname=None,
                                    systemname="sysA")
        assert roots == [str(tmp_path)]

    def test_returns_results_dir_when_systemname_missing(self, tmp_path):
        """No systemname → same flat-layout passthrough."""
        roots = discover_scan_roots(str(tmp_path), orgname="Acme",
                                    systemname=None)
        assert roots == [str(tmp_path)]

    def test_returns_results_dir_when_canonical_slice_absent(self, tmp_path):
        """Orgname + systemname supplied but the canonical layout does not
        exist on disk → fall back to flat layout. Lets legacy users keep
        running reportgen against trees that pre-date `init`."""
        roots = discover_scan_roots(str(tmp_path), orgname="Acme",
                                    systemname="sysA")
        assert roots == [str(tmp_path)]

    def test_returns_closed_slice_when_only_closed_exists(self, tmp_path):
        slice_root = self._make_canonical_slice(
            tmp_path, "closed", "Acme", "sysA"
        )
        roots = discover_scan_roots(str(tmp_path), orgname="Acme",
                                    systemname="sysA")
        assert roots == [str(slice_root)]

    def test_returns_open_slice_when_only_open_exists(self, tmp_path):
        slice_root = self._make_canonical_slice(
            tmp_path, "open", "Acme", "sysA"
        )
        roots = discover_scan_roots(str(tmp_path), orgname="Acme",
                                    systemname="sysA")
        assert roots == [str(slice_root)]

    def test_returns_both_slices_when_both_exist(self, tmp_path):
        """A submitter who staged a tree with both closed/ and open/
        subtrees (uncommon but legal) gets both slices scanned."""
        closed_root = self._make_canonical_slice(
            tmp_path, "closed", "Acme", "sysA"
        )
        open_root = self._make_canonical_slice(
            tmp_path, "open", "Acme", "sysA"
        )
        roots = discover_scan_roots(str(tmp_path), orgname="Acme",
                                    systemname="sysA")
        # Order is closed, open (matches _CANONICAL_MODES iteration).
        assert roots == [str(closed_root), str(open_root)]

    def test_filters_to_requested_system_only(self, tmp_path):
        """Issue #599 bug 3: a multi-system tree must yield only the
        requested system's slice, not every system's subtree mashed into
        one. Pre-fix, get_runs_files walked every system's runs and
        labelled them all with the requested --systemname."""
        wanted = self._make_canonical_slice(
            tmp_path, "closed", "Acme", "sysA"
        )
        # Sibling system that must NOT be returned.
        self._make_canonical_slice(tmp_path, "closed", "Acme", "sysB")
        roots = discover_scan_roots(str(tmp_path), orgname="Acme",
                                    systemname="sysA")
        assert roots == [str(wanted)]

    def test_filters_to_requested_orgname_only(self, tmp_path):
        """Defense-in-depth: a tree with two orgnames under the same
        results-dir yields only the requested org's slice."""
        wanted = self._make_canonical_slice(
            tmp_path, "closed", "Acme", "sysA"
        )
        self._make_canonical_slice(tmp_path, "closed", "OtherOrg", "sysA")
        roots = discover_scan_roots(str(tmp_path), orgname="Acme",
                                    systemname="sysA")
        assert roots == [str(wanted)]

    def test_non_directory_canonical_path_does_not_count(self, tmp_path):
        """If the canonical path exists as a file (broken tree), the helper
        must not treat it as a scan root — fall back to flat layout."""
        bogus = (tmp_path / "closed" / "Acme" / "results")
        bogus.mkdir(parents=True)
        # Make systemname-level entry a FILE, not a directory.
        (bogus / "sysA").write_text("not a dir")
        roots = discover_scan_roots(str(tmp_path), orgname="Acme",
                                    systemname="sysA")
        assert roots == [str(tmp_path)]
