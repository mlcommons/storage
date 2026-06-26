"""Tests for VectorDB-aware results directory validation."""

import json
from pathlib import Path

from mlpstorage_py.reporting.directory_validator import ResultsDirectoryValidator


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
