"""
Verify the bundled example_*.yaml system-description files validate cleanly
against the schema defined in mlpstorage_py.system_description.schema_validator.

These examples are shipped as reference templates for submitters; a regression
in either the schema or the examples should fail this test.
"""

from pathlib import Path

import pytest

from mlpstorage_py.system_description.schema_validator import validate_file

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "mlpstorage_py" / "system_description"
EXAMPLE_FILES = sorted(EXAMPLES_DIR.glob("example_*.yaml"))


def test_example_files_discovered():
    """Guard against the glob silently matching nothing (e.g. dir moved)."""
    assert EXAMPLE_FILES, f"No example_*.yaml files found in {EXAMPLES_DIR}"


@pytest.mark.parametrize("yaml_path", EXAMPLE_FILES, ids=lambda p: p.name)
def test_example_yaml_validates(yaml_path: Path):
    errors = validate_file(yaml_path)
    assert errors == [], (
        f"{yaml_path.name} failed schema validation:\n  "
        + "\n  ".join(errors)
    )
