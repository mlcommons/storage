"""Unit-test conftest.

Suppresses production side-effects that `Benchmark.__init__` triggers when
unit tests instantiate it directly with synthetic Namespace args. Phase 1
(LAY-05 / code-image capture) added a call to
`mlpstorage_py.results_dir.code_image.capture_code_image()` that copies the
`mlpstorage_py/` source tree alongside results on every closed/open
benchmark construction. That is correct production behavior, but unit tests
that construct ConcreteBenchmark with a tmp_path results-dir would otherwise
shell out a full `shutil.copytree` on every test. The autouse fixture below
neutralises that side-effect for the entire unit-test suite.

Integration tests deliberately do NOT install this fixture — they exercise
the real capture pipeline.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _suppress_capture_code_image():
    """Replace capture_code_image with a stub that returns a dummy path.

    The benchmark constructor does a deferred import
    `from mlpstorage_py.results_dir.code_image import capture_code_image`
    inside __init__, so we patch the source module's symbol so the import
    resolves to the stub.
    """
    with patch(
        "mlpstorage_py.results_dir.code_image.capture_code_image",
        return_value="/tmp/mock-code-image",
    ):
        yield
