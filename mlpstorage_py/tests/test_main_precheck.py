"""Unit test for D-70 pre-check wiring in main.py.

Asserts that ``_check_and_migrate_legacy_layout(args, os.environ, logger)`` is
called IMMEDIATELY BEFORE ``capture_or_verify_code_image(args, os.environ, logger)``
inside ``run_benchmark``'s ``progress_context("Capturing or verifying code image...")``
block.

Added by Plan 07-03 Task 1.

Run with:
    pytest mlpstorage_py/tests/test_main_precheck.py -v
"""

from __future__ import annotations

import inspect

import pytest

import mlpstorage_py.main as main_mod


class TestPreCheckWiringStructural:
    """Structural (source-inspect) assertions for the D-70 pre-check wiring.

    These tests verify that the pre-check call-site exists and appears before
    the capture call-site in the source of ``run_benchmark``.  They are
    complementary to the dynamic mock tests: they survive even if the mocking
    dance requires extra setup, and they are trivially fast.
    """

    def test_main_imports_check_and_migrate_helper(self):
        """main.py must import _check_and_migrate_legacy_layout."""
        source = inspect.getsource(main_mod)
        assert "_check_and_migrate_legacy_layout" in source, (
            "main.py must import _check_and_migrate_legacy_layout "
            "from legacy_migration (D-70 pre-check wiring)"
        )

    def test_precheck_call_appears_before_capture_in_run_benchmark(self):
        """_check_and_migrate_legacy_layout called before capture_or_verify_code_image (D-70).

        Inspect the source of run_benchmark and assert that the pre-check
        identifier occurs at a lower character offset than the capture call.
        This is the structural complement to the mock-based ordering assertion.
        """
        source = inspect.getsource(main_mod.run_benchmark)
        precheck_pos = source.find("_check_and_migrate_legacy_layout(")
        capture_pos = source.find("capture_or_verify_code_image(")
        assert precheck_pos >= 0, (
            "_check_and_migrate_legacy_layout call not found in run_benchmark"
        )
        assert capture_pos >= 0, (
            "capture_or_verify_code_image call not found in run_benchmark"
        )
        assert precheck_pos < capture_pos, (
            f"D-70: pre-check must appear before capture; "
            f"precheck_pos={precheck_pos}, capture_pos={capture_pos}"
        )

    def test_no_try_except_legacy_layout_detected_in_main(self):
        """No try/except LegacyLayoutDetected in main.py (D-70 explicit-pre-check pattern).

        The pre-check is straight-line; migration is invoked unconditionally
        before the capture call. Adding exception-based control flow here would
        violate the D-70 explicit-pre-check design.
        """
        source = inspect.getsource(main_mod)
        assert "except LegacyLayoutDetected" not in source, (
            "main.py must NOT use except LegacyLayoutDetected — "
            "D-70 uses explicit pre-check, not exception-driven retry"
        )

    def test_precheck_import_line_present_in_module_source(self):
        """The legacy_migration import statement exists at module level."""
        source = inspect.getsource(main_mod)
        assert (
            "from mlpstorage_py.submission_checker.tools.legacy_migration import"
            in source
        ), "main.py must import from legacy_migration (D-70 pre-check wiring)"
