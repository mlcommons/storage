#!/usr/bin/env python3
"""
Tests for the Phase 2 wiring of capture_or_verify_code_image into main.py.

Covers D-07 (insertion point) and D-22 (exit-code mapping).

Run with:
    pytest mlpstorage_py/tests/test_main_code_image_wiring.py -v
"""

import ast
from pathlib import Path

import pytest

from mlpstorage_py.config import EXIT_CODE


MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


class TestMainImports:
    def test_main_imports_capture_or_verify_helper(self):
        """main.py must import the helper from the Phase 1 module."""
        source = MAIN_PATH.read_text()
        assert "from mlpstorage_py.submission_checker.tools.code_image import" in source, \
            "main.py should have a single import line for the code_image symbols"
        assert "capture_or_verify_code_image" in source
        assert "CodeImageError" in source

    def test_main_importable(self):
        """Importing main must not raise (no syntax / import drift)."""
        from mlpstorage_py.main import main, run_benchmark  # noqa: F401


class TestExceptCodeImageErrorClause:
    """D-22: dedicated except CodeImageError clause returning CODE_IMAGE_ERROR."""

    def test_except_clause_present(self):
        tree = ast.parse(MAIN_PATH.read_text())
        handler_names = [
            getattr(h.type, "id", None)
            for h in ast.walk(tree)
            if isinstance(h, ast.ExceptHandler) and h.type
        ]
        assert "CodeImageError" in handler_names, \
            f"main.py must have `except CodeImageError`; saw {handler_names}"

    def test_except_clause_order_dependency_before_mlpstorage(self):
        """Order: ...DependencyError -> CodeImageError -> MLPStorageException catch-all.

        CodeImageError is NOT a subclass of MLPStorageException so MRO does not
        implicitly fold it in; we need an explicit clause BEFORE the catch-all.
        """
        tree = ast.parse(MAIN_PATH.read_text())
        # Find the main() function and inspect its top-level except handlers.
        main_fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main"]
        assert main_fns, "main() function not found"
        main_fn = main_fns[0]

        names = []
        for node in ast.walk(main_fn):
            if isinstance(node, ast.Try):
                for h in node.handlers:
                    if isinstance(h.type, ast.Name):
                        names.append(h.type.id)
        assert "CodeImageError" in names, names
        ci_idx = names.index("CodeImageError")
        mlps_idx = names.index("MLPStorageException")
        # CodeImageError must come BEFORE the catch-all MLPStorageException
        assert ci_idx < mlps_idx, (
            f"except CodeImageError must precede except MLPStorageException; saw order {names}"
        )

    def test_except_clause_returns_code_image_error(self):
        """The new except clause must return EXIT_CODE.CODE_IMAGE_ERROR."""
        source = MAIN_PATH.read_text()
        assert "EXIT_CODE.CODE_IMAGE_ERROR" in source

    def test_code_image_error_value_is_two(self):
        """EXIT_CODE.CODE_IMAGE_ERROR must integer-equal 2 (D-22)."""
        assert int(EXIT_CODE.CODE_IMAGE_ERROR) == 2


class TestHelperInvocation:
    """D-07: helper called inside run_benchmark BEFORE benchmark instantiation."""

    def test_helper_called_in_run_benchmark(self):
        source = MAIN_PATH.read_text()
        # Strip comment lines so we count actual code call sites only.
        code_only = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "capture_or_verify_code_image(args, os.environ, logger)" in code_only, \
            "main.py must call capture_or_verify_code_image(args, os.environ, logger) in run_benchmark"

    def test_helper_call_precedes_benchmark_instantiation(self):
        """The helper invocation must appear before `benchmark_class(args, ...)`."""
        source = MAIN_PATH.read_text()
        helper_idx = source.find("capture_or_verify_code_image(args, os.environ, logger)")
        benchmark_idx = source.find("benchmark_class(args")
        assert helper_idx >= 0, "helper call site not found"
        assert benchmark_idx >= 0, "benchmark_class(args, ...) instantiation site not found"
        assert helper_idx < benchmark_idx, (
            "capture_or_verify_code_image must be invoked BEFORE benchmark_class(args, ...)"
        )

    def test_helper_call_wrapped_in_progress_context(self):
        """D-07: invocation is wrapped in progress_context for consistent UX."""
        source = MAIN_PATH.read_text()
        # Find the helper call site and check the surrounding lines.
        idx = source.find("capture_or_verify_code_image(args, os.environ, logger)")
        assert idx >= 0
        # Look at the preceding 400 chars for the progress_context wrapper.
        window = source[max(0, idx - 400):idx]
        assert "progress_context" in window, "helper invocation must be inside progress_context"
        assert "Capturing or verifying code image" in window
