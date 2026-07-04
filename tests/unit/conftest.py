"""Unit-test conftest.

Phase 6 Plan 06-03 (D-60) retired the legacy in-``__init__`` code-capture
path (the old ``results_dir`` capture module) and its whole-suite autouse
stub. Code capture now runs once in ``mlpstorage_py/main.py`` before
``Benchmark`` construction, so no autouse patching is needed here anymore.

Integration tests deliberately do NOT install autouse patches — they
exercise the real capture pipeline.
"""
from __future__ import annotations
