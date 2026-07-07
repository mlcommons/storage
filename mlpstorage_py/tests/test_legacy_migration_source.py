"""Structural (grep-testable) invariants over legacy_migration.py.

Tests the source-level structural invariants satisfied by
``mlpstorage_py/submission_checker/tools/legacy_migration.py`` (created in
Plan 07-02):

  - Fixed step order in pass 2: materialize -> pointers -> delete -> sentinel (D-71)
  - Two-pass separation: pass 1 (verify) completes before any pass-2 write (D-73)
  - No try/except wrapping pass-1 verify call (D-73)
  - Sentinel writer uses write-tmp + os.rename (D-65 atomic pattern)
  - HandEditedCodeImage subclasses CodeImageError (in code_image.py)
  - Exactly two log.status() call sites in the module (D-74)

Each test asserts a structural property of the source file by reading it from
disk and applying regex or line-counting operations. They are grep-testable
invariants — not behavioral (no runtime execution of production code).

Wave 0 stubs replaced with real assertions by Plan 07-03 Task 2.

Refs: 07-01-PLAN.md Task 3, 07-CONTEXT.md D-71/D-73/D-74,
07-VALIDATION.md §Structural Invariants, RESEARCH §10 structural-invariants table.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest


def test_fixed_step_order_in_pass_2():
    """Pass-2 step order in migrate_legacy_layout is load-bearing for crash-safety (D-71).

    Read the source of migrate_legacy_layout and assert that the four
    pass-2 helper identifiers appear in the following order:
      1. _materialize_pool_images
      2. _write_pointers_for_migrated_leaves
      3. _delete_legacy_dirs
      4. _write_sentinel_atomic (LAST — after _delete_legacy_dirs)

    A reordering of these steps would break crash-resumability — this structural
    invariant locks the implementation order.

    Note: _write_sentinel_atomic also appears in the N=0 early-return branch
    (before the four pass-2 steps). We assert the LAST occurrence of
    _write_sentinel_atomic comes after _delete_legacy_dirs, which is the
    pass-2 sentinel write.
    """
    from mlpstorage_py.submission_checker.tools import legacy_migration as lm

    source = inspect.getsource(lm.migrate_legacy_layout)
    mat_pos = source.index("_materialize_pool_images")
    ptr_pos = source.index("_write_pointers_for_migrated_leaves")
    del_pos = source.index("_delete_legacy_dirs")
    # Find the LAST occurrence of _write_sentinel_atomic — that is the pass-2 step 4.
    # The first occurrence is the N=0 early-return path which legitimately precedes
    # the main pass-2 block.
    sent_pos = source.rindex("_write_sentinel_atomic")
    assert mat_pos < ptr_pos < del_pos < sent_pos, (
        f"D-71 fixed step order violated: "
        f"materialize={mat_pos}, pointers={ptr_pos}, delete={del_pos}, sentinel={sent_pos}"
    )


def test_two_pass_separation():
    """Pass-1 verify completes before any pass-2 write (D-73 strict two-pass).

    Assert that ``_verify_all_legacy_dirs`` appears as a call site BEFORE
    any of the pass-2 write functions (_materialize_pool_images,
    _write_pointers_for_migrated_leaves) inside the body of
    ``migrate_legacy_layout``.

    This invariant makes the "abort before any writes" guarantee structural
    rather than test-guarded.
    """
    from mlpstorage_py.submission_checker.tools import legacy_migration as lm

    source = inspect.getsource(lm.migrate_legacy_layout)
    verify_pos = source.index("_verify_all_legacy_dirs")
    mat_pos = source.index("_materialize_pool_images")
    ptr_pos = source.index("_write_pointers_for_migrated_leaves")
    assert verify_pos < mat_pos, "D-73 pass-1-before-pass-2 violated (verify before materialize)"
    assert verify_pos < ptr_pos, "D-73 pass-1-before-pass-2 violated (verify before pointers)"


def test_no_try_except_around_pass_1():
    """No try/except wraps the pass-1 verify call in migrate_legacy_layout (D-73).

    Assert via regex counting on the source prefix that there is no open
    ``try:`` block wrapping the ``_verify_all_legacy_dirs`` call. A try/except
    here would silently suppress HandEditedCodeImage and corrupt the
    "abort before any writes" guarantee.

    Strategy: count ``try:`` openings and ``except`` closings that appear
    BEFORE the verify call site. If opens > closes, a try-block is still open
    at the point of the verify call (violation).
    """
    from mlpstorage_py.submission_checker.tools import legacy_migration as lm

    source = inspect.getsource(lm.migrate_legacy_layout)
    verify_pos = source.index("_verify_all_legacy_dirs")
    prefix = source[:verify_pos]
    opens = len(re.findall(r"(?m)^\s*try\s*:", prefix))
    closes = len(re.findall(r"(?m)^\s*except\b", prefix))
    assert opens == closes, (
        f"D-73 pass 1 is wrapped in try/except (opens={opens}, closes={closes}); "
        "HandEditedCodeImage would be suppressed"
    )


def test_sentinel_writer_uses_write_tmp_and_os_rename():
    """_write_sentinel_atomic uses write-tmp + os.rename (D-65 atomic pattern).

    Assert that the source of ``_write_sentinel_atomic`` in legacy_migration.py
    contains both:
    - A tmp file reference matching the ``.tmp.`` pattern
    - An ``os.rename(`` call

    This mirrors the D-65 invariant already enforced in _write_pointer_atomic
    (code_image.py) and the Phase 6 structural tests.
    """
    from mlpstorage_py.submission_checker.tools import legacy_migration as lm

    source = inspect.getsource(lm._write_sentinel_atomic)
    assert "os.rename(" in source, (
        "D-65 atomic sentinel write requires os.rename"
    )
    assert ".tmp." in source, (
        "D-65 atomic pattern requires .tmp. sibling file"
    )


def test_HandEditedCodeImage_subclasses_CodeImageError():
    """HandEditedCodeImage(CodeImageError) exists in code_image.py and is importable (D-73).

    Asserts:
    - HandEditedCodeImage is importable from code_image.
    - issubclass(HandEditedCodeImage, CodeImageError) is True.
    - Instantiation + str round-trip works.
    """
    from mlpstorage_py.submission_checker.tools.code_image import (
        HandEditedCodeImage,
        CodeImageError,
    )
    assert issubclass(HandEditedCodeImage, CodeImageError) is True
    e = HandEditedCodeImage("hand-edited code image detected at 'x'")
    assert str(e) == "hand-edited code image detected at 'x'"


def test_exactly_two_log_status_call_sites_in_module():
    """Exactly two log.status() call sites in legacy_migration.py (D-74).

    Read the source of legacy_migration.py and count occurrences of
    ``log.status(`` excluding comment lines. Assert count == 2 — the header
    line and the completion summary line. Any additional log.status() calls
    would violate D-74's "concise user-facing output" requirement.
    """
    source = Path("mlpstorage_py/submission_checker/tools/legacy_migration.py").read_text()
    non_comment = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    count = non_comment.count("log.status(")
    assert count == 2, (
        f"D-74 requires exactly 2 log.status call sites, found {count}"
    )
