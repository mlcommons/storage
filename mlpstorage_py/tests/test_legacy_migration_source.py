"""Wave-0 xfail scaffolding for structural (grep-testable) invariants over legacy_migration.py.

Tests the source-level structural invariants that Plan 07-02 must satisfy when it
creates ``mlpstorage_py/submission_checker/tools/legacy_migration.py``:

  - Fixed step order in pass 2: materialize -> pointers -> delete -> sentinel
  - Two-pass separation: pass 1 (verify) completes before any pass-2 write
  - No try/except wrapping pass-1 verify call (D-73)
  - Sentinel writer uses write-tmp + os.rename (D-65 atomic pattern)
  - HandEditedCodeImage subclasses CodeImageError (in code_image.py)
  - Exactly two log.status() call sites in the module (D-74)

Each test asserts a structural property of the source file by reading it from
disk and applying regex or line-counting operations. They are grep-testable
invariants — not behavioral (no runtime execution of production code).

Wave 0 note: every test stub raises NotImplementedError and is marked
xfail(strict=True). Wave-1 (Plan 07-03) removes xfail decorators and
populates test bodies once the production module exists on disk. No production
symbols are imported — tests operate on source text only.

Refs: 07-01-PLAN.md Task 3, 07-CONTEXT.md D-71/D-73/D-74,
07-VALIDATION.md §Structural Invariants, RESEARCH §10 structural-invariants table.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


@pytest.mark.xfail(strict=True, reason="Wave 0 scaffold — implementation in Plan 07-02/07-03/07-04", raises=NotImplementedError)
def test_fixed_step_order_in_pass_2():
    """Pass-2 step order in migrate_legacy_layout is load-bearing for crash-safety (D-71).

    Read the source of legacy_migration.py and assert that, inside the body of
    ``migrate_legacy_layout``, the following identifiers appear in the specified
    order (by line number):
      1. _materialize_pool_images (or _capture_new_pool_image call site)
      2. _write_pointers_for_migrated_leaves (or _write_pointer_atomic loop)
      3. _delete_legacy_dirs (or shutil.rmtree call site)
      4. _write_sentinel_atomic

    A reordering of these steps would break crash-resumability — this structural
    invariant locks the implementation order.
    """
    raise NotImplementedError(
        "Wave 0 stub — Plan 07-03 (structural) or Plan 07-04 (unit) populates"
    )


@pytest.mark.xfail(strict=True, reason="Wave 0 scaffold — implementation in Plan 07-02/07-03/07-04", raises=NotImplementedError)
def test_two_pass_separation():
    """Pass-1 verify completes before any pass-2 write (D-73 strict two-pass).

    Assert that ``_verify_all_legacy_dirs`` (or ``_verify_legacy_layout``)
    appears as a call site BEFORE any of the pass-2 write functions
    (_materialize_pool_images, _write_pointer_atomic, _write_sentinel_atomic)
    inside the body of ``migrate_legacy_layout``.

    This invariant makes the "abort before any writes" guarantee structural
    rather than test-guarded.
    """
    raise NotImplementedError(
        "Wave 0 stub — Plan 07-03 (structural) or Plan 07-04 (unit) populates"
    )


@pytest.mark.xfail(strict=True, reason="Wave 0 scaffold — implementation in Plan 07-02/07-03/07-04", raises=NotImplementedError)
def test_no_try_except_around_pass_1():
    """No try/except wraps the pass-1 verify call in migrate_legacy_layout (D-73).

    Assert via regex negative-match on the function body that there is no
    ``try:`` block wrapping the ``_verify_all_legacy_dirs`` call. A try/except
    here would silently suppress HandEditedCodeImage and corrupt the "abort
    before any writes" guarantee.
    """
    raise NotImplementedError(
        "Wave 0 stub — Plan 07-03 (structural) or Plan 07-04 (unit) populates"
    )


@pytest.mark.xfail(strict=True, reason="Wave 0 scaffold — implementation in Plan 07-02/07-03/07-04", raises=NotImplementedError)
def test_sentinel_writer_uses_write_tmp_and_os_rename():
    """_write_sentinel_atomic uses write-tmp + os.rename (D-65 atomic pattern).

    Assert that the source of ``_write_sentinel_atomic`` in legacy_migration.py
    contains both:
    - A tmp file reference matching ``*.tmp.*`` pattern
    - An ``os.rename(`` call

    This mirrors the D-65 invariant already enforced in _write_pointer_atomic
    (code_image.py) and the Phase 6 structural tests.
    """
    raise NotImplementedError(
        "Wave 0 stub — Plan 07-03 (structural) or Plan 07-04 (unit) populates"
    )


@pytest.mark.xfail(strict=True, reason="Wave 0 scaffold — implementation in Plan 07-02/07-03/07-04", raises=NotImplementedError)
def test_HandEditedCodeImage_subclasses_CodeImageError():
    """HandEditedCodeImage(CodeImageError) declaration exists in code_image.py (D-73).

    Assert that the literal string ``class HandEditedCodeImage(CodeImageError):``
    appears in mlpstorage_py/submission_checker/tools/code_image.py.

    This structural test ensures the exception is placed in the correct module
    (alongside LegacyLayoutDetected, PoolCorruption, etc.) so main.py's
    existing exit-code mapping continues to work without a new handler.
    """
    raise NotImplementedError(
        "Wave 0 stub — Plan 07-03 (structural) or Plan 07-04 (unit) populates"
    )


@pytest.mark.xfail(strict=True, reason="Wave 0 scaffold — implementation in Plan 07-02/07-03/07-04", raises=NotImplementedError)
def test_exactly_two_log_status_call_sites_in_module():
    """Exactly two log.status() call sites in legacy_migration.py (D-74).

    Read the source of legacy_migration.py and count occurrences of
    ``log.status(``. Assert count == 2 — the header line and the completion
    summary line. Any additional log.status() calls would violate D-74's
    "concise user-facing output" requirement.
    """
    raise NotImplementedError(
        "Wave 0 stub — Plan 07-03 (structural) or Plan 07-04 (unit) populates"
    )
