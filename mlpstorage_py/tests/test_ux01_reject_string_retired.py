#!/usr/bin/env python3
"""Phase 6 Plan 06-02 Task 3 (RED) — UX-01 source-guard tests.

Locks the UX-01 + D-63 requirement that the retired reject strings are
absent from the `submission_checker/tools/code_image.py` source — including
comment lines. The two strings are:

  1. `changes to the codebase are not allowed [in a CLOSED run]` (CLOSED reject)
  2. `all runs of this type must use the same codebase`             (OPEN reject)

Both are the user-facing UX that CAPVER-03 + UX-01 retire. Once the retire
lands (Plan 06-02 Task 4 GREEN), neither string may appear in the module
source — no code, no comments. A future accidental re-introduction would
give a submitter a stale UX that the runtime no longer emits.

Note: the retired strings are named ONLY in comparison expressions here.
This test module intentionally does not print, log, or export those
strings — the negative-grep assertion is over `code_image.py` source, not
over this test file.

Run with:
    pytest mlpstorage_py/tests/test_ux01_reject_string_retired.py -v
"""

from pathlib import Path


def test_ux01_reject_string_retired():
    """UX-01 (SC#7): the CLOSED-mode reject string is absent from module source.

    Uses the leading substring `changes to the codebase are not allowed`
    (without the trailing `in a CLOSED run` qualifier) to be robust against
    minor rephrasings. If any variant of the substring appears anywhere in
    the source, this test fails.
    """
    import mlpstorage_py.submission_checker.tools.code_image as code_image_module

    source = Path(code_image_module.__file__).read_text(encoding="utf-8")

    forbidden_closed = "changes to the codebase are not allowed"
    assert forbidden_closed not in source, (
        f"UX-01 violation: forbidden substring {forbidden_closed!r} found in "
        f"{code_image_module.__file__}. Per UX-01 + D-63, the retired reject "
        f"UX must not appear in module source (code OR comments)."
    )


def test_ux01_reject_string_retired_by_conceptual_intent():
    """UX-01 (SC#8): the OPEN-mode sibling reject string is also absent.

    Both retired strings must be removed together per D-63 rationale — the
    CLOSED and OPEN retire is a single semantic change (CAPVER-03), not a
    per-mode toggle.
    """
    import mlpstorage_py.submission_checker.tools.code_image as code_image_module

    source = Path(code_image_module.__file__).read_text(encoding="utf-8")

    forbidden_open = "all runs of this type must use the same codebase"
    assert forbidden_open not in source, (
        f"UX-01 violation: forbidden substring {forbidden_open!r} found in "
        f"{code_image_module.__file__}. Per UX-01 + D-63, the retired OPEN "
        f"reject UX must not appear in module source (code OR comments)."
    )
