"""Unit tests for EXIT_CODE.CODE_IMAGE_ERROR (Plan 02-01, Task 1).

Phase 2 introduces a new enum member CODE_IMAGE_ERROR on EXIT_CODE, used by
the typed-exception → process-exit-code mapping in main.py to signal that a
code-image capture or verify operation failed.

Per 02-CONTEXT.md D-22 the value is 2; per 02-PATTERNS.md the symbol is
preferred for grep-ability over reusing INVALID_ARGUMENTS at the call site.
Because IntEnum aliases on duplicate values, CODE_IMAGE_ERROR is an alias of
INVALID_ARGUMENTS — both names resolve to 2.
"""

from mlpstorage_py.config import EXIT_CODE


def test_code_image_error_member_exists():
    """The new enum member is importable."""
    assert hasattr(EXIT_CODE, "CODE_IMAGE_ERROR")


def test_code_image_error_value_is_two():
    """Per D-22 the integer value is 2 (aliased with INVALID_ARGUMENTS)."""
    assert EXIT_CODE.CODE_IMAGE_ERROR.value == 2


def test_code_image_error_int_cast():
    """The member is usable as a process exit code (int-castable)."""
    assert int(EXIT_CODE.CODE_IMAGE_ERROR) == 2


def test_code_image_error_name_grepable():
    """Either the name resolves to CODE_IMAGE_ERROR directly, or — because
    IntEnum's canonical-name resolution prefers the first-defined alias —
    the symbol still exists as a class attribute. The grep-ability acceptance
    criterion is that ``CODE_IMAGE_ERROR`` is a usable symbolic name.
    """
    # Direct attribute access must work regardless of canonical aliasing.
    assert EXIT_CODE.CODE_IMAGE_ERROR is not None
    # And the symbolic identity must be the same as the INVALID_ARGUMENTS alias
    # because they share the integer value 2.
    assert EXIT_CODE.CODE_IMAGE_ERROR == EXIT_CODE.INVALID_ARGUMENTS


def test_preexisting_exit_codes_unchanged():
    """Adding the alias must not renumber pre-existing members."""
    assert EXIT_CODE.SUCCESS.value == 0
    assert EXIT_CODE.GENERAL_ERROR.value == 1
    assert EXIT_CODE.INVALID_ARGUMENTS.value == 2
    assert EXIT_CODE.FILE_NOT_FOUND.value == 3
    assert EXIT_CODE.PERMISSION_DENIED.value == 4
    assert EXIT_CODE.CONFIGURATION_ERROR.value == 5
    assert EXIT_CODE.FAILURE.value == 6
    assert EXIT_CODE.TIMEOUT.value == 7
    assert EXIT_CODE.INTERRUPTED.value == 8


def test_enumeration_does_not_raise():
    """Iterating the enum produces all defined members without error."""
    members = list(EXIT_CODE)
    # IntEnum aliases are not iterated as separate entries; iteration count
    # should equal the count of distinct canonical values (9 in the current
    # enum: SUCCESS..INTERRUPTED).
    assert len(members) == 9
