"""
Domain-specific exceptions for the ``mlpstorage_py.results_dir`` package.

All three classes subclass ``ConfigurationError`` so the existing top-level
exception handler in ``mlpstorage_py/main.py`` (~line 371-376) catches them
uniformly, prints the message, and surfaces the ``suggestion`` to the user.

The default ``code=`` for each error matches the semantic intent:

- ``ResultsDirNotInitializedError`` → ``CONFIG_MISSING_REQUIRED``
  (the sentinel file is required before any non-init command runs)
- ``DoubleInitError`` → ``CONFIG_INVALID_VALUE``
  (the operation is invalid against the current state — sentinel exists)
- ``NonEmptyDirError`` → ``CONFIG_INVALID_VALUE``
  (the supplied path is invalid for fresh init — already has content)

Each accepts a ``suggestion=`` keyword that is forwarded to
``ConfigurationError.__init__``, which prefers an explicit suggestion over
its own ``_default_suggestion`` mapping.

Refs: 01-canonical-layout-and-init / 01-01-PLAN.md Task 1 (PATTERNS.md row
``results_dir/errors.py``).
"""

from __future__ import annotations

from typing import Optional

from mlpstorage_py.errors import ConfigurationError, ErrorCode


class ResultsDirNotInitializedError(ConfigurationError):
    """Raised when a non-init command is run against a results-dir that has
    no ``mlperf-results.yaml`` sentinel (or whose sentinel is malformed).

    Maps to ``ErrorCode.CONFIG_MISSING_REQUIRED`` so the user sees the
    standard "Provide the required parameter…" hint unless the caller
    supplies a more specific ``suggestion=``.
    """

    def __init__(
        self,
        message: str,
        *,
        suggestion: Optional[str] = None,
        code: ErrorCode = ErrorCode.CONFIG_MISSING_REQUIRED,
    ) -> None:
        super().__init__(message=message, suggestion=suggestion, code=code)


class DoubleInitError(ConfigurationError):
    """Raised when ``mlpstorage init`` is run against a results-dir whose
    sentinel already exists and the supplied orgname does not match the
    existing one (or, at the sentinel layer, raised unconditionally on a
    second exclusive-create attempt — the idempotency check happens in
    Slice 2's ``init`` command handler).

    Maps to ``ErrorCode.CONFIG_INVALID_VALUE``.
    """

    def __init__(
        self,
        message: str,
        *,
        suggestion: Optional[str] = None,
        code: ErrorCode = ErrorCode.CONFIG_INVALID_VALUE,
    ) -> None:
        super().__init__(message=message, suggestion=suggestion, code=code)


class NonEmptyDirError(ConfigurationError):
    """Raised when ``mlpstorage init`` targets a directory that contains
    files but has no sentinel — D-09 forbids silently adopting an existing
    non-initialized tree.

    Maps to ``ErrorCode.CONFIG_INVALID_VALUE``.
    """

    def __init__(
        self,
        message: str,
        *,
        suggestion: Optional[str] = None,
        code: ErrorCode = ErrorCode.CONFIG_INVALID_VALUE,
    ) -> None:
        super().__init__(message=message, suggestion=suggestion, code=code)
