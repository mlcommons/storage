"""Unit tests for the Rules.md §6 extension stub (KVCacheCheck).

Pins the structural invariants of ``KVCacheCheck`` (Plan 03-03 Task 4 /
W-02 / CLAUDE.md "new structural checks should have unit tests"):

  1. The stub subclasses ``BaseCheck``.
  2. ``discover_rules(stub_cls) == {}`` — zero ``@rule`` bindings (D-S2;
     Phase 3 success criterion #2).
  3. Instantiation + ``__call__`` returns ``True`` with zero errors and
     zero warnings on the mock logger.
  4. ``init_checks`` registers exactly one method, named
     ``_section_unimplemented``.

Phase 4 Plan 04-02 (D-01) removed ``VdbCheck`` from this fixture: VdbCheck
is no longer a stub — Rules.md §5 was filled by upstream PR #452 and
VdbCheck now carries 16 ``@rule``-decorated methods. Rules.md §6 (KVCache)
is still empty, so ``KVCacheCheck`` continues to satisfy the stub
contract pinned here.

Run with:
    pytest mlpstorage_py/tests/test_stub_checks.py -v
"""

from types import SimpleNamespace

import pytest

from mlpstorage_py.submission_checker.checks.base import BaseCheck
from mlpstorage_py.submission_checker.checks.kvcache_checks import KVCacheCheck
from mlpstorage_py.submission_checker.rule_registry import discover_rules


STUB_CASES = [
    pytest.param(KVCacheCheck, "kvcache checks", id="KVCacheCheck"),
]


def _make_submissions_logs(folder: str):
    """Build the minimal SubmissionLogs stand-in the stubs need.

    Stubs only read ``submissions_logs.loader_metadata.folder`` in
    ``__init__``; nothing else is touched. A ``SimpleNamespace`` is the
    smallest object that satisfies the attribute chain.
    """
    return SimpleNamespace(loader_metadata=SimpleNamespace(folder=folder))


@pytest.mark.parametrize("stub_cls,expected_name", STUB_CASES)
def test_stub_subclasses_basecheck(stub_cls, expected_name):
    assert issubclass(stub_cls, BaseCheck)


@pytest.mark.parametrize("stub_cls,expected_name", STUB_CASES)
def test_stub_discover_rules_is_empty(stub_cls, expected_name):
    # D-S2 invariant: stubs must not contribute any rule-ID bindings.
    assert discover_rules(stub_cls) == {}


@pytest.mark.parametrize("stub_cls,expected_name", STUB_CASES)
def test_stub_init_call_returns_true_with_no_violations(
    stub_cls, expected_name, mock_logger, tmp_path
):
    logs = _make_submissions_logs(str(tmp_path))
    instance = stub_cls(mock_logger, object(), logs)
    assert instance.name == expected_name
    # __call__ routes through run_checks; full instantiation + execution path.
    assert instance() is True
    assert mock_logger.errors == []
    assert mock_logger.warnings == []


@pytest.mark.parametrize("stub_cls,expected_name", STUB_CASES)
def test_stub_registers_exactly_one_check(
    stub_cls, expected_name, mock_logger, tmp_path
):
    logs = _make_submissions_logs(str(tmp_path))
    instance = stub_cls(mock_logger, object(), logs)
    assert len(instance.checks) == 1
    assert instance.checks[0].__name__ == "_section_unimplemented"
