#!/usr/bin/env python3
"""
Tests for BaseCheck.log_violation and BaseCheck.warn_violation helpers.

Covers the locked [<rule_id> <rule_name>] <path>: <msg> emission format
and the routing through self.log.error / self.log.warning.

Run with:
    pytest mlpstorage_py/tests/test_base_check_helpers.py -v
"""

import pytest

from mlpstorage_py.submission_checker.checks.base import BaseCheck


class CapturingMockLogger:
    """Mock logger that captures error() and warning() calls.

    Stores calls as (format_string, args) tuples so tests can assert on
    both the raw format and the formatted output.
    """
    def __init__(self):
        self.error_calls = []
        self.warning_calls = []

    def debug(self, msg, *args, **kwargs):
        pass

    def info(self, msg, *args, **kwargs):
        pass

    def warning(self, msg, *args, **kwargs):
        self.warning_calls.append((msg, args))

    def error(self, msg, *args, **kwargs):
        # **kwargs swallows exc_info=True passed by BaseCheck.run_checks for
        # tracebacks. The mock records only the format string + args; the
        # traceback isn't asserted on at this level.
        self.error_calls.append((msg, args))

    def formatted_errors(self):
        """Return error messages with % formatting applied."""
        results = []
        for fmt, args in self.error_calls:
            results.append(fmt % args if args else fmt)
        return results

    def formatted_warnings(self):
        """Return warning messages with % formatting applied."""
        results = []
        for fmt, args in self.warning_calls:
            results.append(fmt % args if args else fmt)
        return results


class ConcreteCheck(BaseCheck):
    """Minimal concrete subclass for testing BaseCheck helpers."""

    def init_checks(self):
        self.checks = []


@pytest.fixture
def mock_log():
    """Return a fresh CapturingMockLogger for each test."""
    return CapturingMockLogger()


@pytest.fixture
def check(mock_log):
    """Return a ConcreteCheck instance wired to the mock logger."""
    return ConcreteCheck(log=mock_log, path="/test/root")


class TestLogViolation:
    """Tests for BaseCheck.log_violation."""

    def test_log_violation_emits_locked_format(self, check, mock_log):
        """log_violation emits the exact locked format [rule_id rule_name] path: msg."""
        check.log_violation(
            "2.1.2", "topLevelSubdirectories", "/sub/Acme",
            "unexpected entry %r (expected lowercase 'closed')", "Closed",
        )
        formatted = mock_log.formatted_errors()
        assert len(formatted) == 1
        assert formatted[0] == (
            "[2.1.2 topLevelSubdirectories] /sub/Acme: "
            "unexpected entry 'Closed' (expected lowercase 'closed')"
        )

    def test_log_violation_routes_through_log_error(self, check, mock_log):
        """log_violation routes to self.log.error, not print or log.warning."""
        check.log_violation("2.1.2", "topLevelSubdirectories", "/x", "some msg")
        assert len(mock_log.error_calls) == 1
        assert len(mock_log.warning_calls) == 0

    def test_log_violation_does_not_raise_and_returns_none(self, check):
        """log_violation does not raise and returns None (no value)."""
        result = check.log_violation("2.1.2", "topLevelSubdirectories", "/x", "msg")
        assert result is None

    def test_log_violation_percent_formatting_passthrough(self, check, mock_log):
        """log_violation passes *args through so msg % args formatting works."""
        check.log_violation(
            "2.1.17", "runTimestamps", "/results",
            "expected %d, got %d", 6, 5,
        )
        formatted = mock_log.formatted_errors()
        assert len(formatted) == 1
        assert "expected 6, got 5" in formatted[0]


class TestWarnViolation:
    """Tests for BaseCheck.warn_violation."""

    def test_warn_violation_emits_locked_format(self, check, mock_log):
        """warn_violation emits the same locked format as log_violation."""
        check.warn_violation(
            "2.1.6", "codeDirectoryContents",
            "/sub/Acme/code/lnk", "symlink rejected (skipped)",
        )
        formatted = mock_log.formatted_warnings()
        assert len(formatted) == 1
        assert formatted[0] == (
            "[2.1.6 codeDirectoryContents] /sub/Acme/code/lnk: symlink rejected (skipped)"
        )

    def test_warn_violation_routes_through_log_warning(self, check, mock_log):
        """warn_violation routes to self.log.warning, not log.error."""
        check.warn_violation("2.1.6", "codeDirectoryContents", "/x", "msg")
        assert len(mock_log.warning_calls) == 1
        assert len(mock_log.error_calls) == 0

    def test_warn_violation_does_not_raise_and_returns_none(self, check):
        """warn_violation does not raise and returns None."""
        result = check.warn_violation("2.1.6", "codeDirectoryContents", "/x", "msg")
        assert result is None


class TestRunChecksUnchanged:
    """Tests that BaseCheck.run_checks behavior is unchanged after the new helpers."""

    def test_run_checks_returns_true_when_all_pass(self, mock_log):
        """run_checks returns True when all registered checks pass."""
        class AllPassCheck(BaseCheck):
            def init_checks(self):
                self.checks = [self.check_one, self.check_two]

            def check_one(self):
                return True

            def check_two(self):
                return True

        c = AllPassCheck(log=mock_log, path="/p")
        c.init_checks()
        assert c.run_checks() is True

    def test_run_checks_returns_false_when_one_fails(self, mock_log):
        """run_checks returns False when at least one registered check fails."""
        class OneFailCheck(BaseCheck):
            def init_checks(self):
                self.checks = [self.check_pass, self.check_fail]

            def check_pass(self):
                return True

            def check_fail(self):
                return False

        c = OneFailCheck(log=mock_log, path="/p")
        c.init_checks()
        assert c.run_checks() is False

    def test_run_checks_catches_exception_in_check(self, mock_log):
        """run_checks catches exceptions inside check methods and returns False."""
        class ExceptionCheck(BaseCheck):
            def init_checks(self):
                self.checks = [self.boom]

            def boom(self):
                raise ValueError("unexpected error")

        c = ExceptionCheck(log=mock_log, path="/p")
        c.init_checks()
        result = c.run_checks()
        assert result is False
