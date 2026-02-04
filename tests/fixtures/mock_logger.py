"""
Mock logger for testing.

Provides a logger that captures all log calls for verification
without writing to disk or stdout.
"""

from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock


class MockLogger:
    """
    A mock logger that captures all log messages for testing.

    Attributes:
        messages: Dictionary mapping log level to list of messages.
        call_count: Dictionary mapping log level to call count.

    Example:
        logger = MockLogger()
        some_function(logger=logger)

        # Verify logging occurred
        assert logger.has_message('info', 'expected message')
        assert logger.call_count['warning'] == 0
    """

    # All supported log levels
    LOG_LEVELS = [
        'debug', 'info', 'warning', 'error', 'critical',
        'status', 'verbose', 'verboser', 'ridiculous', 'result'
    ]

    def __init__(self):
        self.messages: Dict[str, List[str]] = {level: [] for level in self.LOG_LEVELS}
        self.call_count: Dict[str, int] = {level: 0 for level in self.LOG_LEVELS}
        self._setup_methods()

    def _setup_methods(self):
        """Set up log methods dynamically."""
        for level in self.LOG_LEVELS:
            setattr(self, level, self._make_log_method(level))

    def _make_log_method(self, level: str):
        """Create a log method for the given level."""
        def log_method(msg: str, *args, **kwargs):
            # Handle string formatting if args provided
            if args:
                try:
                    msg = msg % args
                except TypeError:
                    pass
            self.messages[level].append(msg)
            self.call_count[level] += 1
        return log_method

    def has_message(self, level: str, substring: str) -> bool:
        """
        Check if any message at the given level contains the substring.

        Args:
            level: Log level to check.
            substring: Substring to search for.

        Returns:
            True if any message contains the substring.
        """
        return any(substring in msg for msg in self.messages.get(level, []))

    def get_messages(self, level: str) -> List[str]:
        """Get all messages for a log level."""
        return self.messages.get(level, [])

    def get_all_messages(self) -> Dict[str, List[str]]:
        """Get all messages for all levels."""
        return self.messages

    def clear(self):
        """Clear all captured messages."""
        self.messages = {level: [] for level in self.LOG_LEVELS}
        self.call_count = {level: 0 for level in self.LOG_LEVELS}

    def assert_logged(self, level: str, substring: str):
        """
        Assert that a message was logged at the given level.

        Args:
            level: Log level to check.
            substring: Expected substring in message.

        Raises:
            AssertionError: If no message contains the substring.
        """
        if not self.has_message(level, substring):
            messages = self.messages.get(level, [])
            raise AssertionError(
                f"Expected '{substring}' in {level} messages.\n"
                f"Actual messages: {messages}"
            )

    def assert_not_logged(self, level: str, substring: str):
        """
        Assert that no message at the given level contains the substring.

        Args:
            level: Log level to check.
            substring: Substring that should not appear.

        Raises:
            AssertionError: If any message contains the substring.
        """
        if self.has_message(level, substring):
            messages = self.messages.get(level, [])
            raise AssertionError(
                f"Did not expect '{substring}' in {level} messages.\n"
                f"Actual messages: {messages}"
            )


def create_mock_logger() -> MockLogger:
    """Factory function to create a MockLogger instance."""
    return MockLogger()
