"""
Mock command executor for testing.

Provides a command executor that records commands and returns
mock responses without actual subprocess execution.
"""

from typing import Tuple, List, Dict, Any, Optional, Union, Pattern
import re


class MockCommandExecutor:
    """
    Mock command executor for testing without subprocess calls.

    Records all executed commands and returns predefined responses.
    Useful for testing benchmark execution logic without running
    actual DLIO or MPI commands.

    Attributes:
        responses: Dict mapping command patterns to (stdout, stderr, exit_code).
        executed_commands: List of all commands that were "executed".
        default_response: Default response when no pattern matches.

    Example:
        executor = MockCommandExecutor({
            'dlio_benchmark': ('benchmark output', '', 0),
            'mpirun.*error': ('', 'MPI error', 1),
        })

        # In code under test:
        stdout, stderr, code = executor.execute('dlio_benchmark --config test')

        # Verify:
        executor.assert_command_executed('dlio_benchmark')
        assert code == 0
    """

    def __init__(
        self,
        responses: Optional[Dict[str, Tuple[str, str, int]]] = None,
        default_response: Tuple[str, str, int] = ('', '', 0)
    ):
        """
        Initialize the mock executor.

        Args:
            responses: Dictionary mapping command patterns (regex or substring)
                       to (stdout, stderr, exit_code) tuples.
            default_response: Response to return when no pattern matches.
        """
        self.responses = responses or {}
        self.default_response = default_response
        self.executed_commands: List[str] = []
        self.execution_count: int = 0

    def execute(
        self,
        command: str,
        timeout: Optional[int] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Tuple[str, str, int]:
        """
        Record command and return mock response.

        Args:
            command: The command string to "execute".
            timeout: Ignored (for interface compatibility).
            cwd: Ignored (for interface compatibility).
            env: Ignored (for interface compatibility).
            **kwargs: Additional ignored arguments.

        Returns:
            Tuple of (stdout, stderr, exit_code).
        """
        self.executed_commands.append(command)
        self.execution_count += 1

        # Check for pattern matches
        for pattern, response in self.responses.items():
            if self._matches(pattern, command):
                return response

        return self.default_response

    def _matches(self, pattern: str, command: str) -> bool:
        """Check if command matches pattern (regex or substring)."""
        try:
            # Try as regex first
            return bool(re.search(pattern, command))
        except re.error:
            # Fall back to substring match
            return pattern in command

    def add_response(
        self,
        pattern: str,
        stdout: str = '',
        stderr: str = '',
        exit_code: int = 0
    ):
        """
        Add a response for a command pattern.

        Args:
            pattern: Regex or substring pattern to match.
            stdout: Standard output to return.
            stderr: Standard error to return.
            exit_code: Exit code to return.
        """
        self.responses[pattern] = (stdout, stderr, exit_code)

    def assert_command_executed(self, pattern: str) -> str:
        """
        Assert that a command matching the pattern was executed.

        Args:
            pattern: Regex or substring pattern to match.

        Returns:
            The matching command string.

        Raises:
            AssertionError: If no matching command was executed.
        """
        for cmd in self.executed_commands:
            if self._matches(pattern, cmd):
                return cmd
        raise AssertionError(
            f"No command matching '{pattern}' was executed.\n"
            f"Executed commands: {self.executed_commands}"
        )

    def assert_command_not_executed(self, pattern: str):
        """
        Assert that no command matching the pattern was executed.

        Args:
            pattern: Regex or substring pattern that should not match.

        Raises:
            AssertionError: If a matching command was executed.
        """
        for cmd in self.executed_commands:
            if self._matches(pattern, cmd):
                raise AssertionError(
                    f"Command matching '{pattern}' was unexpectedly executed: {cmd}"
                )

    def get_commands_matching(self, pattern: str) -> List[str]:
        """
        Get all executed commands matching a pattern.

        Args:
            pattern: Regex or substring pattern to match.

        Returns:
            List of matching command strings.
        """
        return [cmd for cmd in self.executed_commands if self._matches(pattern, cmd)]

    def clear(self):
        """Clear executed commands history."""
        self.executed_commands.clear()
        self.execution_count = 0

    @property
    def last_command(self) -> Optional[str]:
        """Get the last executed command, or None if no commands executed."""
        return self.executed_commands[-1] if self.executed_commands else None
