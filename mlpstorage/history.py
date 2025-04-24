import os
import shlex
import sys
from typing import Union

from mlpstorage.config import HISTFILE, DATETIME_STR, EXIT_CODE
from mlpstorage.logging import setup_logging


class HistoryTracker:
    """
    Tracks the history of command executions in a file.
    Each line contains sequence_id, datetime, and the full command separated by commas.
    """

    def __init__(self, history_file=None, logger=None):
        self.history_file = history_file or HISTFILE
        self.logger = logger or setup_logging(name="HistoryTracker", stream_log_level="INFO")
        self._ensure_history_file_exists()

    def _ensure_history_file_exists(self):
        """Create the history file if it doesn't exist."""
        if not os.path.exists(self.history_file):
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, 'w') as f:
                pass

    def _parse_history_line(self, line):
        """Parse a history line into sequence_id, timestamp, and command."""
        try:
            self.logger.debug(f"Parsing history line: {line}")
            sequence_id, timestamp, command = line.strip().split(',', 2)
            return int(sequence_id), timestamp, command
        except (ValueError, IndexError):
            self.logger.error(f"Invalid history line: {line}")
            return None

    def get_next_sequence_id(self):
        """Get the next sequence ID by reading the last line of the history file."""
        try:
            with open(self.history_file, 'r') as f:
                lines = f.readlines()
                if len(lines) == 0:
                    return 1

                last_line = lines[-1]
                if not last_line:
                    return 1

                try:
                    # Parse the sequence_id from the first part of the line
                    sequence_id, _, _ = self._parse_history_line(last_line)
                    return sequence_id + 1
                except (ValueError, IndexError):
                    return 1
        except FileNotFoundError:
            return 1

    def add_entry(self, command: Union[str, list[str]]):
        """
        Add a new entry to the history file.

        Args:
            command (str): The full command that was executed
        """
        if isinstance(command, list):
            command = ' '.join(command)

        sequence_id = self.get_next_sequence_id()
        timestamp = DATETIME_STR
        
        # Format the line as: sequence_id, timestamp, command
        history_line = f"{sequence_id},{timestamp},{command}"
        self.logger.verboser(f"Adding command to history: {history_line}")
        with open(self.history_file, 'a') as f:
            f.write(history_line + '\n')

        return sequence_id
        
    def get_command_by_id(self, sequence_id) -> Union[str, None]:
        """
        Retrieve a command by its sequence ID.
        
        Args:
            sequence_id (int): The sequence ID to look for
            
        Returns:
            str or None: The command string if found, None otherwise
        """
        try:
            with open(self.history_file, 'r') as f:
                for line in f:
                    self.logger.ridiculous(f"Parsing history line: {line}")
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        cid, _, command = self._parse_history_line(line)
                        if sequence_id == cid:
                            return command
                    except (ValueError, IndexError):
                        continue

            self.logger.debug(f"Command not found for sequence_id: {sequence_id}")
            return None  # Command not found
        except FileNotFoundError:
            return None
            
    def get_history_entries(self, limit=None):
        """
        Retrieve history entries, optionally limited to a specific number.
        
        Args:
            limit (int, optional): Maximum number of entries to return, starting from most recent
            
        Returns:
            list: List of tuples containing (sequence_id, timestamp, command)
        """
        entries = []
        try:
            with open(self.history_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sequence_id, timestamp, command = self._parse_history_line(line)
                        entries.append((sequence_id, timestamp, command))
                    except (ValueError, IndexError):
                        continue
                        
            # Return the most recent entries if limit is specified
            if limit is not None and limit > 0:
                return entries[-limit:]
            return entries
        except FileNotFoundError:
            return []
            
    def print_history(self, limit=None, sequence_id=None):
        """
        Print history entries to stdout.
        
        Args:
            limit (int, optional): Maximum number of entries to print, starting from most recent
            sequence_id (int, optional): Specific sequence ID to print
            
        Returns:
            bool: True if entries were found and printed, False otherwise
        """
        if sequence_id is not None:
            command = self.get_command_by_id(sequence_id)
            if command:
                print(f"{sequence_id}: {command}")
                return EXIT_CODE.SUCCESS
            else:
                print(f"No command found with ID {sequence_id}")
                return EXIT_CODE.INVALID_ARGUMENTS
                
        entries = self.get_history_entries(limit)
        if not entries:
            print("No history entries found")
            return EXIT_CODE.INVALID_ARGUMENTS
            
        for seq_id, timestamp, command in entries:
            print(f"{seq_id} : {timestamp} : {command}")

        return EXIT_CODE.SUCCESS
        
    def create_args_from_command(self, sequence_id):
        """
        Create an args object from a command in the history.
        
        Args:
            sequence_id (int): The sequence ID of the command to use
            
        Returns:
            argparse.Namespace or None: The args object if command found, None otherwise
        """
        command = self.get_command_by_id(sequence_id)
        if not command:
            return None
            
        # Remove the script name if present
        command_parts = shlex.split(command)
        if command_parts and os.path.basename(command_parts[0]) == os.path.basename(sys.argv[0]):
            command_parts = command_parts[1:]
            
        # Import here to avoid circular imports
        from mlpstorage.cli import parse_arguments
        
        # Save original argv and restore after parsing
        original_argv = sys.argv
        try:
            sys.argv = [original_argv[0]] + command_parts
            args = parse_arguments()
            return args
        except Exception as e:
            print(f"Error parsing command: {e}")
            return None

    def handle_history_command(self, args):
        """
        Handle the history command based on CLI arguments.
        
        Args:
            args: The parsed command-line arguments
            
        Returns:
            int: Exit code (0 for success, non-zero for failure)
        """
        if hasattr(args, 'id') and args.id is not None:
            # Print specific history entry
            return self.print_history(sequence_id=args.id)
        elif hasattr(args, 'limit') and args.limit is not None:
            # Print limited history entries
            return self.print_history(limit=args.limit)
        elif hasattr(args, 'rerun_id') and args.rerun_id is not None:
            # Return args from a specific history entry
            new_args = self.create_args_from_command(args.rerun_id)
            if new_args is None:
                print(f"Command with ID {args.rerun_id} not found or could not be parsed")
                return EXIT_CODE.INVALID_ARGUMENTS
            return new_args
        else:
            # Print all history entries
            if not self.print_history():
                return EXIT_CODE.GENERAL_ERROR
                
        return EXIT_CODE.SUCCESS