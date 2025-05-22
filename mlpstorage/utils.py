import concurrent.futures
import enum
import io
import json
import logging
import math
import os
import pprint
import psutil
import subprocess
import shlex
import select
import signal
import sys
import threading
import yaml

from typing import List, Union, Optional, Dict, Tuple, Set

from mlpstorage.config import CONFIGS_ROOT_DIR, MPIRUN, MPIEXEC, MPI_RUN_BIN, MPI_EXEC_BIN


class MLPSJsonEncoder(json.JSONEncoder):

    def default(self, obj):
        try:
            if isinstance(obj, (float, int, str, list, tuple, dict)):
                return super().default(obj)
            if isinstance(obj, set):
                return list(obj)
            elif "Logger" in str(type(obj)):
                return "Logger object"
            elif 'ClusterInformation' in str(type(obj)):
                return obj.info
            elif isinstance(obj, enum.Enum):
                return obj.value
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                return super().default(obj)
        except Exception as e:
            return str(obj)


def is_valid_datetime_format(datetime_str):
    """
    Check if a string is a valid datetime in the format "yyyymmdd_hhMMss"
    
    :param datetime_str: String to check
    :return: True if the string is a valid datetime, False otherwise
    """
    try:
        # Check if the string has the correct length and format
        if len(datetime_str) != 15 or datetime_str[8] != '_':
            return False
        
        # Try to parse the datetime string
        parsed_datetime = datetime.strptime(datetime_str, "%Y%m%d_%H%M%S")
        return True
    except ValueError:
        # If parsing fails, the format is invalid
        return False


def get_datetime_from_timestamp(datetime_str):
    if is_valid_datetime_format(datetime_str):
        return datetime.strptime(datetime_str, "%Y%m%d_%H%M%S")
    else:
        return None


def read_config_from_file(relative_path):
    config_path = os.path.join(CONFIGS_ROOT_DIR, relative_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def update_nested_dict(original_dict, update_dict):
    updated_dict = {}
    for key, value in original_dict.items():
        if key in update_dict:
            if isinstance(value, dict) and isinstance(update_dict[key], dict):
                updated_dict[key] = update_nested_dict(value, update_dict[key])
            else:
                updated_dict[key] = update_dict[key]
        else:
            updated_dict[key] = value
    for key, value in update_dict.items():
        if key not in original_dict:
            updated_dict[key] = value
    return updated_dict


def create_nested_dict(flat_dict, parent_dict=None, separator='.'):
    if parent_dict is None:
        parent_dict = {}

    for key, value in flat_dict.items():
        keys = key.split(separator)
        current_dict = parent_dict
        for i, k in enumerate(keys[:-1]):
            if k not in current_dict:
                current_dict[k] = {}
            current_dict = current_dict[k]
        current_dict[keys[-1]] = value

    return parent_dict


def flatten_nested_dict(nested_dict, parent_key='', separator='.'):
    """
    Flatten a nested dictionary structure into a single-level dictionary with keys
    joined by a separator.

    Example:
        Input: {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
        Output: {'a': 1, 'b.c': 2, 'b.d.e': 3}

    Args:
        nested_dict (dict): The nested dictionary to flatten
        parent_key (str): The parent key prefix (used in recursion)
        separator (str): The character to use for joining keys

    Returns:
        dict: A flattened dictionary with compound keys
    """
    flat_dict = {}

    for key, value in nested_dict.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key

        if isinstance(value, dict):
            # Recursively flatten any nested dictionaries
            flat_dict.update(flatten_nested_dict(value, new_key, separator))
        else:
            # Add the leaf value to our flattened dictionary
            flat_dict[new_key] = value

    return flat_dict


def remove_nan_values(input_dict):
    # Remove any NaN values from the input dictionary
    ret_dict = dict()
    for k, v in input_dict.items():
        if type(v) in [float, int] and math.isnan(v):  # Ignore NaN values
            continue
        else:
            ret_dict[k] = v

    return ret_dict


class CommandExecutor:
    """
    A class to execute shell commands in a subprocess with live output streaming and signal handling.
    
    This class allows:
    - Executing commands as a string or list of arguments
    - Capturing stdout and stderr
    - Optionally printing stdout and stderr in real-time
    - Handling signals to gracefully terminate the process
    """
    
    def __init__(self, logger: logging.Logger, debug: bool = False):
        """
        Initialize the CommandExecutor.
        
        Args:
            debug: If True, enables debug mode with additional logging
        """
        self.logger = logger
        self.debug = debug
        self.process = None
        self.terminated_by_signal = False
        self.signal_received = None
        self._original_handlers = {}
        self._stop_event = threading.Event()
    
    def execute(self, 
                command: Union[str, List[str]], 
                print_stdout: bool = False,
                print_stderr: bool = False,
                watch_signals: Optional[Set[int]] = None) -> Tuple[str, str, int]:
        """
        Execute a command and return its stdout, stderr, and return code.
        
        Args:
            command: The command to execute (string or list of strings)
            print_stdout: If True, prints stdout in real-time
            print_stderr: If True, prints stderr in real-time
            watch_signals: Set of signals to watch for (e.g., {signal.SIGINT, signal.SIGTERM})
                          If any of these signals are received, the process will be terminated
        
        Returns:
            Tuple of (stdout_content, stderr_content, return_code)
        """

        self.logger.debug(f"DEBUG - Executing command: {command}")
        
        # Parse command if it's a string
        if isinstance(command, str):
            cmd_args = shlex.split(command)
        else:
            cmd_args = command
        
        # Set up signal handlers if requested
        if watch_signals:
            self._setup_signal_handlers(watch_signals)
        
        # Reset state
        self._stop_event.clear()
        self.terminated_by_signal = False
        self.signal_received = None
        
        # Initialize output buffers
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        return_code = None
        
        try:
            # Start the process
            self.process = subprocess.Popen(
                cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Get file descriptors for select
            stdout_fd = self.process.stdout.fileno()
            stderr_fd = self.process.stderr.fileno()
            
            # Process output until completion or signal
            while self.process.poll() is None and not self._stop_event.is_set():
                # Wait for output with timeout to allow checking for signals
                readable, _, _ = select.select(
                    [self.process.stdout, self.process.stderr], 
                    [], 
                    [], 
                    0.1
                )
                
                for stream in readable:
                    line = stream.readline()
                    if not line:  # EOF
                        continue
                        
                    if stream.fileno() == stdout_fd:
                        stdout_buffer.write(line)
                        if print_stdout:
                            sys.stdout.write(line)
                            sys.stdout.flush()
                    elif stream.fileno() == stderr_fd:
                        stderr_buffer.write(line)
                        if print_stderr:
                            sys.stderr.write(line)
                            sys.stderr.flush()
            
            # Read any remaining output
            stdout_remainder = self.process.stdout.read()
            if stdout_remainder:
                stdout_buffer.write(stdout_remainder)
                if print_stdout:
                    sys.stdout.write(stdout_remainder)
                    sys.stdout.flush()
                    
            stderr_remainder = self.process.stderr.read()
            if stderr_remainder:
                stderr_buffer.write(stderr_remainder)
                if print_stderr:
                    sys.stderr.write(stderr_remainder)
                    sys.stderr.flush()
            
            # Get the return code
            return_code = self.process.poll()
            
            # Check if we were terminated by a signal
            if self.terminated_by_signal:
                self.logger.debug(f"DEBUG - Process terminated by signal: {self.signal_received}")
                
            return stdout_buffer.getvalue(), stderr_buffer.getvalue(), return_code
            
        finally:
            # Clean up
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            
            # Restore original signal handlers
            self._restore_signal_handlers()
    
    def _setup_signal_handlers(self, signals: Set[int]):
        """Set up signal handlers for the specified signals."""
        self._original_handlers = {}
        
        def signal_handler(sig, frame):
            self.logger.debug(f"DEBUG - Received signal: {sig}")
            self.terminated_by_signal = True
            self.signal_received = sig
            self._stop_event.set()
            
            if self.process and self.process.poll() is None:
                self.process.terminate()

            for handler in self._original_handlers.values():
                handler(sig, frame)
        
        for sig in signals:
            self._original_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, signal_handler)
    
    def _restore_signal_handlers(self):
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)
        self._original_handlers = {}


def generate_mpi_prefix_cmd(mpi_cmd, hosts, num_processes, oversubscribe, allow_run_as_root, logger):
    # Check if we got slot definitions with the hosts:
    slots_configured = False
    for host in hosts:
        if ":" in host:
            slots_configured = True
            break

    if slots_configured:
        # Ensure the configured number of slots is >= num_processes
        num_slots = sum(int(slot) for host, slot in (host.split(":") for host in hosts))
        logger.debug(f"Configured slots: {num_slots}")
        if num_slots < num_processes:
            raise ValueError(f"Configured slots ({num_slots}) are not sufficient to run {num_processes} processes")
    elif not slots_configured:
        slotted_hosts = []
        # manually define how many slots per host to evenly distribute the processes across hosts. If num_processes
        # is not divisible by the number of hosts, distribute the remaining processes to the rest of the hosts.
        base_slots_per_host = num_processes // len(hosts)
        remaining_slots = num_processes % len(hosts)

        for i, host in enumerate(hosts):
            # Add one extra slot to hosts until we've distributed all remaining slots
            slots_for_this_host = base_slots_per_host + (1 if i < remaining_slots else 0)
            slotted_hosts.append(f"{host}:{slots_for_this_host}")

        # Replace the original hosts list with the slotted version
        hosts = slotted_hosts
        logger.debug(f"Configured slots for hosts: {hosts}")

    if mpi_cmd == MPIRUN:
        prefix = f"{MPI_RUN_BIN} -n {num_processes} -host {','.join(hosts)}"
    elif mpi_cmd == MPIEXEC:
        prefix = f"{MPI_EXEC_BIN} -n {num_processes} -host {','.join(hosts)}"
    else:
        raise ValueError(f"Unsupported MPI command: {mpi_cmd}")

    if oversubscribe:
        prefix += " --oversubscribe"

    if allow_run_as_root:
        prefix += " --allow-run-as-root"

    return prefix