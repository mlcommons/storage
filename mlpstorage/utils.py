import concurrent.futures
import enum
import io
import json
import logging
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

from mlpstorage.config import CONFIGS_ROOT_DIR, MPIRUN, MPIEXEC, MPI_RUN_BIN


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
        
        for sig in signals:
            self._original_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, signal_handler)
    
    def _restore_signal_handlers(self):
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)
        self._original_handlers = {}


class ClusterInformation:
    def __init__(self, hosts, username, debug=False):
        self.debug = debug
        self.hosts = hosts
        self.username = username
        self.info = dict(
            host_info={},
            accumulated_mem_info_bytes={},
            total_client_cpus=0,
        )

        if len(hosts) == 1 and hosts[0] in ['localhost', '127.0.0.1']:
            self.collect_info(local=True)
        else:
            self.collect_info(local=False)

    def __str__(self):
        return pprint.pformat(self.info)

    def __repr__(self):
        return str(self.info)

    def collect_info(self, local=False):
        if local:
            getter_func = self.get_local_info
        else:
            getter_func = self.get_remote_info

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_host = {executor.submit(getter_func, host): host for host in self.hosts}
            for future in concurrent.futures.as_completed(future_to_host):
                host = future_to_host[future]
                cpu_core_count, memory_info = future.result()
                self.info["host_info"][host] = {
                    'cpu_core_count': cpu_core_count,
                    'memory_info': memory_info
                }

        for host_info in self.info["host_info"].values():
            self.info["total_client_cpus"] += host_info['cpu_core_count']
            for mem_key in host_info['memory_info']:
                if mem_key not in self.info["accumulated_mem_info_bytes"]:
                    self.info["accumulated_mem_info_bytes"][mem_key] = 0
                self.info["accumulated_mem_info_bytes"][mem_key] += host_info['memory_info'][mem_key]

    @staticmethod
    def get_local_info(host=None):
        cpu_core_count = psutil.cpu_count(logical=False)
        memory_info = dict(psutil.virtual_memory()._asdict())
        return cpu_core_count, memory_info

    def get_remote_info(self, host):
        # TODO: Add proper handling if passwordless-ssh is not conffigured.
        cpu_core_count = self.get_remote_cpu_core_count(host)
        memory_info = self.get_remote_memory_info(host)
        return cpu_core_count, memory_info

    def get_remote_cpu_core_count(self, host):
        cpu_core_count = 0
        cpu_info_path = f"ssh {self.username}@{host} cat /proc/cpuinfo"
        try:
            output = os.popen(cpu_info_path).read()
            cpu_core_count = output.count('processor')
        except Exception as e:
            print(f"Error getting CPU core count for host {host}: {e}")
        return cpu_core_count

    def get_remote_memory_info(self, host):
        memory_info = {}
        meminfo_path = f"ssh {self.username}@{host} cat /proc/meminfo"
        try:
            output = os.popen(meminfo_path).read()
            lines = output.split('\n')
            for line in lines:
                if line.startswith('MemTotal:'):
                    memory_info['total'] = int(line.split()[1]) * 1024
                elif line.startswith('MemFree:'):
                    memory_info['free'] = int(line.split()[1]) * 1024
                elif line.startswith('MemAvailable:'):
                    memory_info['available'] = int(line.split()[1]) * 1024
        except Exception as e:
            print(f"Error getting memory information for host {host}: {e}")
        return memory_info


def generate_mpi_prefix_cmd(mpi_cmd, hosts, num_processes, oversubscribe, allow_run_as_root):
    if mpi_cmd == MPIRUN:
        prefix = f"{MPI_RUN_BIN} -n {num_processes} -host {','.join(hosts)}"
    elif mpi_cmd == MPIEXEC:
        raise NotImplementedError(f"Unsupported MPI command: {mpi_cmd}")
    else:
        raise ValueError(f"Unsupported MPI command: {mpi_cmd}")

    if oversubscribe:
        prefix += " --oversubscribe"

    if allow_run_as_root:
        prefix += " --allow-run-as-root"

    return prefix