"""Centralized run configuration summary for MLPerf Storage.

Provides print_run_summary(args), which formats and emits a structured
table of effective Tier 1 CLI parameters and environment variables
immediately before benchmark execution.

NOTE: .env file loading happens in _apply_object_storage_params(), which
runs after run_benchmark(). This summary shows pre-.env-load env state
— by design.
"""

import os

from mlpstorage_py import VERSION
from mlpstorage_py.mlps_logging import setup_logging
from mlpstorage_py.storage_config import resolve_object_storage_config

logger = setup_logging("MLPerfStorage")

# Label column width
_WIDTH = 32


def _row(label: str, value) -> str:
    """Return a formatted label/value row string.

    Args:
        label: Column label (left-justified to _WIDTH chars).
        value: Value to display (converted to str).

    Returns:
        Indented "  label<pad>value" string.
    """
    return f"  {label:<{_WIDTH}}{value}"


def print_run_summary(args) -> None:
    """Print a structured table of Tier 1 CLI args and env vars via logger.status().

    The summary is printed immediately before benchmark execution.  When the
    data_access_protocol is 'object', a second section with S3 environment
    variable values is appended.

    Credentials are never shown as plain text — resolve_object_storage_config()
    pre-redacts them before returning.

    Args:
        args: argparse Namespace (or compatible object).  All attribute access
              uses getattr() with safe defaults so this function is safe to call
              regardless of which subcommand populated ``args``.
    """
    # Guard: suppress entirely when --quiet is passed.
    if getattr(args, 'quiet', False):
        return

    lines = ["", f"--- Run Configuration (mlpstorage {VERSION}) ---"]

    # Tier 1 CLI args — use getattr so absent attrs are '[not set]' not AttributeError.
    _tier1 = [
        ("benchmark",                 getattr(args, 'benchmark',                 None)),
        ("command",                   getattr(args, 'command',                   None)),
        ("mode",                      getattr(args, 'mode',                      None)),
        ("data_dir",                  getattr(args, 'data_dir',                  None)),
        ("results_dir",               getattr(args, 'results_dir',               None)),
        ("data_access_protocol",      getattr(args, 'data_access_protocol',      None)),
        ("num_accelerators",          getattr(args, 'num_accelerators',          None)),
        ("num_processes",             getattr(args, 'num_processes',             None)),
        ("accelerator_type",          getattr(args, 'accelerator_type',          None)),
        ("client_host_memory_in_gb",  getattr(args, 'client_host_memory_in_gb',  None)),
        ("hosts",                     getattr(args, 'hosts',                     None)),
        ("exec_type",                 getattr(args, 'exec_type',                 None)),
        ("mpi_bin",                   getattr(args, 'mpi_bin',                   None)),
        ("loops",                     getattr(args, 'loops',                     None)),
    ]
    for label, val in _tier1:
        display = val if val is not None else '[not set]'
        lines.append(_row(label + ":", display))

    # Always-visible environment section.
    lines.append("")
    lines.append("--- Environment ---")
    lines.append(_row("MLPERF_RESULTS_DIR:", os.environ.get('MLPERF_RESULTS_DIR', '[not set]')))
    lines.append(_row("MPI_RUN_BIN:",        os.environ.get('MPI_RUN_BIN',        '[not set]')))
    lines.append(_row("MPI_EXEC_BIN:",       os.environ.get('MPI_EXEC_BIN',       '[not set]')))

    # Object storage section — only when protocol is explicitly 'object'.
    if getattr(args, 'data_access_protocol', None) == 'object':
        config = resolve_object_storage_config()
        endpoint_val, endpoint_src = config['endpoint']
        if endpoint_val:
            endpoint_display = f"{endpoint_val}  [from {endpoint_src}]"
        else:
            endpoint_display = '[not set]'

        lines.append("")
        lines.append("--- Object Storage (S3) ---")
        lines.append(_row("bucket:",                config['bucket'] or '[not set]'))
        lines.append(_row("storage_library:",       config['storage_library']))
        lines.append(_row("uri_scheme:",            config['uri_scheme']))
        lines.append(_row("endpoint:",              endpoint_display))
        lines.append(_row("load_balance_strategy:", config['load_balance_strategy']))
        lines.append(_row("aws_region:",            config['aws_region']))
        lines.append(_row("aws_ca_bundle:",         config['aws_ca_bundle'] or '[not set]'))
        lines.append(_row("AWS_ACCESS_KEY_ID:",     config['aws_access_key_id_redacted']))
        lines.append(_row("AWS_SECRET_ACCESS_KEY:", config['aws_secret_access_key_redacted']))

    lines.append("")

    for line in lines:
        logger.status(line)
