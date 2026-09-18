"""
Per-user mlpstorage config file and results-dir resolution.

``mlpstorage init`` records the results-dir it initialized in
``$XDG_CONFIG_HOME/mlpstorage/config.yaml`` (``~/.config/mlpstorage/config.yaml``
when ``XDG_CONFIG_HOME`` is unset or relative). Every later command resolves
its results-dir through :func:`resolve_results_dir`, in this order:

1. ``--results-dir`` on the command line (or ``results_dir`` in a
   ``--config-file`` YAML, which the parser applies on top of the flag);
2. the ``MLPSTORAGE_RESULTS_DIR`` environment variable;
3. ``results_dir`` in the per-user config file.

The winning tier is reported alongside the path so ``main`` can print
``results-dir: <path> (from <source>)`` and the choice is never silent.

The config file holds defaults for flags that describe the *environment*
(where results go, later: systemname, data dirs, hosts, MPI settings). It
never supplies workload-selecting flags (model, accelerator, mode), so a
stale file cannot reshape a run. Only ``results_dir`` is read today.

The file is written atomically (tmp + ``os.replace``) and existing keys are
preserved, so hand-added keys survive the next ``mlpstorage init``.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import yaml

from mlpstorage_py.config import MLPSTORAGE_RESULTS_DIR_ENVVAR
from mlpstorage_py.errors import ConfigurationError, ErrorCode

USER_CONFIG_DIRNAME = "mlpstorage"
USER_CONFIG_FILENAME = "config.yaml"
RESULTS_DIR_KEY = "results_dir"

#: Where ``mlpstorage init`` puts the tree when no ``[path]`` is given.
DEFAULT_RESULTS_DIR = "~/mlpstorage-results"

#: Source labels returned by :func:`resolve_results_dir`.
SOURCE_CLI = "--results-dir"
SOURCE_CONFIG_FILE_FLAG = "--config-file"
SOURCE_ENV = MLPSTORAGE_RESULTS_DIR_ENVVAR

_HEADER = (
    "# Per-user mlpstorage defaults. Written by `mlpstorage init`; hand edits\n"
    "# are kept. Precedence: command-line flag > MLPSTORAGE_* env var > this file.\n"
)


def user_config_dir() -> str:
    """``$XDG_CONFIG_HOME/mlpstorage`` or ``~/.config/mlpstorage``.

    Per the XDG Base Directory spec a relative ``XDG_CONFIG_HOME`` is
    invalid and ignored.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME", "")
    if xdg and os.path.isabs(xdg):
        base = xdg
    else:
        base = os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, USER_CONFIG_DIRNAME)


def user_config_path() -> str:
    return os.path.join(user_config_dir(), USER_CONFIG_FILENAME)


def default_results_dir() -> str:
    return os.path.expanduser(DEFAULT_RESULTS_DIR)


def read_user_config(path: Optional[str] = None) -> dict:
    """Return the config mapping, or ``{}`` when the file does not exist.

    Raises:
        ConfigurationError: the file exists but is not a YAML mapping.
    """
    path = path or user_config_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"user config {path} is not valid YAML: {exc}",
            suggestion=f"Fix or delete {path}, then re-run `mlpstorage init`.",
            code=ErrorCode.CONFIG_INVALID_VALUE,
        ) from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"user config {path} must be a YAML mapping "
            f"(got {type(data).__name__})",
            suggestion=f"Fix or delete {path}, then re-run `mlpstorage init`.",
            code=ErrorCode.CONFIG_INVALID_VALUE,
        )
    return data


def write_user_config(values: dict, path: Optional[str] = None) -> str:
    """Merge ``values`` into the config file and write it atomically."""
    path = path or user_config_path()
    merged = dict(read_user_config(path))
    merged.update(values)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(_HEADER)
            yaml.safe_dump(merged, fh, default_flow_style=False, sort_keys=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def record_results_dir(results_dir: str, path: Optional[str] = None) -> str:
    """Record ``results_dir`` (made absolute) as the per-user default."""
    return write_user_config({RESULTS_DIR_KEY: os.path.abspath(results_dir)}, path)


def configured_results_dir(path: Optional[str] = None) -> Optional[str]:
    """The ``results_dir`` recorded in the config file, ``~`` expanded."""
    value = read_user_config(path).get(RESULTS_DIR_KEY)
    if not value:
        return None
    if not isinstance(value, str):
        raise ConfigurationError(
            f"user config key {RESULTS_DIR_KEY!r} must be a path string "
            f"(got {type(value).__name__})",
            suggestion="Re-run `mlpstorage init <orgname> <path>` to rewrite it.",
            code=ErrorCode.CONFIG_INVALID_VALUE,
        )
    return os.path.expanduser(value)


def resolve_results_dir(
    cli_value: Optional[str],
    *,
    cli_source: str = SOURCE_CLI,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve the results-dir for this invocation.

    Args:
        cli_value: the value given explicitly (``--results-dir`` or a
            ``--config-file`` YAML), or ``None``/``""`` when neither was.
        cli_source: label to report when ``cli_value`` wins.

    Returns:
        ``(path, source)``. ``source`` is ``cli_source``,
        ``"MLPSTORAGE_RESULTS_DIR"`` or the config file path. Both are
        ``None`` when nothing supplies a results-dir.
    """
    if cli_value:
        return cli_value, cli_source
    env_value = os.environ.get(MLPSTORAGE_RESULTS_DIR_ENVVAR, "")
    if env_value:
        return env_value, SOURCE_ENV
    config_path = user_config_path()
    configured = configured_results_dir(config_path)
    if configured:
        return configured, config_path
    return None, None


#: One line that says how to get a results-dir, used by every "no
#: results-dir" error so the advice is identical everywhere.
HOW_TO_SUPPLY = (
    "run `mlpstorage init <orgname> [path]` once to record a default, "
    "pass it on the command line, or set MLPSTORAGE_RESULTS_DIR"
)
