"""Load backend connection parameters from environment variables and ``.env`` files.

Variable naming convention::

    {BACKEND_NAME}__{PARAM_NAME}

Both parts are **upper-cased** and separated by a **double underscore**.
The ``PARAM_NAME`` corresponds to a ``ParamDescriptor.name`` from the
backend's ``connection_params``, also upper-cased.

Examples::

    MILVUS__HOST=10.0.0.5
    MILVUS__PORT=19530
    PGVECTOR__PASSWORD=s3cret
    ELASTICSEARCH__API_KEY=abc123

If the `python-dotenv`_ package is installed, a ``.env`` file in the
current working directory (or the path given to :func:`load_env_file`) is
loaded automatically so that the variables are available via
``os.environ``.  When ``python-dotenv`` is not installed the module
falls back to reading ``os.environ`` directly (i.e. only real shell
environment variables are considered).

.. _python-dotenv: https://pypi.org/project/python-dotenv/
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BackendDescriptor

logger = logging.getLogger(__name__)

# Double underscore separates the backend name from the parameter name.
_SEP = "__"


# ------------------------------------------------------------------
# .env file loading
# ------------------------------------------------------------------

def load_env_file(path: Optional[str] = None) -> bool:
    """Load a ``.env`` file into ``os.environ``.

    Parameters
    ----------
    path : str, optional
        Explicit path to the ``.env`` file.  When *None*, ``python-dotenv``
        searches upward from the current working directory.

    Returns
    -------
    bool
        ``True`` if a ``.env`` file was loaded, ``False`` otherwise
        (including when ``python-dotenv`` is not installed).
    """
    try:
        from dotenv import load_dotenv, find_dotenv  # type: ignore[import-untyped]
    except ImportError:
        logger.debug(
            "python-dotenv is not installed; skipping .env file loading.  "
            "Install it with:  pip install python-dotenv"
        )
        return False

    dotenv_path = path or find_dotenv(usecwd=True)
    if not dotenv_path or not os.path.isfile(dotenv_path):
        logger.debug("No .env file found")
        return False

    load_dotenv(dotenv_path, override=False)
    logger.info("Loaded .env file: %s", dotenv_path)
    return True


# ------------------------------------------------------------------
# Type coercion
# ------------------------------------------------------------------

def _coerce(value: str, type_hint: str) -> Any:
    """Convert a string *value* to the Python type indicated by *type_hint*.

    Supported hints (matching ``ParamDescriptor.type``):
    ``"int"``, ``"float"``, ``"str"``, ``"bool"``.
    """
    type_hint = type_hint.lower()
    if type_hint == "int":
        return int(value)
    if type_hint == "float":
        return float(value)
    if type_hint == "bool":
        return value.lower() in ("1", "true", "yes", "on")
    return value  # "str" or anything else


# ------------------------------------------------------------------
# Read env vars for a backend
# ------------------------------------------------------------------

def env_for_backend(
    backend_name: str,
    desc: "BackendDescriptor",
) -> Dict[str, Any]:
    """Return a dict of connection parameters sourced from the environment.

    For each ``ParamDescriptor`` in *desc.connection_params*, the function
    looks for an environment variable named
    ``{BACKEND_NAME}__{PARAM_NAME}`` (both upper-cased, separated by a
    double underscore).

    Values are coerced to the type declared in ``ParamDescriptor.type``.
    Variables that are not set in the environment are omitted from the
    returned dict.

    Parameters
    ----------
    backend_name : str
        Short backend key (e.g. ``"milvus"``).
    desc : BackendDescriptor
        The backend's descriptor (used to enumerate connection params and
        their types).

    Returns
    -------
    dict[str, Any]
        Mapping of ``param_name -> coerced_value`` for every env var that
        was found.
    """
    prefix = backend_name.upper() + _SEP
    result: Dict[str, Any] = {}

    for param in desc.connection_params:
        env_key = prefix + param.name.upper()
        raw = os.environ.get(env_key)
        if raw is None:
            continue
        try:
            result[param.name] = _coerce(raw, param.type)
            logger.debug("Env var %s -> %s = %r", env_key, param.name, result[param.name])
        except (ValueError, TypeError) as exc:
            logger.warning(
                "Ignoring env var %s: could not coerce %r to %s: %s",
                env_key, raw, param.type, exc,
            )

    return result
