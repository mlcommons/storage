"""
One layered resolver for every default that does not come from argparse.

Every key is filled in the same order, and the winning tier is recorded on
``args.config_sources[dest]`` so ``main`` can say where each value came from:

    1. the flag typed on the command line
    2. the ``--config-file`` YAML named on this command line
    3. the ``MLPSTORAGE_*`` environment variable (four keys have one)
    4. the per-user file ``~/.config/mlpstorage/config.yaml``
    5. argparse's built-in default

Tier 2 may carry any flag of the leaf command (it is explicit, per
invocation: the README's repeatable-knobs example with ``accelerator_type``
and ``params`` stays valid). Tier 4 is restricted to the keys in
:data:`mlpstorage_py.results_dir.user_config.USER_CONFIG_KEYS` — flags that
describe the *environment* — and a workload-selecting key there is a hard
error, so a stale file can never reshape a run.

argparse cannot say whether a value was typed or defaulted, so
:func:`explicit_dests` re-parses the same command line against a probe copy
of the parser whose every default is a sentinel: whatever is not the
sentinel afterwards was on the command line.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, FrozenSet, Optional, Tuple

import yaml

from mlpstorage_py.config import EXIT_CODE
from mlpstorage_py.errors import ConfigurationError
from mlpstorage_py.results_dir.user_config import (
    ENV_BACKED_KEYS,
    SOURCE_CONFIG_FILE_FLAG,
    USER_CONFIG_KEYS,
    read_user_config,
    user_config_path,
)

_SENTINEL = object()

# Actions whose __call__ reads the current value back (append/count) cannot
# start from an arbitrary object; None is their natural "not supplied".
_ACCUMULATING = (
    argparse._AppendAction,
    argparse._AppendConstAction,
    argparse._CountAction,
)
try:  # 3.8+
    _ACCUMULATING += (argparse._ExtendAction,)
except AttributeError:  # pragma: no cover
    pass

_SKIP = (argparse._HelpAction, argparse._VersionAction)


def _build_probe():
    from mlpstorage_py.cli_parser import build_parser

    probe = build_parser()
    stack = [probe]
    while stack:
        parser = stack.pop()
        parser._defaults = {}  # set_defaults() seeds are never "typed"
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                stack.extend(action.choices.values())
            elif isinstance(action, _SKIP) or action.dest is argparse.SUPPRESS:
                continue
            elif isinstance(action, _ACCUMULATING):
                action.default = None
            else:
                action.default = _SENTINEL
    return probe


def _path_actions(probe, namespace) -> Dict[str, argparse.Action]:
    """``dest -> Action`` for every parser on the path the command line took
    (top level down to the leaf), leaf definitions winning."""
    actions: Dict[str, argparse.Action] = {}
    parser = probe
    while parser is not None:
        nxt = None
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                chosen = getattr(namespace, action.dest, None)
                nxt = action.choices.get(chosen) if isinstance(chosen, str) else None
            elif not isinstance(action, _SKIP) and action.dest is not argparse.SUPPRESS:
                actions[action.dest] = action
        parser = nxt
    return actions


def probe(argv) -> Tuple[FrozenSet[str], Dict[str, argparse.Action]]:
    """``(explicit dests, dest -> Action on the path)`` for ``argv``
    (``sys.argv[1:]`` shape; must already have parsed successfully)."""
    p = _build_probe()
    namespace, _ = p.parse_known_args(list(argv))
    explicit = frozenset(
        dest for dest, value in vars(namespace).items()
        if value is not _SENTINEL and value is not None
    )
    return explicit, _path_actions(p, namespace)


def explicit_dests(argv) -> FrozenSet[str]:
    """The dests the command line supplied itself."""
    return probe(argv)[0]


_ALL_DESTS: Optional[FrozenSet[str]] = None


def all_parser_dests() -> FrozenSet[str]:
    """Every dest any command in the tree defines (cached)."""
    global _ALL_DESTS
    if _ALL_DESTS is None:
        from mlpstorage_py.cli_parser import build_parser

        found = set()
        stack = [build_parser()]
        while stack:
            parser = stack.pop()
            for action in parser._actions:
                if isinstance(action, argparse._SubParsersAction):
                    stack.extend(action.choices.values())
                    if action.dest is not argparse.SUPPRESS:
                        found.add(action.dest)  # mode / benchmark / model / command
                elif not isinstance(action, _SKIP) and action.dest is not argparse.SUPPRESS:
                    found.add(action.dest)
        _ALL_DESTS = frozenset(found)
    return _ALL_DESTS


# --------------------------------------------------------------------------- #
# Value coercion: YAML scalars/lists -> what argparse would have produced      #
# --------------------------------------------------------------------------- #


def _fail(origin: str, message: str) -> None:
    print(f"error: {origin}: {message}", file=sys.stderr)
    sys.exit(EXIT_CODE.INVALID_ARGUMENTS)


def _warn(origin: str, message: str) -> None:
    print(f"warning: {origin}: {message}", file=sys.stderr)


def coerce(key: str, value, action: Optional[argparse.Action], origin: str):
    """Turn a YAML value into the shape ``action`` would have stored.

    ``origin`` names the file for error messages. ``action`` may be ``None``
    for dests that exist only via ``set_defaults`` — the value is then taken
    as written.
    """
    if action is None:
        return value

    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        if not isinstance(value, bool):
            _fail(origin, f"key {key!r} must be true or false (got {value!r})")
        return value

    if key == "params":
        # Dict form (recommended) or the CLI's list of "k=v" strings.
        if isinstance(value, dict):
            return [f"{k}={v}" for k, v in value.items()]
        if isinstance(value, list):
            return [str(v) for v in value]
        _fail(origin, f"key 'params' must be a mapping or a list of key=value strings")

    if isinstance(action, _ACCUMULATING):
        # append-style flags (--mpi-params): one string or a list of them.
        if isinstance(value, str):
            return [value]
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return list(value)
        _fail(origin, f"key {key!r} must be a string or a list of strings (got {value!r})")

    if action.nargs in ("+", "*") or isinstance(action.nargs, int):
        # list-valued flags (--hosts): a list, or one comma/space separated string.
        if isinstance(value, str):
            items = [tok for tok in re.split(r"[,\s]+", value.strip()) if tok]
        elif isinstance(value, list):
            items = value
        else:
            _fail(origin, f"key {key!r} must be a list (got {value!r})")
        out = []
        for item in items:
            if action.type is not None and isinstance(item, str):
                item = _convert(key, item, action, origin)
            elif not isinstance(item, str):
                item = str(item)
            out.append(item)
        return out

    if isinstance(value, bool) and action.type is not bool:
        _fail(origin, f"key {key!r} must not be a boolean (got {value!r})")
    if action.type is not None:
        value = _convert(key, value, action, origin)
    elif not isinstance(value, str) and value is not None:
        # A plain string flag: accept scalars by their YAML spelling.
        value = str(value)

    if action.choices is not None and value not in action.choices:
        allowed = ", ".join(str(c) for c in action.choices)
        _fail(origin, f"key {key!r} value {value!r} is not one of {allowed}")
    return value


def _convert(key, value, action, origin):
    try:
        return action.type(value)
    except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
        _fail(origin, f"key {key!r} value {value!r} is invalid: {exc}")


def _flag_label(action: Optional[argparse.Action], dest: str) -> str:
    if action is not None and action.option_strings:
        return action.option_strings[0]
    return dest


# --------------------------------------------------------------------------- #
# The tiers                                                                    #
# --------------------------------------------------------------------------- #


def load_override_file(path: str) -> Optional[dict]:
    """The ``--config-file`` mapping, or ``None`` when the file is empty.
    Exits with INVALID_ARGUMENTS on a missing or malformed file."""
    try:
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError:
        print(f"Error: Config file {path} not found")
        sys.exit(EXIT_CODE.INVALID_ARGUMENTS)
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML config file: {exc}")
        sys.exit(EXIT_CODE.INVALID_ARGUMENTS)
    if not data:
        print(f"Warning: Config file {path} is empty or invalid")
        return None
    if not isinstance(data, dict):
        print(f"Error: Config file {path} must be a YAML mapping")
        sys.exit(EXIT_CODE.INVALID_ARGUMENTS)
    return data


def apply_override_file(args, explicit=frozenset(), actions=None, sources=None):
    """Tier 2: fill every dest the ``--config-file`` YAML names and the
    command line did not type. Unknown keys warn; ``null`` values are
    skipped so a file cannot blank a flag."""
    path = getattr(args, "config_file", None)
    if not path:
        return args
    data = load_override_file(path)
    if data is None:
        return args
    actions = actions or {}
    args_dict = vars(args)
    for key, value in data.items():
        if key not in args_dict:
            print(f"Warning: Config file contains unknown parameter '{key}', skipping")
            continue
        if value is None:
            continue
        if key in explicit:
            continue
        action = actions.get(key)
        if action is None and key == "hosts" and isinstance(value, str):
            value = value.split(",")
        elif action is None and key == "params" and isinstance(value, dict):
            value = [f"{k}={v}" for k, v in value.items()]
        elif action is None and key == "params":
            print("Warning: Invalid format for 'params' in config file, skipping")
            continue
        else:
            value = coerce(key, value, action, f"--config-file {path}")
        args_dict[key] = value
        if sources is not None:
            sources[key] = SOURCE_CONFIG_FILE_FLAG
    return args


def _validate_user_config_keys(values: dict, path: str) -> None:
    """A workload-selecting key is a hard error; an unknown key a warning."""
    tree_dests = all_parser_dests()
    for key in values:
        if key in USER_CONFIG_KEYS:
            continue
        if key in tree_dests:
            _fail(
                path,
                f"{key!r} selects the workload and is not read from the per-user "
                f"config file; pass it on the command line or in a --config-file "
                f"YAML. Keys this file may carry: {', '.join(USER_CONFIG_KEYS)}",
            )
        _warn(path, f"unknown key {key!r} ignored")


def apply_config_layers(args, argv) -> argparse.Namespace:
    """Apply tiers 2-4 on top of the parsed ``args`` for ``argv``
    (``sys.argv[1:]``), recording ``args.config_sources``."""
    explicit, actions = probe(argv)
    sources: Dict[str, str] = {
        dest: _flag_label(actions.get(dest), dest) for dest in explicit
    }

    apply_override_file(args, explicit, actions, sources)

    user_path = user_config_path()
    try:
        user_values = read_user_config(user_path)
    except ConfigurationError as exc:
        _fail(user_path, str(exc))
    _validate_user_config_keys(user_values, user_path)

    args_dict = vars(args)
    for key in USER_CONFIG_KEYS:
        if key not in args_dict or key in sources:
            continue
        env_name = ENV_BACKED_KEYS.get(key)
        env_value = os.environ.get(env_name, "") if env_name else ""
        if env_value:
            args_dict[key] = env_value
            sources[key] = env_name
            continue
        value = user_values.get(key)
        if value is not None:
            if key in ("results_dir", "data_dir", "checkpoint_folder", "dlio_bin_path") \
                    and isinstance(value, str):
                value = os.path.expanduser(value)
            args_dict[key] = coerce(key, value, actions.get(key), user_path)
            sources[key] = user_path
            continue
        if env_name and args_dict.get(key):
            # argparse's default captured the env var at import time.
            sources[key] = env_name

    args.config_sources = sources
    args.user_config_path = user_path
    return args


def keys_from(args, source: str):
    """Sorted dests whose value ``source`` supplied. ``results_dir`` is
    reported on its own status line and ``config_file`` is the flag whose
    label doubles as the tier-2 source, so both are left out."""
    sources = getattr(args, "config_sources", None) or {}
    return sorted(
        k for k, s in sources.items()
        if s == source and k not in ("results_dir", "config_file")
    )
