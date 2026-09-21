"""
``mlpstorage config show | set | unset | path``.

The per-user file (:mod:`mlpstorage_py.results_dir.user_config`) supplies
environment defaults to every command. Until this command existed only
``mlpstorage init`` wrote it, and only ``results_dir``; every other key was
a hand edit, and a mistaken workload key made every command fail at parse
time with nothing but an editor to fix it. ``config`` is dispatched before
the tier resolver in ``cli.config_layers`` runs, so it works on a file the
resolver refuses.

``set`` validates exactly as the file-read path does: the key must be in
:data:`USER_CONFIG_KEYS` (a workload-selecting key is the same hard error
the resolver gives, an unknown key an error too since the user typed it),
and the value goes through :func:`config_layers.coerce` with the flag's
real argparse action, so type and choice mistakes are caught here rather
than on the next benchmark command.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import yaml

from mlpstorage_py.config import EXIT_CODE
from mlpstorage_py.results_dir.user_config import (
    ENV_BACKED_KEYS,
    HOW_TO_SUPPLY,
    RESULTS_DIR_KEY,
    USER_CONFIG_KEYS,
    read_user_config,
    remove_user_config_key,
    user_config_path,
    write_user_config,
)

_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}

_ACCUMULATING = (argparse._AppendAction, argparse._AppendConstAction, argparse._CountAction)
try:  # 3.8+
    _ACCUMULATING += (argparse._ExtendAction,)
except AttributeError:  # pragma: no cover
    pass


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def action_for(dest: str) -> Optional[argparse.Action]:
    """The first argparse action in the command tree with this ``dest``.

    Every :data:`USER_CONFIG_KEYS` flag is declared identically wherever it
    appears (they all come from ``cli/common_args.py``), so the first one
    found is as good as any.
    """
    from mlpstorage_py.cli_parser import build_parser

    stack = [build_parser()]
    while stack:
        parser = stack.pop()
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                stack.extend(action.choices.values())
            elif action.dest == dest:
                return action
    return None


def _fail(message: str) -> None:
    from mlpstorage_py.cli.config_layers import _fail as fail

    fail("config set", message)


def _check_key(key: str) -> None:
    from mlpstorage_py.cli.config_layers import all_parser_dests

    if key in USER_CONFIG_KEYS:
        return
    allowed = ", ".join(USER_CONFIG_KEYS)
    if key in all_parser_dests():
        _fail(
            f"{key!r} selects the workload and is not read from the per-user "
            f"config file; pass it on the command line or in a --config-file "
            f"YAML. Keys this file may carry: {allowed}"
        )
    _fail(f"unknown key {key!r}. Keys this file may carry: {allowed}")


def shape_value(key: str, values: List[str], action: Optional[argparse.Action]) -> Any:
    """Turn the command-line words into the YAML value the file stores.

    One word stores a scalar, several a list; ``hosts`` also splits one
    comma-separated word. Boolean keys take true/false spellings. The
    result is then run through :func:`config_layers.coerce`, which applies
    the flag's own type and choice checks and exits on a mistake.
    """
    from mlpstorage_py.cli.config_layers import coerce

    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        if len(values) != 1:
            _fail(f"key {key!r} takes one value: true or false")
        word = values[0].strip().lower()
        if word in _TRUE:
            stored: Any = True
        elif word in _FALSE:
            stored = False
        else:
            _fail(f"key {key!r} must be true or false (got {values[0]!r})")
    elif isinstance(action, _ACCUMULATING):
        stored = values[0] if len(values) == 1 else list(values)
    elif action is not None and (action.nargs in ("+", "*") or isinstance(action.nargs, int)):
        stored = values[0] if len(values) == 1 else list(values)
        if isinstance(stored, str) and "," in stored:
            stored = [tok.strip() for tok in stored.split(",") if tok.strip()]
        elif isinstance(stored, str):
            stored = [stored]
    else:
        if len(values) != 1:
            _fail(f"key {key!r} takes one value (got {len(values)})")
        stored = values[0]

    coerce(key, stored, action, "config set")  # validates; exits on a mistake
    return stored


def _yaml_scalar(value: Any) -> str:
    """One-line YAML spelling of a stored value (``[unset]`` for none)."""
    if value is None:
        return "[unset]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    return yaml.safe_dump(value, default_flow_style=True, width=10**6).strip()


# --------------------------------------------------------------------------- #
# the report                                                                   #
# --------------------------------------------------------------------------- #


def config_report(path: Optional[str] = None) -> Dict[str, Any]:
    """Everything ``show`` prints, as data.

    ``keys`` covers every allowed key: the file's value (``None`` when
    unset) and the ``MLPSTORAGE_*`` env var that outranks it when one is
    set. ``rejected`` holds workload-selecting keys the file must not carry
    (the tier resolver refuses the whole file while they are present);
    ``ignored`` holds keys no command knows.
    """
    from mlpstorage_py.cli.config_layers import all_parser_dests

    path = path or user_config_path()
    exists = os.path.isfile(path)
    values = read_user_config(path) if exists else {}

    keys: Dict[str, Dict[str, Any]] = {}
    for key in USER_CONFIG_KEYS:
        env_name = ENV_BACKED_KEYS.get(key)
        env_value = os.environ.get(env_name, "") if env_name else ""
        keys[key] = {
            "value": values.get(key),
            "env": {"name": env_name, "value": env_value} if env_value else None,
        }

    tree_dests = all_parser_dests()
    rejected = {k: v for k, v in values.items()
                if k not in USER_CONFIG_KEYS and k in tree_dests}
    ignored = {k: v for k, v in values.items()
               if k not in USER_CONFIG_KEYS and k not in tree_dests}
    return {"path": path, "exists": exists, "keys": keys,
            "rejected": rejected, "ignored": ignored}


def format_report(report: Dict[str, Any]) -> str:
    lines = [report["path"] + ("" if report["exists"] else "  [not present]")]
    width = max(len(k) for k in report["keys"]) + 2
    for key, entry in report["keys"].items():
        line = f"  {key:<{width}}{_yaml_scalar(entry['value'])}"
        env = entry["env"]
        if env:
            if entry["value"] is None:
                line += f"  ({env['name']}={env['value']} supplies it)"
            else:
                line += f"  (outranked by {env['name']}={env['value']})"
        lines.append(line)
    if report["rejected"]:
        lines.append("")
        for key, value in report["rejected"].items():
            lines.append(
                f"  {key:<{width}}{_yaml_scalar(value)}  REJECTED: selects the workload; "
                f"every command fails until it is removed "
                f"(`mlpstorage config unset {key}`)"
            )
    if report["ignored"]:
        lines.append("")
        for key, value in report["ignored"].items():
            lines.append(f"  {key:<{width}}{_yaml_scalar(value)}  ignored: no command has this key")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# the subcommands                                                              #
# --------------------------------------------------------------------------- #


def cmd_show(args, logger) -> int:
    report = config_report()
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))
    return EXIT_CODE.SUCCESS


def cmd_set(args, logger) -> int:
    key = args.key
    _check_key(key)
    stored = shape_value(key, list(args.values), action_for(key))
    path = write_user_config({key: stored})
    print(f"{key}: {_yaml_scalar(stored)}  (written to {path})")
    return EXIT_CODE.SUCCESS


def cmd_unset(args, logger) -> int:
    key = args.key
    path = user_config_path()
    if not os.path.isfile(path) or key not in read_user_config(path):
        print(f"{key}: not set in {path}")
        return EXIT_CODE.SUCCESS
    remove_user_config_key(key, path)
    print(f"{key}: removed from {path}")
    if key == RESULTS_DIR_KEY:
        print(f"Commands now need a results-dir: {HOW_TO_SUPPLY}.")
    return EXIT_CODE.SUCCESS


def cmd_path(args, logger) -> int:
    print(user_config_path())
    return EXIT_CODE.SUCCESS


_HANDLERS = {"show": cmd_show, "set": cmd_set, "unset": cmd_unset, "path": cmd_path}


def run_config_command(args, logger=None) -> int:
    """Entry point from ``main``: ``args.command`` picks the handler."""
    handler = _HANDLERS.get(getattr(args, "command", None))
    if handler is None:
        print(f"config: unknown subcommand {getattr(args, 'command', None)!r}")
        return EXIT_CODE.INVALID_ARGUMENTS
    return handler(args, logger)
