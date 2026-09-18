"""Console, file, and recap logging for mlpstorage.

Three sinks share one record stream:

* **Console** — one shared rich ``Console`` on stderr. The level tag is
  coloured (``✖ ERROR`` bold red, ``⚠ WARNING`` yellow, ``STATUS`` bold blue,
  ``RESULT`` green); the message body is never styled so long messages stay
  readable and ``grep`` still works. Colour follows ``--color``:
  ``auto`` (default) colours only when stderr is a TTY and rich's own
  ``NO_COLOR`` / ``TERM=dumb`` detection allows it, ``always`` forces ANSI,
  ``never`` emits plain text. Progress bars (``progress.py``) render on the
  same console so log lines and live spinners never garble each other.

* **Per-run files** — a DEBUG-level buffer collects every record from process
  start. When a run leaf is reserved, ``attach_run_log_files(<leaf>)`` flushes
  the buffer into ``<leaf>/mlpstorage.log`` (all levels, uncoloured, with
  module:line) and ``<leaf>/mlpstorage.errors.log`` (WARNING and above), then
  streams later records straight to both until ``detach()``. Both files
  travel with the run: removing the leaf removes its logs.

* **Recap** — WARNING+ records are collected for ``emit_recap()``, which
  ``main()`` calls on the way out. A clean invocation prints nothing; otherwise
  a bordered block lists every warning and error and names the errors log.

Every ``mlpstorage_py.*`` module logger (``logging.getLogger(__name__)``)
reaches the same three sinks through the ``mlpstorage_py`` package logger, so
no confirmation line is silently dropped.

Level numbers and the ``logger.status()`` / ``logger.verbose()`` / ... helper
methods predate this module's rewrite and are unchanged.
"""

from __future__ import annotations

import collections
import enum
import logging
import os
import time
from typing import Deque, Iterable, List, Optional, Tuple

from rich.console import Console
from rich.text import Text

# --------------------------------------------------------------------------- #
# Levels                                                                      #
# --------------------------------------------------------------------------- #

CRITICAL = logging.CRITICAL
FATAL = CRITICAL
ERROR = logging.ERROR
RESULT = 35
WARNING = logging.WARNING   # 30
WARN = WARNING
STATUS = 25
INFO = logging.INFO         # 20
VERBOSE = 19
VERBOSER = 18
VERBOSEST = 17
DEBUG = logging.DEBUG       # 10
RIDICULOUS = 7
LUDICROUS = 5
PLAID = 3
NOTSET = logging.NOTSET

DEFAULT_STREAM_LOG_LEVEL = logging.INFO

custom_levels = {
    'RESULT': RESULT,
    'STATUS': STATUS,
    'VERBOSE': VERBOSE,
    'VERBOSER': VERBOSER,
    'VERBOSEST': VERBOSEST,
    'RIDICULOUS': RIDICULOUS,
    'LUDICROUS': LUDICROUS,
    'PLAID': PLAID
}


def log_level_factory(level_name):
    level_num = custom_levels.get(level_name, logging.NOTSET)

    def log_func(self, message, *args, **kwargs):
        # stacklevel=2: attribute module/line to the caller, not this wrapper.
        kwargs.setdefault("stacklevel", 2)
        self._log(level_num, message, args, **kwargs)
    return log_func


for custom_name, custom_num in custom_levels.items():
    logging.addLevelName(custom_num, custom_name)
    setattr(logging.Logger, custom_name.lower(), log_level_factory(custom_name))


# --------------------------------------------------------------------------- #
# Legacy ANSI table — kept for callers that import it; the console path below #
# uses rich styles instead.                                                   #
# --------------------------------------------------------------------------- #

class COLORS(enum.Enum):
    grey = "\033[0;30m"
    red = "\033[0;31m"
    green = "\033[0;32m"
    yellow = "\033[0;33m"
    blue = "\033[0;34m"
    purple = "\033[0;35m"
    cyan = "\033[0;36m"
    white = "\033[0;37m"
    bred = "\033[1;31m"
    bgreen = "\033[1;32m"
    byellow = "\033[1;33m"
    bblue = "\033[1;34m"
    bpurple = "\033[1;35m"
    bipurple = "\033[1;95m"
    normal = "\033[0m"


level_to_color_map = {
    ERROR: COLORS.bred,
    CRITICAL: COLORS.bred,
    WARNING: COLORS.yellow,
    RESULT: COLORS.green,
    STATUS: COLORS.bblue,
    PLAID: COLORS.bipurple,
}


def get_level_color(level):
    return level_to_color_map.get(level, COLORS.normal).value


# --------------------------------------------------------------------------- #
# Level tags and styles                                                       #
# --------------------------------------------------------------------------- #

_LEVEL_GLYPHS = {
    CRITICAL: "✖ ",
    ERROR: "✖ ",
    WARNING: "⚠ ",
}

_LEVEL_STYLES = {
    CRITICAL: "bold red",
    ERROR: "bold red",
    WARNING: "yellow",
    RESULT: "green",
    STATUS: "bold blue",
    PLAID: "bold magenta",
}


def format_level_tag(levelno: int) -> str:
    """Return the level tag printed on the console, e.g. ``✖ ERROR``.

    Only ERROR/CRITICAL and WARNING carry a glyph, so lines stay classifiable
    on terminals without colour and by colour-blind readers.
    """
    return f"{_LEVEL_GLYPHS.get(levelno, '')}{logging.getLevelName(levelno)}"


def level_style(levelno: int) -> str:
    return _LEVEL_STYLES.get(levelno, "")


# --------------------------------------------------------------------------- #
# Shared console                                                              #
# --------------------------------------------------------------------------- #

COLOR_MODES = ("auto", "always", "never")

_UNSET = object()


class _State:
    """Module-level singletons shared by every managed logger."""

    def __init__(self) -> None:
        self.console: Optional[Console] = None
        self.console_file = None            # None → live sys.stderr
        self.color_mode: str = "auto"
        self.console_handler: Optional["ConsoleLogHandler"] = None
        self.buffer: Optional["BufferingRecordHandler"] = None
        self.recap: Optional["RecapHandler"] = None
        self.managed: set = set()
        self.files: Optional["RunLogFiles"] = None
        self.last_files: Optional["RunLogFiles"] = None


_state = _State()


def _build_console(color: str, file) -> Console:
    kwargs = dict(
        file=file,
        stderr=file is None,
        markup=False,
        emoji=False,
        highlight=False,
        soft_wrap=True,
    )
    if color == "always":
        kwargs.update(force_terminal=True, color_system="standard")
    elif color == "never":
        kwargs.update(color_system=None, force_terminal=False)
    return Console(**kwargs)


def configure_console(color: str = "auto", file=_UNSET) -> Console:
    """(Re)build the shared console.

    Args:
        color: one of ``auto`` / ``always`` / ``never``.
        file: stream to write to; ``None`` means live ``sys.stderr``. When
            omitted the previously configured stream is kept, so
            ``apply_logging_options`` can change colour without disturbing a
            test's capture buffer.
    """
    if color not in COLOR_MODES:
        raise ValueError(f"--color must be one of {', '.join(COLOR_MODES)}; got {color!r}")
    if file is not _UNSET:
        _state.console_file = file
    _state.color_mode = color
    _state.console = _build_console(color, _state.console_file)
    return _state.console


def get_console() -> Console:
    """The shared stderr console used by log output and progress bars."""
    if _state.console is None:
        configure_console(_state.color_mode, _state.console_file)
    return _state.console


# --------------------------------------------------------------------------- #
# Handlers                                                                    #
# --------------------------------------------------------------------------- #

def _timestamp(record: logging.LogRecord) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))


class ConsoleLogHandler(logging.Handler):
    """Writes ``<time>|<tag>: <message>`` through the shared rich console.

    Only the tag is styled. With ``show_location`` (``--debug``) the line reads
    ``<time>|<tag>:<module>:<line>: <message>``.
    """

    def __init__(self, level: int = DEFAULT_STREAM_LOG_LEVEL) -> None:
        super().__init__(level)
        self.show_location = False

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            if record.exc_info:
                message = f"{message}\n{logging.Formatter().formatException(record.exc_info)}"
            text = Text(f"{_timestamp(record)}|")
            text.append(format_level_tag(record.levelno), style=level_style(record.levelno))
            if self.show_location:
                text.append(f":{record.module}:{record.lineno}")
            text.append(f": {message}")
            get_console().print(text, soft_wrap=True, highlight=False, markup=False, emoji=False)
        except Exception:  # pragma: no cover — logging must never raise
            self.handleError(record)


class BufferingRecordHandler(logging.Handler):
    """Keeps every record (DEBUG and up) until a run leaf exists to write to."""

    def __init__(self, maxlen: int = 20000) -> None:
        super().__init__(logging.DEBUG)
        self.records: Deque[logging.LogRecord] = collections.deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def drain(self) -> List[logging.LogRecord]:
        out = list(self.records)
        self.records.clear()
        return out


class RecapHandler(logging.Handler):
    """Collects WARNING+ records for the end-of-invocation recap."""

    MAX_ENTRIES = 50

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.addFilter(_ProblemFilter())
        self.entries: List[Tuple[int, str]] = []
        self.overflow = 0

    def emit(self, record: logging.LogRecord) -> None:
        if len(self.entries) >= self.MAX_ENTRIES:
            self.overflow += 1
            return
        self.entries.append((record.levelno, record.getMessage()))

    def clear(self) -> None:
        self.entries.clear()
        self.overflow = 0


def is_problem(levelno: int) -> bool:
    """True for WARNING and above, excluding RESULT (35), which is a success
    line that merely sits numerically above WARNING (30)."""
    return levelno >= WARNING and levelno != RESULT


class _ProblemFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return is_problem(record.levelno)


_FILE_FORMATTER = logging.Formatter(
    "%(asctime)s|%(levelname)s:%(module)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _shared_handlers() -> Tuple[logging.Handler, ...]:
    if _state.console_handler is None:
        _state.console_handler = ConsoleLogHandler()
        _state.buffer = BufferingRecordHandler()
        _state.recap = RecapHandler()
    return (_state.console_handler, _state.buffer, _state.recap)


# --------------------------------------------------------------------------- #
# Managed loggers                                                             #
# --------------------------------------------------------------------------- #

PACKAGE_LOGGER_NAME = "mlpstorage_py"


def _has_managed_ancestor(name: str) -> bool:
    parts = name.split(".")
    return any(".".join(parts[:i]) in _state.managed for i in range(1, len(parts)))


def _manage(name: str) -> logging.Logger:
    """Attach the shared handlers to ``name`` unless an ancestor already
    carries them (records would otherwise print twice via propagation)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if name not in _state.managed and not _has_managed_ancestor(name):
        for handler in _shared_handlers():
            if handler not in logger.handlers:
                logger.addHandler(handler)
        if _state.files is not None:
            for handler in _state.files.handlers:
                if handler not in logger.handlers:
                    logger.addHandler(handler)
        _state.managed.add(name)
    return logger


def _managed_loggers() -> Iterable[logging.Logger]:
    return (logging.getLogger(n) for n in sorted(_state.managed))


def setup_logging(name=__name__, stream_log_level=DEFAULT_STREAM_LOG_LEVEL):
    """Return a logger wired to the shared console/file/recap sinks.

    The console threshold is taken from ``stream_log_level`` only when the
    shared console handler is first created; later calls (helper classes that
    build their own fallback logger) do not lower or raise it. Use
    ``apply_logging_options`` for the CLI-driven threshold.
    """
    if isinstance(stream_log_level, str):
        stream_log_level = logging.getLevelName(stream_log_level.upper())

    first = _state.console_handler is None
    _shared_handlers()
    if first and isinstance(stream_log_level, int):
        _state.console_handler.setLevel(stream_log_level)

    _manage(PACKAGE_LOGGER_NAME)
    return _manage(name)


def apply_logging_options(_logger, args):
    """Apply ``--color`` / ``--stream-log-level`` / ``--verbose`` / ``--debug``.

    Only the console threshold moves; the per-run log buffer stays at DEBUG so
    ``mlpstorage.log`` is complete regardless of what the terminal shows.
    """
    if args is None:
        return
    _shared_handlers()
    console_handler = _state.console_handler

    color = getattr(args, "color", None)
    if color:
        configure_console(color)

    stream_level = getattr(args, "stream_log_level", None)
    if stream_level:
        console_handler.setLevel(stream_level.upper() if isinstance(stream_level, str) else stream_level)

    if getattr(args, "verbose", False) and console_handler.level > VERBOSE:
        console_handler.setLevel(VERBOSE)

    if getattr(args, "debug", False):
        console_handler.show_location = True
        if console_handler.level > DEBUG:
            console_handler.setLevel(DEBUG)


# --------------------------------------------------------------------------- #
# Per-run log files                                                           #
# --------------------------------------------------------------------------- #

RUN_LOG_FILENAME = "mlpstorage.log"
RUN_ERRORS_LOG_FILENAME = "mlpstorage.errors.log"


class RunLogFiles:
    """The pair of log files living inside one run leaf.

    Construct via :func:`attach_run_log_files`. Usable as a context manager;
    ``detach()`` is idempotent.
    """

    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir
        self.log_path = os.path.join(run_dir, RUN_LOG_FILENAME)
        self.errors_path = os.path.join(run_dir, RUN_ERRORS_LOG_FILENAME)

        full = logging.FileHandler(self.log_path, encoding="utf-8")
        full.setLevel(logging.DEBUG)
        full.setFormatter(_FILE_FORMATTER)
        errors = logging.FileHandler(self.errors_path, encoding="utf-8")
        errors.setLevel(logging.WARNING)
        errors.addFilter(_ProblemFilter())
        errors.setFormatter(_FILE_FORMATTER)
        self.handlers: Tuple[logging.Handler, ...] = (full, errors)
        self._attached = False

    def _attach(self) -> None:
        _shared_handlers()
        for record in _state.buffer.drain():
            for handler in self.handlers:
                if record.levelno >= handler.level:
                    handler.handle(record)   # handle() applies the filters
        for logger in _managed_loggers():
            for handler in self.handlers:
                if handler not in logger.handlers:
                    logger.addHandler(handler)
        _state.files = self
        self._attached = True

    def detach(self) -> None:
        if not self._attached:
            return
        for logger in _managed_loggers():
            for handler in self.handlers:
                logger.removeHandler(handler)
        for handler in self.handlers:
            handler.close()
        # Records emitted while attached already reached the files; drop them
        # so a later attach (``--loops``) does not replay them.
        _state.buffer.drain()
        _state.files = None
        _state.last_files = self
        self._attached = False

    def __enter__(self) -> "RunLogFiles":
        return self

    def __exit__(self, *exc) -> None:
        self.detach()


def attach_run_log_files(run_dir: str) -> RunLogFiles:
    """Start writing ``mlpstorage.log`` / ``mlpstorage.errors.log`` in ``run_dir``.

    Everything logged since process start (or since the previous detach) is
    written first, so the files open with the CLI parsing, environment
    validation, and code-image lines that preceded leaf reservation.
    """
    if _state.files is not None:
        _state.files.detach()
    files = RunLogFiles(run_dir)
    files._attach()
    return files


# --------------------------------------------------------------------------- #
# Recap                                                                       #
# --------------------------------------------------------------------------- #

def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def emit_recap(console: Optional[Console] = None) -> int:
    """Print the warnings/errors collected during this invocation.

    Returns the number of WARNING+ records seen. Prints nothing when that is
    zero, so clean invocations and data-only commands stay quiet.
    """
    _shared_handlers()
    recap = _state.recap
    total = len(recap.entries) + recap.overflow
    if total == 0:
        return 0

    console = console or get_console()
    errors = sum(1 for lvl, _ in recap.entries if lvl >= ERROR)
    warnings = len(recap.entries) - errors
    title = f"mlpstorage finished with {_plural(errors, 'error')}, {_plural(warnings, 'warning')}"
    console.print(Text(f"── {title} ").append("─" * 20), soft_wrap=True, highlight=False, markup=False, emoji=False)
    files = _state.files or _state.last_files
    where = "in the errors log" if files is not None else "above"
    for levelno, message in recap.entries:
        line = Text("  ")
        line.append(format_level_tag(levelno), style=level_style(levelno))
        # First line only: the Details/Suggestion lines were already printed
        # in full where the error happened and live in the errors log.
        first, _, rest = message.partition("\n")
        line.append(f": {first}")
        if rest.strip():
            line.append(f"  (+{len(rest.splitlines())} more lines {where})")
        console.print(line, soft_wrap=True, highlight=False, markup=False, emoji=False)
    if recap.overflow:
        console.print(Text(f"  ... {recap.overflow} more, see the errors log"), soft_wrap=True, highlight=False, markup=False, emoji=False)
    if files is not None:
        console.print(Text(f"  errors log: {files.errors_path}"), soft_wrap=True, highlight=False, markup=False, emoji=False)
        console.print(Text(f"  full log:   {files.log_path}"), soft_wrap=True, highlight=False, markup=False, emoji=False)
    return total


# --------------------------------------------------------------------------- #
# Test support                                                                #
# --------------------------------------------------------------------------- #

def reset_for_tests() -> None:
    """Forget buffered/recap records, detach run files, restore defaults."""
    if _state.files is not None:
        _state.files.detach()
    _state.last_files = None
    if _state.buffer is not None:
        _state.buffer.drain()
    if _state.recap is not None:
        _state.recap.clear()
    if _state.console_handler is not None:
        _state.console_handler.setLevel(DEFAULT_STREAM_LOG_LEVEL)
        _state.console_handler.show_location = False
    _state.console_file = None
    _state.color_mode = "auto"
    _state.console = None
