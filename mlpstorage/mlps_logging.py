import collections
import datetime
import enum
import logging
import sys

# Define the custom log levels
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


# Custom colors for various logging levels
class COLORS(enum.Enum):
    grey = "\033[0;30m"
    red = "\033[0;31m"
    green = "\033[0;32m"
    yellow = "\033[0;33m"
    blue = "\033[0;34m"
    purple = "\033[0;35m"
    cyan = "\033[0;36m"
    white = "\033[0;37m"
    igrey = "\033[0;90m"
    ired = "\033[0;91m"
    igreen = "\033[0;92m"
    iyellow = "\033[0;93m"
    iblue = "\033[0;94m"
    ipurple = "\033[0;95m"
    icyan = "\033[0;96m"
    iwhite = "\033[0;97m"
    bgrey = "\033[1;30m"
    bred = "\033[1;31m"
    bgreen = "\033[1;32m"
    byellow = "\033[1;33m"
    bblue = "\033[1;34m"
    bpurple = "\033[1;35m"
    bcyan = "\033[1;36m"
    bwhite = "\033[1;37m"
    bigrey = "\033[1;90m"
    bired = "\033[1;91m"
    bigreen = "\033[1;92m"
    biyellow = "\033[1;93m"
    biblue = "\033[1;94m"
    bipurple = "\033[1;95m"
    bicyan = "\033[1;96m"
    biwhite = "\033[1;97m"
    normal = "\033[0m"


level_to_color_map = {
    ERROR: COLORS.bred,
    CRITICAL: COLORS.bred,
    WARNING: COLORS.yellow,
    RESULT: COLORS.green,
    STATUS: COLORS.bblue,
    INFO: COLORS.normal,
    VERBOSE: COLORS.normal,
    VERBOSER: COLORS.normal,
    VERBOSEST: COLORS.normal,
    DEBUG: COLORS.normal,
    RIDICULOUS: COLORS.normal,
    LUDICROUS: COLORS.normal,
    PLAID: COLORS.bipurple,
}


def get_level_color(level):
    return level_to_color_map.get(level, COLORS.normal).value


def log_level_factory(level_name):
    level_num = custom_levels.get(level_name, logging.NOTSET)

    def log_func(self, message, *args, **kwargs):
        self._log(level_num, message, args, **kwargs)
    return log_func


# Add the custom levels to the logger
for custom_name, custom_num in custom_levels.items():
    logging.addLevelName(custom_num, custom_name)
    setattr(logging.Logger, custom_name.lower(), log_level_factory(custom_name))


class ColoredStandardFormatter(logging.Formatter):
    def format(self, record):
        formatted_time = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        color = get_level_color(record.levelno)
        return f"{color}{formatted_time}|{record.levelname}: {record.getMessage()}{COLORS['normal'].value}"


class ColoredDebugFormatter(logging.Formatter):
    def format(self, record):
        formatted_time = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        color = get_level_color(record.levelno)
        return f"{color}{formatted_time}|{record.levelname}:{record.module}:{record.lineno}: " \
               f"{record.getMessage()}{COLORS['normal'].value}"


def setup_logging(name=__name__, stream_log_level=DEFAULT_STREAM_LOG_LEVEL):
    if isinstance(stream_log_level, str):
        stream_log_level = logging.getLevelName(stream_log_level.upper())

    _logger = logging.getLogger(name)
    _logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(ColoredStandardFormatter())
    stream_handler.setLevel(stream_log_level)  # Adjust this level as needed
    _logger.addHandler(stream_handler)

    return _logger


def apply_logging_options(_logger, args):
    if args is None:
        return
    # Set log level to VERBOSE unless the current log level is higher. In which case set it 1 level higher
    stream_handlers = [h for h in _logger.handlers if not hasattr(h, 'baseFilename')]
    log_levels = sorted([v for k, v in sys.modules[__name__].__dict__.items() if type(v) is int])

    if hasattr(args, "stream_log_level") and args.stream_log_level:
        for stream_handler in stream_handlers:
            stream_handler.setLevel(args.stream_log_level.upper())

    if hasattr(args, "verbose") and args.verbose:
        for stream_handler in stream_handlers:
            if stream_handler.level > VERBOSE:
                stream_handler.setLevel(VERBOSE)

    if hasattr(args, "debug") and args.debug:
        for stream_handler in stream_handlers:
            stream_handler.setFormatter(ColoredDebugFormatter())
            if stream_handler.level > DEBUG:
                stream_handler.setLevel(DEBUG)
