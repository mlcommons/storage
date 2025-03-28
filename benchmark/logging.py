import datetime
import enum
import logging

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


def get_level_color(level):
    color_enum = {
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
    }.get(level, COLORS.normal)
    return color_enum.value


for level_name, level_num in custom_levels.items():
    logging.addLevelName(level_num, level_name)


# Create a custom logger
logger = logging.getLogger('custom_logger')
DEFAULT_STREAM_LOG_LEVEL = logging.DEBUG
logger.setLevel(DEFAULT_STREAM_LOG_LEVEL)


def status(self, message, *args, **kwargs):
    self._log(STATUS, message, args, **kwargs)


def result(self, msg, *args, **kwargs):
    self._log(RESULT, msg, args, **kwargs)


# Define the custom log level methods
def verbose(self, message, *args, **kwargs):
    self._log(VERBOSE, message, args, **kwargs)


def verboser(self, message, *args, **kwargs):
    self._log(VERBOSER, message, args, **kwargs)


def verbosest(self, message, *args, **kwargs):
    self._log(VERBOSEST, message, args, **kwargs)


def ridiculous(self, msg, *args, **kwargs):
    self._log(RIDICULOUS, msg, args, **kwargs)


def ludicrous(self, msg, *args, **kwargs):
    self._log(LUDICROUS, msg, args, **kwargs)


def plaid(self, msg, *args, **kwargs):
    self._log(PLAID, msg, args, **kwargs)


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        formatted_time = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        color = get_level_color(record.levelno)
        return f"{color}{formatted_time}|{record.levelname}:{record.module}:{record.lineno}: " \
               f"{record.getMessage()}{COLORS['normal'].value}"


logging.Logger.result = result
logging.Logger.status = status
logging.Logger.verbose = verbose
logging.Logger.verboser = verboser
logging.Logger.verbosest = verbosest
logging.Logger.ridiculous = ridiculous
logging.Logger.ludicrous = ludicrous
logging.Logger.plaid = plaid


def setup_logging(name=__name__, stream_log_level=DEFAULT_STREAM_LOG_LEVEL):
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(ColoredFormatter())
    stream_handler.setLevel(stream_log_level)  # Adjust this level as needed
    _logger.addHandler(stream_handler)

    return _logger

