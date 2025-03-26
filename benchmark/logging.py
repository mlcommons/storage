import logging

# Define the custom log levels
STATUS = 25
VERBOSE = 13
VERBOSER = 12
VERBOSEST = 11

STREAM_LOG_LEVEL = logging.DEBUG

COLOR_MAP = {
    'normal': "\033[0m",
    'white': "\033[37m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    'intense_purple': '\033[1;95m',
}

# Create a custom logger
logger = logging.getLogger('custom_logger')
logger.setLevel(STREAM_LOG_LEVEL)

# Define the custom log levels
logging.addLevelName(VERBOSE, "VERBOSE")
logging.addLevelName(VERBOSER, "VERBOSER")
logging.addLevelName(VERBOSEST, "VERBOSEST")
logging.addLevelName(STATUS, "STATUS")


# Define the custom log level methods
def verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kwargs)


def verboser(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSER):
        self._log(VERBOSER, message, args, **kwargs)


def verbosest(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSEST):
        self._log(VERBOSEST, message, args, **kwargs)


def status(self, message, *args, **kwargs):
    if self.isEnabledFor(STATUS):
        self._log(STATUS, message, args, **kwargs)


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        color_map = {
            logging.DEBUG: COLOR_MAP['white'],
            logging.WARNING: COLOR_MAP['yellow'],
            logging.ERROR: COLOR_MAP['red'],
            logging.CRITICAL: COLOR_MAP['red'],
            STATUS: COLOR_MAP['intense_purple'],
        }
        formatted_time = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        color = color_map.get(record.levelno, COLOR_MAP['normal'])
        return f"{color}{formatted_time}|{record.levelname}:{record.module}:{record.lineno}: " \
               f"{record.getMessage()}{COLOR_MAP['normal']}"


logging.Logger.verbose = verbose
logging.Logger.verboser = verboser
logging.Logger.verbosest = verbosest
logging.Logger.status = status


def setup_logging(name=__name__, stream_log_level=STREAM_LOG_LEVEL):
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(ColoredFormatter())
    stream_handler.setLevel(stream_log_level)  # Adjust this level as needed
    _logger.addHandler(stream_handler)

    return _logger

