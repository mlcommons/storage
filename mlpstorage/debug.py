import sys
import types
import traceback

from functools import partialmethod, wraps

from mlpstorage.config import MLPS_DEBUG


def debug_tryer_wrapper(on_error, debug, logger, func):
    @wraps(func)
    def debug_tryed(*args, **kwargs):
        with DebugTryer(on_error=on_error, debug=debug, logger=logger, description=func.__name__):
            return func(*args, **kwargs)

    return debug_tryed


def debugger_hook(type, value, tb):
    """
    This hook is enabled with:

    `sys.excepthook = debugger_hook`

    This will result in exceptions dropping the user into the pdb debugger

    https://stackoverflow.com/questions/242485/starting-python-debugger-automatically-on-error

    :param type:
    :param value:
    :param tb:
    :return:
    """
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import pdb
        import traceback
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print()
        # ...then start the debugger in post-mortem mode.
        # pdb.pm() # deprecated
        pdb.post_mortem(tb)  # more "modern"
