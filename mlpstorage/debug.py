import sys
import types
import traceback

from functools import partialmethod, wraps

from mlpstorage.config import MLPS_DEBUG


class DebugTryer:

    EXIT = 'exit'
    CONTINUE = 'continue'
    DEBUGGER = 'pdb'
    RAISE = 'raise'

    system_debug_state = MLPS_DEBUG

    def __init__(self, on_error=RAISE, on_keyboard=EXIT, debug=system_debug_state, logger=None, description=None, **kwargs):
        """
        This context manager is designed to replace the following pattern:

        .. code-block:: python

            try:
                instance = ClassName(param1=value1)
            except Exception as e:
                logger.error(e.msg)
                if debug:
                    import traceback
                    tb = traceback.format_exc()
                    print(tb)

                if exit_on_error:
                    import sys
                    sys.exit(1)
                else:
                    pass

        This code can be replaced with:

        .. code-block:: python

            from adputil.debug import DebugTryer

            debug = True
            on_error = DebugTryer.CONTINUE

            with DebugTryer(debug=debug, on_error=on_error, logger=logger)
                instance = ClassName(param1=value1)

        With the DebugTryer, you can flip the flag for debug at a top level to determine whether tracebacks
        are printed or surpressed. The on_error can be flipped to either exit on an error, go into debugger,
        or continue.

        This means there are TWO methods to get to a debugger on an error. Set on_error to DEBUGGER, set debug=True.
        The expected implementation is that --debug can be set by the CLI args and passed around like the logger
        parameters. on_error can be set in the script during development as a top of script constant. Change
        on_error=DebugTryer.DEBUGGER and use on_error=on_error in the DebugTryer instantiation call to get to the
        debugger.

        This should make it easier to suppress trace dumps for normal operation so the end user has a better experience
        while making it easier to see that information when debugging a problem.

        :param on_error:
        :param debug:
        :param logger:
        :param kwargs: Allow the user to pass unused kwargs
        """

        on_error_allowed_values = [type(self).EXIT, type(self).CONTINUE, type(self).DEBUGGER, type(self).RAISE]
        if on_error not in on_error_allowed_values:
            raise ValueError(f'on_error must be one of: {", ".join(on_error_allowed_values)}')

        self.on_error = on_error
        self.on_keyboard = on_keyboard
        self.debug = debug
        if logger:
            self.logger = logger
        else:
            from mlpstorage.logging import setup_logging
            self.logger = setup_logging('DebugLogging', stream_log_level='debug')

        self.description = description if description else ""

        self.verbose = kwargs.get('verbose')

    def __enter__(self):
        if self.verbose:
            self.logger.verbose(f'Doing: {self.description}')

    def __exit__(self, exc_type, exc_value, traceback_object):
        if exc_type is KeyboardInterrupt:
            if self.on_keyboard == type(self).RAISE:
                raise exc_value

            if self.on_keyboard == type(self).EXIT:
                sys.exit(1)

            if self.on_keyboard == type(self).CONTINUE:
                return True

            if self.on_keyboard == type(self).DEBUGGER:
                import pdb
                pdb.post_mortem()

        if exc_type is not None:
            if not self.description:
                if self.on_error == type(self).RAISE:
                    raise exc_value

            if self.description:
                self.logger.error(f'Exception caught during: "{self.description}"')

            # We have an exception, what do we do now?
            tb = traceback.format_exc()
            self.logger.error(exc_value)
            self.logger.error(str(tb))

            # When debug is true, we will print the trace for easier debugging. Otherwise it's suppressed
            if self.debug:
                self.logger.error(f'Traceback for the exception is: ')
                print(tb)

            # Currently 4 supported on_error actions. We can raise, exit, continue, or start the pdb debugger.
            # Continue is done by returning True. This is specific to exceptions in __exit__ of a context handler
            # Exit is obvious (we exit), Raise is obvious (we raise)
            # For the debugger we import pdb and start the post_mortem on the current exception (by passing no tb)
            if self.on_error == type(self).DEBUGGER or self.debug is True:
                import pdb
                pdb.post_mortem()

            if self.on_error == type(self).RAISE:
                raise exc_value

            if self.on_error == type(self).EXIT:
                sys.exit(1)

            if self.on_error == type(self).CONTINUE:
                return True




def debug_tryer_wrapper(on_error, debug, logger, func):
    @wraps(func)
    def debug_tryed(*args, **kwargs):
        with DebugTryer(on_error=on_error, debug=debug, logger=logger):
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
