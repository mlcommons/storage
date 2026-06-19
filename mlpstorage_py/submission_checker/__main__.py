"""Package entry point so ``python -m mlpstorage_py.submission_checker`` works.

The Definition-of-Done end-to-end test (CONTEXT.md D-E3, ROADMAP success
criteria #3 / #4) locks the subprocess invocation to
``python -m mlpstorage_py.submission_checker``, so the package needs a
``__main__`` module that delegates to ``main.main()``.
"""

import sys

from .main import main


if __name__ == "__main__":
    sys.exit(main())
