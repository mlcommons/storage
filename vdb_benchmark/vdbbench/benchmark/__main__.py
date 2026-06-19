"""Allow running the benchmark as ``python -m vdbbench.benchmark``."""

import sys

from .run_benchmark import main

sys.exit(main())
