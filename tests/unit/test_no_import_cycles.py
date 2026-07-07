"""
Import-direction invariant tests (Phase 5 D-11 / D-18).

After Phase 5, ``mlpstorage_py.config`` owns all ``MLPSTORAGE_*_ENVVAR``
string constants (D-10) and ``mlpstorage_py.rules.utils`` imports them from
``config`` — never the other direction. This one-way dependency prevents
circular imports and lets ``config`` stay a leaf module that any layer can
import without dragging the rules subsystem in.

This file guards two invariants:

  (a) A fresh Python interpreter can execute
      ``import mlpstorage_py.config; import mlpstorage_py.rules.utils``
      with exit code 0 — proving no cycle exists at import time. A fresh
      subprocess is essential; an in-process check lies because both
      modules are cached in ``sys.modules`` by the time pytest runs.

  (b) Structural grep — ``mlpstorage_py/config.py`` contains no
      ``from mlpstorage_py.rules`` or ``import mlpstorage_py.rules`` line.
      This catches accidental reintroduction of the reverse dependency in
      code review by making the direction violation loudly visible as a
      test failure rather than a runtime surprise.

If either invariant breaks in a future PR, one of these tests fails with a
message that names the invariant, so the reviewer sees the direction
violation immediately.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys


# Path to the project root — walk up from this test file to the repo root,
# then to ``mlpstorage_py/config.py``. This test file lives at
# ``tests/unit/test_no_import_cycles.py``, so root is parent x 2.
_TESTS_UNIT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_UNIT_DIR.parent.parent
_CONFIG_PY = _REPO_ROOT / "mlpstorage_py" / "config.py"
_RULES_UTILS_PY = _REPO_ROOT / "mlpstorage_py" / "rules" / "utils.py"


class TestNoImportCycles:
    """Guard the config → rules.utils one-way dependency direction (D-11)."""

    def test_config_then_rules_utils_imports_cleanly_in_fresh_interpreter(self):
        """A fresh subprocess must be able to import config THEN rules.utils.

        In-process checks are unreliable because both modules are already
        cached in ``sys.modules`` by the time pytest runs — a cycle would
        not resurface. Spawning a fresh interpreter forces the import
        machinery to walk the full dependency graph from scratch.
        """
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import mlpstorage_py.config; import mlpstorage_py.rules.utils",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Include stderr in the failure message so the diagnosis is
        # immediate — the reviewer sees the ImportError / stack trace
        # without needing to re-run manually.
        assert result.returncode == 0, (
            "Fresh-interpreter import of config then rules.utils failed "
            "(D-11 violated).\n"
            f"stdout: {result.stdout!r}\n"
            f"stderr: {result.stderr!r}"
        )
        # Belt-and-suspenders: even if the subprocess somehow returned 0
        # with an ImportError logged, catch that too.
        assert "ImportError" not in result.stderr, (
            f"ImportError surfaced in stderr: {result.stderr!r}"
        )
        assert "circular" not in result.stderr.lower(), (
            f"'circular' appeared in stderr: {result.stderr!r}"
        )

    def test_config_does_not_import_from_rules(self):
        """Structural check: ``config.py`` must not import from ``rules.*``.

        Grep-style assertion over the file text. Catches accidental
        reintroduction of the reverse dependency in code review by making
        the direction violation loudly visible as a test failure. This
        assertion is what makes D-11 durable across future refactors —
        the runtime import check above catches the symptom; this catches
        the cause.
        """
        text = _CONFIG_PY.read_text()

        # Both syntactic forms are prohibited.
        assert "from mlpstorage_py.rules" not in text, (
            "mlpstorage_py/config.py contains 'from mlpstorage_py.rules' — "
            "one-way dependency direction violated (D-11). config MUST NOT "
            "import from rules; rules imports from config."
        )
        assert "import mlpstorage_py.rules" not in text, (
            "mlpstorage_py/config.py contains 'import mlpstorage_py.rules' — "
            "one-way dependency direction violated (D-11). config MUST NOT "
            "import from rules; rules imports from config."
        )

    def test_rules_utils_imports_env_var_constants_from_config(self):
        """Positive-direction assertion: rules.utils sources env-var names from config.

        Documents the intended direction (rules → config) and pins the
        specific constants moved in Plan 05-01 per D-10. If a future
        refactor tries to re-inline the env-var-name string literals in
        ``rules/utils.py``, this test surfaces the duplication.
        """
        text = _RULES_UTILS_PY.read_text()

        assert "from mlpstorage_py.config import" in text, (
            "mlpstorage_py/rules/utils.py must import from "
            "mlpstorage_py.config (positive D-11 direction)."
        )
        # The two constants moved in Plan 05-01 per D-10.
        assert "MLPSTORAGE_ORGNAME_ENVVAR" in text, (
            "rules/utils.py does not reference MLPSTORAGE_ORGNAME_ENVVAR — "
            "the D-10 single-source-of-truth import may have regressed."
        )
        assert "MLPSTORAGE_SYSTEMNAME_ENVVAR" in text, (
            "rules/utils.py does not reference MLPSTORAGE_SYSTEMNAME_ENVVAR — "
            "the D-10 single-source-of-truth import may have regressed."
        )
