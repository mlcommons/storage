"""
Phase 7 DOC-05 sync test — env-var inventory gate.

Purpose
-------
Assert that every environment variable mlpstorage reads in Python source
(``mlpstorage_py/**/*.py``) and the kv-cache shell wrapper is documented in
``ManPage.md``'s ``## ENVIRONMENT`` section, and vice-versa.  An automated
symmetric-difference check makes future doc-drift a CI failure.

Decisions enforced
------------------
  D-01  Three literal-arg grep patterns: ``os.environ.get(<literal>)``,
        ``os.environ[<literal>]``, ``os.getenv(<literal>)``.
  D-02  Constant-based reads caught via ``_ENV(VAR)?\\s*=\\s*['"]...['"]``
        module-level regex and an indirect-chain scanner for files that use
        ``os.environ.get(var)`` with a loop variable (storage_config.py).
  D-03  Shell-wrapper scan: ``os.getenv("KVCACHE_SELECTED_WORKLOADS")``
        literal-arg in ``kv-cache-wrapper.sh``.
  D-04  Docstring examples are structurally excluded because the D-01
        regexes don't anchor to ``>>> `` lines; the bracket regex catches
        ``os.environ['S3_ENDPOINT_URIS']`` in a docstring, but since that
        name IS documented, this is a false-positive in the safe direction.

Four test-class purposes (D-15/D-20/pattern-map):
  TestEnvVarInventorySync    — symmetric-difference + phantom-read sanity
  TestPerVarTierAssignment   — MANPAGE_ENV_VAR_TIERS vs ManPage tier headers
  TestVerbatimPinnedStrings  — D-20 anti-leaderboard and D-23 fallback strings
  TestMlperfRenameGate       — SC-1 / SC-6: no legacy MLPERF_* names in docs
"""

from __future__ import annotations

import re
import pathlib
import pytest

from mlpstorage_py.config import _MANPAGE_SYNC_ALLOWLIST, MANPAGE_ENV_VAR_TIERS


# ---------------------------------------------------------------------------
# Path constants (mirror test_no_import_cycles.py:38-42)
# ---------------------------------------------------------------------------

_TESTS_UNIT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_UNIT_DIR.parent.parent
_MANPAGE = _REPO_ROOT / "ManPage.md"
_MLPSTORAGE_PY = _REPO_ROOT / "mlpstorage_py"
_KVCACHE_WRAPPER = _REPO_ROOT / "kv_cache_benchmark" / "utils" / "kv-cache-wrapper.sh"


# ---------------------------------------------------------------------------
# D-20 verbatim pinned strings (MUST NOT be paraphrased — Phase 5/6 doctrine)
# ---------------------------------------------------------------------------

_D20_ANTI_LEADERBOARD_S1 = (
    "Aggregate columns are informational and NOT a leaderboard-input contract."
)
_D20_ANTI_LEADERBOARD_S2 = (
    "External ranking pipelines MUST compute their own aggregates from "
    "per-invocation summary.json files."
)

# D-23: multi-system-fallback sentence (verbatim from Phase 6 D-07 / Plan 07-03)
_D23_MULTI_SYSTEM_FALLBACK = (
    "When omitted, reportgen operates on the entire results-dir tree and "
    "derives systemname per row from the workload dir's path segment "
    "(open/<org>/results/<systemname>/... or checkpointing/training/vdb/kvcache analog)."
)


# ---------------------------------------------------------------------------
# Regex patterns (D-01 / D-02)
# ---------------------------------------------------------------------------

# D-01: three literal-arg read patterns
_ENV_GET_RE = re.compile(
    r"""os\.environ\.get\(\s*['"]([A-Z_][A-Z0-9_]*)['"]"""
)
_ENV_BRACKET_RE = re.compile(
    r"""os\.environ\[\s*['"]([A-Z_][A-Z0-9_]*)['"]\s*\]"""
)
_ENV_GETENV_RE = re.compile(
    r"""os\.getenv\(\s*['"]([A-Z_][A-Z0-9_]*)['"]"""
)

# D-02: module-level _ENV / _ENVVAR constant declarations
_ENV_CONST_RE = re.compile(
    r"""_ENV(?:VAR)?\s*=\s*['"]([A-Z_][A-Z0-9_]*)['"]""",
    re.MULTILINE,
)

# D-02 (indirect): detect files using os.environ.get(var) with a loop variable;
#   if found, also scan for list-item string literals that are env-var names.
_ENV_INDIRECT_RE = re.compile(r"""os\.environ\.get\(var\b""")
_ENV_CHAIN_ITEM_RE = re.compile(
    r"""^\s+['"]([A-Z][A-Z0-9_]{2,})['"]\s*,?\s*$""",
    re.MULTILINE,
)

# D-03: narrow shell-wrapper scan (os.getenv literal inside .sh embedded Python)
_KVCACHE_SHELL_RE = re.compile(
    r"""os\.getenv\(\s*['"]([A-Z_][A-Z0-9_]*)['"]"""
)


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------

def _scan_python_reads(root: pathlib.Path) -> set[str]:
    """Walk ``root/**/*.py`` and collect every env-var name read literally.

    Applies D-01 (three literal-arg patterns) and D-02 (module-level constant
    declarations + indirect-chain scanner) per the Phase 7 DOC-05 contract.

    Files containing ``# sync-test-scan-skip`` on their first line are skipped
    (escape hatch; no current files use this).

    The indirect-chain scanner (D-02 extension) activates when a file uses the
    ``os.environ.get(var)`` loop pattern (like ``storage_config._resolve_endpoint``).
    In that case, every string literal that looks like an env-var name appearing
    as a list element is also included.  This catches ``S3_ENDPOINT_TEMPLATE``,
    ``S3_ENDPOINT_FILE``, and ``S3_ENDPOINT`` which are read via an iterated chain
    rather than a direct literal-arg call.

    Raises:
        FileNotFoundError: if ``root`` does not exist (propagated, never swallowed).
    """
    reads: set[str] = set()
    for py_file in root.rglob("*.py"):
        text = py_file.read_text(errors="replace")
        # Check for skip marker on first line
        first_line = text.split("\n", 1)[0]
        if "# sync-test-scan-skip" in first_line:
            continue

        # D-01 literal-arg patterns
        for m in _ENV_GET_RE.finditer(text):
            reads.add(m.group(1))
        for m in _ENV_BRACKET_RE.finditer(text):
            reads.add(m.group(1))
        for m in _ENV_GETENV_RE.finditer(text):
            reads.add(m.group(1))

        # D-02 module-level constant declarations
        for m in _ENV_CONST_RE.finditer(text):
            reads.add(m.group(1))

        # D-02 indirect-chain scanner
        if _ENV_INDIRECT_RE.search(text):
            for m in _ENV_CHAIN_ITEM_RE.finditer(text):
                candidate = m.group(1)
                # Only admit uppercase-only names (no mixed-case false positives)
                if candidate == candidate.upper():
                    reads.add(candidate)

    return reads


def _scan_shell_reads(wrapper_path: pathlib.Path) -> set[str]:
    """Scan ``wrapper_path`` for ``os.getenv(<literal>)`` env-var reads (D-03).

    The kv-cache-wrapper.sh embeds a Python script that calls
    ``os.getenv("KVCACHE_SELECTED_WORKLOADS")``.  This function applies the same
    ``_KVCACHE_SHELL_RE`` pattern as D-03 describes.

    Raises:
        FileNotFoundError: if the wrapper does not exist.  Per the loud-failure
        doctrine, NEVER caught here — if the wrapper moves, the test fails loudly.
    """
    text = wrapper_path.read_text(errors="replace")
    reads: set[str] = set()
    for m in _KVCACHE_SHELL_RE.finditer(text):
        reads.add(m.group(1))
    return reads


def _parse_manpage_env_vars(manpage_path: pathlib.Path) -> dict[str, str]:
    """Parse ``ManPage.md``'s ``## ENVIRONMENT`` section and return a mapping.

    Returns:
        Dict mapping each env-var name (backtick-fenced in the first column of
        a tier table) to its tier-header text (lowercased, with ``-borrowed``
        preserved).  Examples::

            'MLPSTORAGE_RESULTS_DIR' -> 'owned'
            'OMPI_COMM_WORLD_RANK'   -> 'mpi-borrowed'
            'DLIO_DROP_CACHES_TIMEOUT' -> 'internal-write'

    Only table-row env-var-column entries are parsed; cross-ref bullet lines
    under ``Internal-write`` are **not** added to this dict because they
    reference vars whose primary tier is in another section (D-13).

    Raises:
        ValueError: if ``## ENVIRONMENT`` section is absent.
    """
    text = manpage_path.read_text()

    # Extract the ENVIRONMENT section body
    env_split = text.split("## ENVIRONMENT", 1)
    if len(env_split) < 2:
        raise ValueError(
            "ManPage.md is missing '## ENVIRONMENT' section — "
            "Phase 7 Plan C (07-03) may not have run yet."
        )
    env_body = env_split[1]

    # Trim at the next top-level ## heading
    tail_split = re.split(r"\n## ", env_body, maxsplit=1)
    env_body = tail_split[0]

    # Split body on ### sub-headers to get per-tier chunks
    # First chunk is the intro prose (before the first ###)
    tier_chunks = re.split(r"\n### ", env_body)

    tier_map: dict[str, str] = {}
    tier_canonical = {
        "owned": "owned",
        "mpi-borrowed": "mpi-borrowed",
        "aws-borrowed": "aws-borrowed",
        "storage-borrowed": "storage-borrowed",
        "diagnostic": "diagnostic",
        "internal-write": "internal-write",
    }

    for chunk in tier_chunks[1:]:  # skip intro prose
        lines = chunk.split("\n")
        tier_header_raw = lines[0].strip()
        tier_key = tier_header_raw.lower()

        if tier_key not in tier_canonical:
            continue
        canonical_tier = tier_canonical[tier_key]

        # Walk table rows: pattern is `| \`VAR_NAME\` | ... |`
        row_re = re.compile(r"^\|\s*`([A-Z_][A-Z0-9_]*)`\s*\|")
        for line in lines[1:]:
            m = row_re.match(line)
            if m:
                var_name = m.group(1)
                tier_map[var_name] = canonical_tier

    return tier_map


# ===========================================================================
# Test Class 1: Inventory sync (SC-3 / D-01 / D-02 / D-03 / D-04)
# ===========================================================================


class TestEnvVarInventorySync:
    """SC-3: code reads vs ManPage docs-mentions must be symmetric (modulo allowlist).

    Enforces D-01 / D-02 / D-03 / D-04 and provides negative-simulation
    methods that prove the grep pipeline fires on synthetic inputs.
    """

    def test_code_reads_equal_manpage_mentions_modulo_allowlist(self):
        """Symmetric-difference between code reads and ManPage mentions must be empty.

        Failure message shows BOTH the missing set (in code, not in ManPage) and
        the extra set (in ManPage, not in code) so a developer can fix the drift
        without re-instrumenting.
        """
        code_reads = _scan_python_reads(_MLPSTORAGE_PY) | _scan_shell_reads(_KVCACHE_WRAPPER)
        docs_mentions = set(_parse_manpage_env_vars(_MANPAGE).keys())

        missing_in_docs = code_reads - _MANPAGE_SYNC_ALLOWLIST - docs_mentions
        extra_in_docs = docs_mentions - _MANPAGE_SYNC_ALLOWLIST - code_reads

        assert missing_in_docs == set() and extra_in_docs == set(), (
            "Env-var inventory out of sync (SC-3 / DOC-05).\n\n"
            f"  In code but NOT in ManPage (missing from docs):\n"
            f"    {sorted(missing_in_docs)}\n\n"
            f"  In ManPage but NOT in code (extra in docs):\n"
            f"    {sorted(extra_in_docs)}\n\n"
            "To fix:\n"
            "  - For vars in 'missing from docs': add them to ManPage.md's "
            "ENVIRONMENT section under the appropriate tier AND add them to "
            "MANPAGE_ENV_VAR_TIERS in mlpstorage_py/config.py.\n"
            "  - For vars in 'extra in docs': either add the env-var read to "
            "the codebase, remove it from the ManPage tier table, or add it to "
            "_MANPAGE_SYNC_ALLOWLIST in config.py with a rationale comment."
        )

    def test_allowlist_names_are_never_documented(self):
        """Allowlist names (POSIX identity and legacy MLPERF_*) must not appear in ManPage tables.

        Verifies _MANPAGE_SYNC_ALLOWLIST's purpose: these names are exempt
        because they are NOT functional contracts (they are either POSIX identity
        plumbing or deprecated migration-detection sentinels).  If an allowlist
        name leaks into a ManPage tier table, this test fails.
        """
        docs_mentions = set(_parse_manpage_env_vars(_MANPAGE).keys())
        leaked = _MANPAGE_SYNC_ALLOWLIST & docs_mentions
        assert leaked == set(), (
            "Allowlist names found in ManPage tier tables — "
            "these names must not be documented (per D-05/D-10):\n"
            f"  {sorted(leaked)}\n\n"
            "Remove them from the ManPage ENVIRONMENT tier tables.  If they "
            "need to stay in ManPage prose (e.g. a 'formerly known as' note), "
            "that is fine — only table rows are checked here."
        )

    def test_scan_detects_phantom_read(self):
        """Negative simulation: the three D-01 regexes catch synthetic env-var reads.

        Constructs a synthetic text snippet with each of the three literal-arg
        patterns and asserts each regex matches the phantom name 'PHANTOM_ENV_VAR_XYZ'.
        Proves the grep pipeline fires on real code; a future refactor that
        accidentally breaks the regex (e.g., adds mandatory triple-quotes) will
        fail here LOUDLY rather than silently passing on an empty match set.
        """
        synthetic = (
            "value1 = os.environ.get('PHANTOM_ENV_VAR_XYZ', 'default')\n"
            "value2 = os.environ['PHANTOM_ENV_VAR_XYZ']\n"
            "value3 = os.getenv('PHANTOM_ENV_VAR_XYZ')\n"
        )
        found: set[str] = set()
        for m in _ENV_GET_RE.finditer(synthetic):
            found.add(m.group(1))
        for m in _ENV_BRACKET_RE.finditer(synthetic):
            found.add(m.group(1))
        for m in _ENV_GETENV_RE.finditer(synthetic):
            found.add(m.group(1))

        assert "PHANTOM_ENV_VAR_XYZ" in found, (
            "D-01 regex pipeline did not detect 'PHANTOM_ENV_VAR_XYZ' in the "
            "synthetic test string — one or more of the three regexes is broken.\n"
            f"  Matched names: {sorted(found)}"
        )

    def test_scan_detects_phantom_constant_declaration(self):
        """Negative simulation: D-02 _ENV_CONST_RE catches synthetic constant declarations.

        Constructs a synthetic text snippet with the module-level constant
        pattern and asserts _ENV_CONST_RE matches 'PHANTOM_CONST_NAME'.
        """
        synthetic = "PHANTOM_ENV = 'PHANTOM_CONST_NAME'\n"
        found: set[str] = set()
        for m in _ENV_CONST_RE.finditer(synthetic):
            found.add(m.group(1))

        assert "PHANTOM_CONST_NAME" in found, (
            "D-02 _ENV_CONST_RE did not detect 'PHANTOM_CONST_NAME' in the "
            "synthetic test string — the constant-declaration regex is broken.\n"
            f"  Matched names: {sorted(found)}"
        )


# ===========================================================================
# Test Class 2: Per-var tier assignment (SC-4 / D-12 / D-13 / D-14)
# ===========================================================================


class TestPerVarTierAssignment:
    """SC-4: every env var's declared primary tier matches its ManPage sub-header.

    Enforces MANPAGE_ENV_VAR_TIERS consistency with the ManPage table layout,
    pins specific SC-4 vars (OMPI_COMM_WORLD_RANK / PMI_RANK as mpi-borrowed;
    MLPSTORAGE_CHECKPOINT_URI_SCHEME as owned), and validates tier-value spelling.
    """

    def test_every_manpage_env_var_matches_declared_tier(self):
        """Each key in MANPAGE_ENV_VAR_TIERS must appear under the matching ManPage sub-header.

        Parses the ManPage tier tables and compares each var's discovered tier
        against the declared tier in MANPAGE_ENV_VAR_TIERS.  Failure message
        identifies the mismatched var with expected and actual values.
        """
        manpage_tiers = _parse_manpage_env_vars(_MANPAGE)
        mismatches = []
        for var_name, declared_tier in MANPAGE_ENV_VAR_TIERS.items():
            actual_tier = manpage_tiers.get(var_name)
            if actual_tier is None:
                mismatches.append(
                    f"  {var_name!r}: declared={declared_tier!r}, "
                    f"actual=NOT FOUND in ManPage tables"
                )
            elif actual_tier != declared_tier:
                mismatches.append(
                    f"  {var_name!r}: declared={declared_tier!r}, actual={actual_tier!r}"
                )

        assert not mismatches, (
            "MANPAGE_ENV_VAR_TIERS tier assignments disagree with ManPage sub-headers "
            "(SC-4 / D-12 / D-13):\n\n"
            + "\n".join(mismatches)
            + "\n\n"
            "Fix by updating either MANPAGE_ENV_VAR_TIERS in mlpstorage_py/config.py "
            "or the ManPage tier table (or both)."
        )

    def test_every_tier_value_is_one_of_six_canonical_strings(self):
        """MANPAGE_ENV_VAR_TIERS values must be one of the six canonical tier strings (D-07).

        The six valid values are:
            'owned', 'mpi-borrowed', 'aws-borrowed', 'storage-borrowed',
            'diagnostic', 'internal-write'

        Any spelling variation (extra whitespace, wrong capitalisation, unknown tier)
        fails this test.
        """
        valid_tiers = {
            "owned",
            "mpi-borrowed",
            "aws-borrowed",
            "storage-borrowed",
            "diagnostic",
            "internal-write",
        }
        bad_values = {
            v for v in MANPAGE_ENV_VAR_TIERS.values() if v not in valid_tiers
        }
        assert not bad_values, (
            f"MANPAGE_ENV_VAR_TIERS contains invalid tier value(s): {sorted(bad_values)}\n"
            f"Valid tier values: {sorted(valid_tiers)}\n"
            "Fix by correcting the tier string in mlpstorage_py/config.py."
        )

    def test_sc4_specific_tier_pins(self):
        """SC-4 spot-checks: specific vars must have specific tier assignments.

        Pins the three vars explicitly called out in SC-4 and D-12:
          - MLPSTORAGE_CHECKPOINT_URI_SCHEME: 'owned' (per D-12; dual-role via
            internal-write sub-tag in ManPage prose only, NOT a second tier value)
          - OMPI_COMM_WORLD_RANK: 'mpi-borrowed' (SC-4 OpenMPI rank injection)
          - PMI_RANK: 'mpi-borrowed' (SC-4 PMI/MPICH fallback rank injection)
        """
        assert MANPAGE_ENV_VAR_TIERS.get("MLPSTORAGE_CHECKPOINT_URI_SCHEME") == "owned", (
            "MANPAGE_ENV_VAR_TIERS['MLPSTORAGE_CHECKPOINT_URI_SCHEME'] must be 'owned' "
            "(SC-4 / D-12).  The internal-write sub-tag is ManPage prose only — "
            "not a second MANPAGE_ENV_VAR_TIERS entry."
        )
        assert MANPAGE_ENV_VAR_TIERS.get("OMPI_COMM_WORLD_RANK") == "mpi-borrowed", (
            "MANPAGE_ENV_VAR_TIERS['OMPI_COMM_WORLD_RANK'] must be 'mpi-borrowed' (SC-4)."
        )
        assert MANPAGE_ENV_VAR_TIERS.get("PMI_RANK") == "mpi-borrowed", (
            "MANPAGE_ENV_VAR_TIERS['PMI_RANK'] must be 'mpi-borrowed' (SC-4)."
        )


# ===========================================================================
# Test Class 3: Verbatim pinned strings (D-20 / D-23)
# ===========================================================================


class TestVerbatimPinnedStrings:
    """D-20 / D-23: specific prose sentences must be present verbatim in ManPage.md.

    Continues the Phase 5 D-02 / Phase 6 D-24 verbatim-pinning doctrine:
    user-pinned wording changes are CI failures, not merge conflicts.
    """

    def test_d20_anti_leaderboard_sentence_1_present(self):
        """ManPage.md must contain the D-20 first anti-leaderboard sentence verbatim."""
        text = _MANPAGE.read_text()
        assert _D20_ANTI_LEADERBOARD_S1 in text, (
            f"D-20 first anti-leaderboard sentence missing from ManPage.md.\n"
            f"Expected substring:\n  {_D20_ANTI_LEADERBOARD_S1!r}\n\n"
            "Fix: add this exact sentence (no paraphrase) to the "
            "'### Aggregate Interpretation' subsection under '## RESULTS DIRECTORY'."
        )

    def test_d20_anti_leaderboard_sentence_2_present(self):
        """ManPage.md must contain the D-20 second anti-leaderboard sentence verbatim."""
        text = _MANPAGE.read_text()
        assert _D20_ANTI_LEADERBOARD_S2 in text, (
            f"D-20 second anti-leaderboard sentence missing from ManPage.md.\n"
            f"Expected substring:\n  {_D20_ANTI_LEADERBOARD_S2!r}\n\n"
            "Fix: add this exact sentence (no paraphrase) to the "
            "'### Aggregate Interpretation' subsection under '## RESULTS DIRECTORY'."
        )

    def test_d23_multi_system_fallback_present(self):
        """ManPage.md must contain the D-23 multi-system-fallback sentence verbatim.

        Pins the reportgen ``--systemname`` optional-fallback behaviour documented
        in Phase 6 D-07 and the 07-03 ManPage rewrite.
        """
        text = _MANPAGE.read_text()
        assert _D23_MULTI_SYSTEM_FALLBACK in text, (
            f"D-23 multi-system-fallback sentence missing from ManPage.md.\n"
            f"Expected substring:\n  {_D23_MULTI_SYSTEM_FALLBACK!r}\n\n"
            "Fix: add this exact sentence (no paraphrase) to the reportgen "
            "'--systemname' subsection under '### Reports' (approximately line 852)."
        )


# ===========================================================================
# Test Class 4: MLPERF_* rename gate (SC-1 / SC-6)
# ===========================================================================


class TestMlperfRenameGate:
    """SC-1 / SC-6: no legacy MLPERF_* env-var names in ManPage.md or docs/.

    SC-1 is the executable form of Phase 5's grep-visible ADR enforcement:
    ``grep -nE "MLPERF_(SYSTEMNAME|RESULTS_DIR|ORGNAME|DATA_DIR)" ManPage.md`` = 0 hits.
    SC-6 extends the gate to ``docs/``.

    Note: the string ``MLPERF_DATA_DIR`` IS in ``mlpstorage_py/`` source code
    (at ``cli/training_args.py``) for migration-hint detection — it is NOT
    expected to be zero across the codebase; only the ManPage and docs/ are
    tested here.
    """

    _LEGACY_PATTERN = re.compile(
        r"MLPERF_(SYSTEMNAME|RESULTS_DIR|ORGNAME|DATA_DIR)"
    )

    def test_no_legacy_env_var_names_in_manpage(self):
        """ManPage.md must contain zero hits for MLPERF_(SYSTEMNAME|RESULTS_DIR|ORGNAME|DATA_DIR).

        This is the SC-1 executable form.  A match means the Phase 7 rename
        pass (Plan 07-03 D-25) missed a line.  Failure message shows the
        offending line number and surrounding context.
        """
        text = _MANPAGE.read_text()
        lines = text.splitlines()
        offenders = []
        for lineno, line in enumerate(lines, start=1):
            if self._LEGACY_PATTERN.search(line):
                offenders.append(f"  L{lineno}: {line.rstrip()}")

        assert not offenders, (
            "ManPage.md contains legacy MLPERF_* env-var names (SC-1 violated).\n"
            "Offending lines:\n"
            + "\n".join(offenders)
            + "\n\nFix: rename to the corresponding MLPSTORAGE_* equivalent "
            "(see Phase 7 Plan 07-03 D-25 for the rename map)."
        )

    def test_no_legacy_env_var_names_in_docs_dir(self):
        """docs/*.md must contain zero hits for MLPERF_(SYSTEMNAME|RESULTS_DIR|ORGNAME|DATA_DIR).

        Extends SC-1 to the ``docs/`` directory (SC-6).  Skips cleanly if the
        directory does not yet exist (not-yet-created case, not a failure).
        The ``examples/`` directory is intentionally not scanned (it does not
        exist in this repo; SC-6 scope adjustment recorded in Plan 07-03 SUMMARY).
        """
        docs_dir = _REPO_ROOT / "docs"
        if not docs_dir.exists():
            pytest.skip(
                "docs/ directory does not exist — SC-6 gate is a no-op until "
                "docs/ is created."
            )

        offenders = []
        for md_file in docs_dir.rglob("*.md"):
            text = md_file.read_text(errors="replace")
            lines = text.splitlines()
            for lineno, line in enumerate(lines, start=1):
                if self._LEGACY_PATTERN.search(line):
                    rel = md_file.relative_to(_REPO_ROOT)
                    offenders.append(f"  {rel}:L{lineno}: {line.rstrip()}")

        assert not offenders, (
            "docs/*.md contains legacy MLPERF_* env-var names (SC-6 violated).\n"
            "Offending lines:\n"
            + "\n".join(offenders)
            + "\n\nFix: rename to MLPSTORAGE_* equivalents throughout docs/."
        )
