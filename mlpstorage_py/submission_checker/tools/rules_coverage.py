"""Standalone CLI tool that reconciles Rules.md §2/§3/§4 against checker code.

Usage:
    python -m mlpstorage_py.submission_checker.tools.rules_coverage [--rules-md PATH]

Walks every kebab-case rule ID in ``Rules.md`` §2/§3/§4 (matched by the locked
regex ``^([234]\\.\\d+\\.\\d+)\\.\\s+\\*\\*([a-zA-Z][a-zA-Z0-9]+)\\*\\*``) and
reconciles each ID against four coverage sources in priority order (D-A4):

1. ``@rule``-decorated method on any ``BaseCheck`` subclass (via
   ``discover_rules`` introspection over the seven concrete check classes).
2. ``SystemYamlSchemaCheck.SCHEMA_ERROR_RULE_MAP`` values.
3. ``STUB_COVERAGE`` from ``coverage_mapping.py``.
4. ``OUT_OF_SCOPE_RULES`` from ``coverage_mapping.py``.

Output: Markdown table to stdout with columns
``| Rule ID | Rule Name | Disposition | Source |`` (D-A5). Column widths are
auto-sized via ``str.ljust`` over the actual parse rows so the header
separator row matches the data-row widths (Gemini-suggested upgrade #2).

Exit code: 0 if every live Rules.md ID has a disposition; 1 if any ID is
unmapped (success criterion #1). When unmapped, an error line per ID is
emitted to stderr with the locked wording.

Drift warnings (Gemini-suggested upgrade #1; CONTEXT.md ``<deferred>``):
``reconcile()`` additionally computes the set of IDs in ``OUT_OF_SCOPE_RULES``
and ``STUB_COVERAGE`` that are no longer present in Rules.md (purgatory
entries) and emits one ``log.warning`` per stale entry. Warnings are
**non-fatal** — exit code only changes for live unmapped IDs. The drift
signal helps maintainers prune registry entries when Rules.md evolves.

See ``compute_code_checksum.py`` for the sibling CLI-tool pattern that this
module mirrors.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(filename)s:%(lineno)d %(levelname)s] %(message)s",
)
log = logging.getLogger("rules_coverage")


# D-A3: locked regex for Rules.md §2/§3/§4 ID enumeration.
_RULE_ID_PATTERN = re.compile(r"^([234]\.\d+\.\d+)\.\s+\*\*([a-zA-Z][a-zA-Z0-9]+)\*\*")


def _default_rules_md_path() -> str:
    """Resolve the project-root ``Rules.md`` path from this module's location.

    Walks up three parents: ``tools/`` → ``submission_checker/`` →
    ``mlpstorage_py/`` → project root, then appends ``Rules.md``.
    """
    return str(Path(__file__).resolve().parents[3] / "Rules.md")


def _enumerate_rules_md(path: str) -> list:
    """Parse Rules.md and return a list of ``(rule_id, rule_name)`` tuples.

    Applies the locked regex to each line; preserves Rules.md's natural order
    (kebab-case IDs are sorted in spec order: 2.x.y → 3.x.y → 4.x.y).

    Args:
        path: Filesystem path to ``Rules.md``.

    Returns:
        List of ``(rule_id, rule_name)`` tuples. Empty list if the file is
        missing (an error is logged); callers propagate this through the
        exit code path (no rules → no work → exit 0 trivially, though
        ``main()`` treats missing-file as exit 1 separately).
    """
    rules_path = Path(path)
    if not rules_path.is_file():
        log.error("Rules.md not found at %s", path)
        return []
    result = []
    # Per WR-12 (review 2026-06-10): track fenced-code-block state so the
    # locked regex does not match against a line inside a ``` ... ```
    # block. The regex is strict enough that accidental matches are
    # unlikely in normal documentation prose, but a contributor pasting
    # an example block demonstrating *the format itself* would otherwise
    # register a phantom ID. Toggle on any line whose stripped leading
    # whitespace starts with ``` (the CommonMark fence marker).
    in_fence = False
    with rules_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            m = _RULE_ID_PATTERN.match(line)
            if m:
                result.append((m.group(1), m.group(2)))
    return result


def _collect_check_method_coverage() -> dict:
    """Discover every ``@rule``-decorated method on the seven check classes.

    Imports are scoped inside this helper so the CLI module stays importable
    when the broader submission_checker dependency graph (e.g., pydantic)
    is unavailable — a graceful-degradation hook for dev environments. Each
    import is wrapped individually so a missing optional dep on one class
    does not erase coverage signal from the others.

    Returns:
        Dict mapping ``rule_id`` → ``"ClassName.method_name"``. Last writer
        wins if two classes claim the same ID (impossible by D-A4 design;
        each ID is owned by exactly one check method).
    """
    from .. rule_registry import discover_rules  # noqa: E211 (style: relative)

    coverage = {}
    # Each check class is imported individually so a single broken import
    # does not blank the entire coverage signal. The reconcile path will
    # report any rule_id that ends up unmapped as a hard failure.
    check_classes = []

    try:
        from ..checks.submission_structure_checks import SubmissionStructureCheck
        check_classes.append(SubmissionStructureCheck)
    except ImportError as exc:
        log.warning("Could not import SubmissionStructureCheck: %s", exc)

    try:
        from ..checks.directory_checks import DirectoryCheck
        check_classes.append(DirectoryCheck)
    except ImportError as exc:
        log.warning("Could not import DirectoryCheck: %s", exc)

    try:
        from ..checks.training_checks import TrainingCheck
        check_classes.append(TrainingCheck)
    except ImportError as exc:
        log.warning("Could not import TrainingCheck: %s", exc)

    try:
        from ..checks.checkpointing_checks import CheckpointingCheck
        check_classes.append(CheckpointingCheck)
    except ImportError as exc:
        log.warning("Could not import CheckpointingCheck: %s", exc)

    try:
        from ..checks.system_yaml_schema_checks import SystemYamlSchemaCheck
        check_classes.append(SystemYamlSchemaCheck)
    except ImportError as exc:
        log.warning("Could not import SystemYamlSchemaCheck: %s", exc)

    try:
        from ..checks.vdb_checks import VdbCheck
        check_classes.append(VdbCheck)
    except ImportError as exc:
        log.warning("Could not import VdbCheck: %s", exc)

    try:
        from ..checks.kvcache_checks import KVCacheCheck
        check_classes.append(KVCacheCheck)
    except ImportError as exc:
        log.warning("Could not import KVCacheCheck: %s", exc)

    # Per WR-01 (review 2026-06-10): the D-A4 design says each rule_id is
    # owned by exactly one check method, but nothing enforced the invariant.
    # Surface duplicate bindings (almost certainly a copy-paste regression)
    # with a non-fatal warning. Last-writer still wins so the coverage
    # report renders, but the duplicate is no longer silent.
    for cls in check_classes:
        for rule_id, (_rule_name, method_name) in discover_rules(cls).items():
            source = "{}.{}".format(cls.__name__, method_name)
            if rule_id in coverage:
                log.warning(
                    "rule_id %s is decorated on both %s and %s — the second "
                    "binding wins in the coverage report but this is almost "
                    "certainly a copy-paste regression.",
                    rule_id, coverage[rule_id], source,
                )
            coverage[rule_id] = source
    return coverage


def _collect_schema_coverage() -> set:
    """Return the set of rule_ids carried in ``SCHEMA_ERROR_RULE_MAP`` values.

    Returns:
        Set of ``rule_id`` strings. Empty set if ``SystemYamlSchemaCheck``
        cannot be imported (graceful degradation for environments missing
        optional deps like pydantic).
    """
    try:
        from ..checks.system_yaml_schema_checks import SystemYamlSchemaCheck
    except ImportError as exc:
        log.warning(
            "Could not import SystemYamlSchemaCheck for SCHEMA_ERROR_RULE_MAP "
            "coverage: %s", exc,
        )
        return set()
    return {rid for (rid, _name) in SystemYamlSchemaCheck.SCHEMA_ERROR_RULE_MAP.values()}


def _collect_stub_coverage() -> dict:
    """Return a mapping ``rule_id → stub_class_name`` from STUB_COVERAGE.

    Re-imports ``coverage_mapping`` inside the function so monkeypatches in
    tests take effect (D-A4 priority 3).
    """
    from .. import coverage_mapping
    result = {}
    for stub_class_name, rule_ids in coverage_mapping.STUB_COVERAGE.items():
        for rule_id in rule_ids:
            result[rule_id] = stub_class_name
    return result


def _compute_drift(live_ids: set) -> tuple:
    """Compute the stale-registry entries (Gemini-suggested upgrade #1).

    Args:
        live_ids: The set of rule_ids currently present in Rules.md.

    Returns:
        ``(stale_oos, stale_stubs)`` where ``stale_oos`` is the set of
        rule_ids in ``OUT_OF_SCOPE_RULES`` no longer in Rules.md, and
        ``stale_stubs`` maps ``stub_class_name → [stale_rule_ids]`` (only
        stub classes with at least one stale ID are included).

    Note:
        ``coverage_mapping`` is re-imported here so test-time monkeypatches
        of either constant are observed (the test injects via
        ``monkeypatch.setattr("...coverage_mapping.STUB_COVERAGE", {...})``).
    """
    from .. import coverage_mapping

    stale_oos = set(coverage_mapping.OUT_OF_SCOPE_RULES.keys()) - live_ids
    stale_stubs = {}
    for stub_class_name, rule_ids in coverage_mapping.STUB_COVERAGE.items():
        stale = [rid for rid in rule_ids if rid not in live_ids]
        if stale:
            stale_stubs[stub_class_name] = stale
    return stale_oos, stale_stubs


def reconcile(rules_md_path=None) -> dict:
    """Reconcile Rules.md §2/§3/§4 IDs against the four coverage sources.

    Applies the locked priority order from CONTEXT.md D-A4:

    1. ``@rule``-decorated check method (``"check method"`` disposition)
    2. ``SCHEMA_ERROR_RULE_MAP`` values (``"schema check"`` disposition)
    3. ``STUB_COVERAGE`` (``"stub"`` disposition)
    4. ``OUT_OF_SCOPE_RULES`` (``"out of scope"`` disposition)
    5. Otherwise → unmapped (``"UNMAPPED"`` disposition; contributes to
       the returned ``unmapped`` set)

    Side-effects:
        Emits one ``log.warning`` per stale registry entry (per
        ``_compute_drift``). Warnings are **non-fatal**: stale entries
        do not contribute to ``unmapped`` and do not change exit code.
        Only IDs that are LIVE in Rules.md but have no mapping trigger
        the unmapped path.

    Args:
        rules_md_path: Optional override for the Rules.md location.
            Defaults to the project-root ``Rules.md`` resolved from
            this module's location.

    Returns:
        Dict with four keys:

        * ``rows``: list of ``(rule_id, rule_name, disposition, source)``
          tuples in Rules.md order.
        * ``unmapped``: set of rule_ids with no disposition (the
          gatekeeper signal that drives exit code 1).
        * ``stale_oos``: set of rule_ids in ``OUT_OF_SCOPE_RULES`` no
          longer in Rules.md (drift signal).
        * ``stale_stubs``: dict ``stub_class_name → [stale_rule_ids]``.
    """
    if rules_md_path is None:
        rules_md_path = _default_rules_md_path()

    parsed = _enumerate_rules_md(rules_md_path)
    live_ids = {rid for (rid, _name) in parsed}

    check_method_coverage = _collect_check_method_coverage()
    schema_coverage = _collect_schema_coverage()
    stub_coverage = _collect_stub_coverage()

    # Read OUT_OF_SCOPE_RULES live (per-call) so test monkeypatches apply.
    from .. import coverage_mapping
    out_of_scope = dict(coverage_mapping.OUT_OF_SCOPE_RULES)

    # Drift detection BEFORE building rows (warnings fire even when every
    # live ID is mapped; the unmapped set is unaffected by drift).
    stale_oos, stale_stubs = _compute_drift(live_ids)
    for rid in sorted(stale_oos):
        log.warning(
            "OUT_OF_SCOPE_RULES contains stale rule_id %s "
            "(no longer in Rules.md); remove it from coverage_mapping.py",
            rid,
        )
    for stub_class_name in sorted(stale_stubs):
        for rid in sorted(stale_stubs[stub_class_name]):
            log.warning(
                "STUB_COVERAGE contains stale rule_id %s on stub %s "
                "(no longer in Rules.md); remove it from coverage_mapping.py",
                rid,
                stub_class_name,
            )

    rows = []
    unmapped = set()
    for rule_id, rule_name in parsed:
        if rule_id in check_method_coverage:
            disposition = "check method"
            source = check_method_coverage[rule_id]
        elif rule_id in schema_coverage:
            disposition = "schema check"
            source = "SystemYamlSchemaCheck.SCHEMA_ERROR_RULE_MAP"
        elif rule_id in stub_coverage:
            disposition = "stub"
            source = "{} (stub)".format(stub_coverage[rule_id])
        elif rule_id in out_of_scope:
            disposition = "out of scope"
            source = "OUT_OF_SCOPE_RULES: {}".format(out_of_scope[rule_id])
        else:
            disposition = "UNMAPPED"
            source = ""
            unmapped.add(rule_id)
        rows.append((rule_id, rule_name, disposition, source))

    return {
        "rows": rows,
        "unmapped": unmapped,
        "stale_oos": stale_oos,
        "stale_stubs": stale_stubs,
    }


def _print_table(rows) -> None:
    """Print a Markdown table with auto-sized column widths.

    Per Gemini-suggested upgrade #2, the header, separator row, and data
    rows all use the same per-column widths computed from
    ``max(len(header), max(len(row[i]) for row in rows))``. This guarantees
    the separator dash count matches the data-row column width, which the
    Markdown renderer needs for proper alignment.

    Args:
        rows: List of ``(rule_id, rule_name, disposition, source)`` tuples.
            If empty, prints only the header + separator (no data rows).
    """
    headers = ("Rule ID", "Rule Name", "Disposition", "Source")
    # Width = max(header label, max-data-value). When rows is empty, fall
    # back to the header label length so the separator still renders.
    if rows:
        width_id = max(len(headers[0]), max(len(r[0]) for r in rows))
        width_name = max(len(headers[1]), max(len(r[1]) for r in rows))
        width_disp = max(len(headers[2]), max(len(r[2]) for r in rows))
        width_src = max(len(headers[3]), max(len(r[3]) for r in rows))
    else:
        width_id = len(headers[0])
        width_name = len(headers[1])
        width_disp = len(headers[2])
        width_src = len(headers[3])

    def _format_row(vals):
        return "| {} | {} | {} | {} |".format(
            vals[0].ljust(width_id),
            vals[1].ljust(width_name),
            vals[2].ljust(width_disp),
            vals[3].ljust(width_src),
        )

    # Separator row: per-column dash count equals (width + 2) to cover the
    # single leading/trailing space inside each cell. This makes the bar
    # positions identical to the header and data rows.
    separator = "|{}|{}|{}|{}|".format(
        "-" * (width_id + 2),
        "-" * (width_name + 2),
        "-" * (width_disp + 2),
        "-" * (width_src + 2),
    )

    print(_format_row(headers))
    print(separator)
    for row in rows:
        print(_format_row(row))


def get_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace with attribute ``rules_md`` (optional path
        override; ``None`` selects the project-root ``Rules.md``).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Reconcile every Rules.md §2/§3/§4 ID against @rule-decorated "
            "check methods, SCHEMA_ERROR_RULE_MAP, STUB_COVERAGE, and "
            "OUT_OF_SCOPE_RULES. Exits 1 if any ID is unmapped."
        ),
    )
    parser.add_argument(
        "--rules-md",
        default=None,
        help="Path to Rules.md (defaults to the project-root Rules.md).",
    )
    return parser.parse_args()


def run(args) -> int:
    """Reconcile Rules.md coverage against a parsed argument namespace.

    Public entry for in-process invocation (e.g. the top-level
    ``mlpstorage rules-coverage`` CLI shim, or tests). The standalone
    ``__main__`` entry below builds the namespace via ``get_args()``
    and delegates here.

    Args:
        args: ``argparse.Namespace`` with attribute ``rules_md``
            (optional path override; ``None`` selects the project-root
            ``Rules.md``).

    Returns:
        0 if every live Rules.md ID has a disposition (drift warnings
        for stale registry entries do not change exit code); 1 if any
        live ID is unmapped (success criterion #1 second clause); 2 if
        the Rules.md file does not exist (per WR-11, review 2026-06-10
        — a missing Rules.md means coverage cannot be verified, which
        must be a hard failure rather than a silent exit 0).
    """
    # WR-11: hard-fail when Rules.md is absent. The previous behavior
    # delegated to ``_enumerate_rules_md`` which logged an error and
    # returned an empty list — that produced an unmapped set of size 0
    # and exit 0, masking the missing-file failure mode (e.g. when the
    # CLI is invoked from an installed wheel where ``parents[3]`` does
    # not resolve to a Rules.md location).
    rules_md_path = args.rules_md or _default_rules_md_path()
    if not Path(rules_md_path).is_file():
        log.error(
            "Rules.md not found at %s; cannot verify coverage. Pass "
            "--rules-md <path> to override the default location.",
            rules_md_path,
        )
        return 2
    result = reconcile(rules_md_path=rules_md_path)
    _print_table(result["rows"])

    if result["unmapped"]:
        # Build an id→name lookup over the parse so the error message can
        # name the rule. The rows already carry both columns; build the
        # lookup directly from rows for the unmapped ids.
        name_by_id = {row[0]: row[1] for row in result["rows"]}
        for rule_id in sorted(result["unmapped"]):
            log.error(
                "Rule %s (%s) is in Rules.md but no mapping found. "
                "Either implement the check, register a stub, or add it "
                "to OUT_OF_SCOPE_RULES with a reason.",
                rule_id,
                name_by_id.get(rule_id, "?"),
            )
        return 1
    return 0


def main() -> int:
    """Standalone entry point: parse argv, then delegate to ``run``."""
    return run(get_args())


if __name__ == "__main__":
    sys.exit(main())
