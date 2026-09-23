"""``mlpstorage status`` and the per-result recap printed after every run.

The habitual command over an initialized results-dir ("git status" role,
design decision 3, 2026-09-23): one row per result (Rules.md 1.3), the RUNS
column ``n/m`` as the headline with ``m`` from the edition's
``runs_per_result``, a SUBMIT token, a one-line NOTE, then a footer that
names each system's paperwork once (so twelve rows on one system do not read
as twelve problems) and a "Next:" line. ``--runs`` expands each result into
its run rows. Everything printed comes from :mod:`mlpstorage_py.readiness`,
which scores the tree with the same checker ``mlpstorage validate`` uses.

Output goes to stdout, like ``runs list``; the logger (stderr) carries only
the results-dir line and any warning. The post-run recap is printed after
``benchmark.run()`` has returned and the leaf's metadata is written, so it
never interleaves with DLIO's output (BACKLOG B-02) and never changes the
run's exit code.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

from mlpstorage_py.config import EXIT_CODE
from mlpstorage_py.readiness import (
    SUBMIT_INVALID,
    SUBMIT_NA,
    SUBMIT_PAPERWORK,
    SUBMIT_READY,
    SUBMIT_SHORT,
    Paperwork,
    ResultReadiness,
    SubmissionReadiness,
    evaluate,
    leaf_key,
    result_for_leaf,
)
from mlpstorage_py.runs.ledger import BENCHMARK_ALIASES

SUBMIT_FILTERS = (SUBMIT_SHORT, SUBMIT_INVALID, SUBMIT_PAPERWORK, SUBMIT_READY)

_WHATIF = "whatif"
_DIVISION_ORDER = {"closed": 0, "open": 1, _WHATIF: 2}
_RESULT_HEADERS = ["SYSTEM", "BENCHMARK", "MODEL", "ACCEL", "RUNS", "SUBMIT", "NOTE"]
_RUN_HEADERS = ["ID", "STATUS", "STARTED", "COUNTED", "NOTE"]
_RUN_INDENT = "  "


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_results(sub: SubmissionReadiness, args) -> List[ResultReadiness]:
    """The results matching the ``runs list`` style filters on ``args``
    (``mode_filter``, ``benchmark``, ``model``, ``systemname``) plus
    ``submit``."""
    mode = getattr(args, "mode_filter", None)
    benchmark = getattr(args, "benchmark", None)
    benchmark = BENCHMARK_ALIASES.get(benchmark, benchmark) if benchmark else None
    model = getattr(args, "model", None)
    systemname = getattr(args, "systemname", None)
    submit = getattr(args, "submit", None)
    out = []
    for r in sub.results:
        if mode and r.division != mode:
            continue
        if benchmark and r.benchmark != benchmark:
            continue
        if model and r.model != model:
            continue
        if systemname and r.systemname != systemname:
            continue
        if submit and r.submit != submit:
            continue
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _short_paperwork(paper: Paperwork) -> str:
    """The system's paperwork as the NOTE column says it: what, not where."""
    items = []
    if paper.yaml_missing:
        items.append(f"{paper.systemname}.yaml missing")
    if paper.blank_fields:
        items.append(f"{paper.systemname}.yaml: {_plural(len(paper.blank_fields), 'blank field')}")
    if paper.pdf_missing:
        items.append(f"{paper.systemname}.pdf missing")
    for f in paper.other:
        items.append(f.short())
    return ", ".join(items)


def _note(result: ResultReadiness) -> str:
    """The NOTE cell: the evaluator's note, with the system's paperwork
    shortened to its what (the footer says where)."""
    if result.submit != SUBMIT_PAPERWORK:
        return result.note
    parts = []
    if result.paperwork:
        parts.append("code image: " + "; ".join(f.short() for f in result.paperwork))
    if result.system_paperwork is not None:
        parts.append(_short_paperwork(result.system_paperwork))
    return "; ".join(p for p in parts if p)


def _format_rows(headers: Sequence[str], rows: Sequence[Sequence[str]], indent: str = "") -> List[str]:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join("{:<%d}" % w for w in widths)
    return [indent + fmt.format(*headers).rstrip()] + [indent + fmt.format(*row).rstrip() for row in rows]


def _result_row(result: ResultReadiness) -> List[str]:
    return [result.systemname, result.benchmark, result.model, result.accelerator or "-",
            f"{result.have}/{result.required}", result.submit, _note(result)]


def _run_row(run) -> List[str]:
    return [str(run.id), run.status, run.started, "yes" if run.counted else "-", run.reason]


def _division_header(division: str, orgname: str, edition: str, results: Sequence[ResultReadiness]) -> str:
    n_runs = sum(len(r.runs) for r in results)
    counts = f"{_plural(len(results), 'result')}, {_plural(n_runs, 'run')}"
    if division == _WHATIF:
        return f"{division}/{orgname}   {counts} (never packaged)"
    return f"{division}/{orgname}   rules edition {edition}   {counts}"


def _section(division: str, sub: SubmissionReadiness, results: Sequence[ResultReadiness],
             show_runs: bool) -> List[str]:
    lines = [_division_header(division, sub.orgname, sub.edition, results), ""]
    result_rows = [_result_row(r) for r in results]
    table = _format_rows(_RESULT_HEADERS, result_rows)
    if not show_runs:
        return lines + table
    run_rows = [[_run_row(run) for run in r.runs] for r in results]
    flat = [row for rows in run_rows for row in rows]
    run_table = _format_rows(_RUN_HEADERS, flat, indent=_RUN_INDENT)
    run_header, run_lines = run_table[0], run_table[1:]
    lines.append(table[0])
    lines.append(run_header)
    cursor = 0
    for i, result_line in enumerate(table[1:]):
        lines.append(result_line)
        n = len(run_rows[i])
        lines.extend(run_lines[cursor:cursor + n])
        cursor += n
    return lines


def _paperwork_lines(results: Sequence[ResultReadiness]) -> List[str]:
    seen: Dict[Tuple[str, str], Paperwork] = {}
    for r in results:
        if r.division != _WHATIF and r.system_paperwork is not None:
            seen.setdefault(r.system_paperwork.key, r.system_paperwork)
    return [f"Paperwork for {name} ({division}): {'; '.join(paper.items())}"
            for (division, name), paper in sorted(seen.items())]


def _next_line(sub: SubmissionReadiness, results: Sequence[ResultReadiness]) -> str:
    tokens = {r.submit for r in results if r.division != _WHATIF}
    if tokens & {SUBMIT_SHORT, SUBMIT_INVALID}:
        return "Next: mlpstorage status --runs   (which runs count, which must go)"
    if sub.tree_problems:
        return f"Next: mlpstorage validate {sub.results_dir}   (the tree problems above, in full)"
    if SUBMIT_PAPERWORK in tokens:
        return "Next: fill in the paperwork above, then mlpstorage status"
    return f"Next: mlpstorage validate {sub.results_dir}"


def render(sub: SubmissionReadiness, results: Optional[Sequence[ResultReadiness]] = None, *,
           show_runs: bool = False) -> List[str]:
    """The ``status`` text for ``results`` (default: every result of
    ``sub``), as lines without trailing newlines."""
    if results is None:
        results = sub.results
    by_division: Dict[str, List[ResultReadiness]] = {}
    for r in results:
        by_division.setdefault(r.division, []).append(r)
    lines: List[str] = []
    for division in sorted(by_division, key=lambda d: _DIVISION_ORDER.get(d, 9)):
        if lines:
            lines.append("")
        lines.extend(_section(division, sub, by_division[division], show_runs))

    package = [r for r in results if r.division != _WHATIF]
    lines.append("")
    if package:
        ready = sum(1 for r in package if r.submit == SUBMIT_READY)
        lines.append(f"{ready} of {_plural(len(package), 'result')} ready.")
    lines.extend(_paperwork_lines(results))
    if sub.tree_problems:
        lines.append("Tree problems (not attributable to a run or a system; fix before submitting):")
        lines.extend(f"  {f.short()}" for f in sub.tree_problems)
    if sub.warnings:
        lines.append(f"{_plural(len(sub.warnings), 'warning')} from the checker do not block a "
                     f"submission; mlpstorage validate {sub.results_dir} lists them.")
    lines.append(_next_line(sub, results))
    return lines


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_status_command(args, results_dir: str, logger) -> int:
    """``mlpstorage status`` on an initialized ``results_dir``."""
    sub = evaluate(results_dir)
    results = select_results(sub, args)
    if getattr(args, "json", False):
        payload = sub.to_dict()
        payload["results"] = [r.to_dict() for r in results]
        print(json.dumps(payload, indent=2))
        return EXIT_CODE.SUCCESS
    if not sub.results:
        print(f"No runs in {results_dir}.")
        return EXIT_CODE.SUCCESS
    if not results:
        print(f"No results match the selection ({len(sub.results)} in {results_dir}).")
        return EXIT_CODE.SUCCESS
    for line in render(sub, results, show_runs=getattr(args, "runs", False)):
        print(line)
    return EXIT_CODE.SUCCESS


def print_post_run_status(results_dir: str, leaf_path: str, logger) -> None:
    """Print the ``status`` table for the result that the run at
    ``leaf_path`` belongs to. Never raises: a failure to score the tree is a
    warning, the run's own exit code stands."""
    try:
        leaf = leaf_key(results_dir, leaf_path)
        if leaf is None:
            return
        sub = evaluate(results_dir)
        result = result_for_leaf(sub, leaf)
        if result is None:
            return
        print("")
        for line in render(sub, [result]):
            print(line)
    except Exception as exc:  # noqa: BLE001 -- the run is done; this is a courtesy
        logger.warning(f"Could not evaluate submission readiness after this run: {exc} "
                       f"(`mlpstorage status` will retry).")
