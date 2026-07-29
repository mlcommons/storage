"""Warning issues must survive the ``show_all=False`` filter (#836).

``ReportGenerator.print_results`` renders each workload's issues via
``ValidationMessageFormatter.format_issues_list(..., show_all=False)``,
whose filter drops every ``Issue`` whose ``validation`` is ``CLOSED``.

That conflates two orthogonal fields. ``validation`` says which
submission category a run qualifies for; ``severity`` says whether a
human needs to look. A finding that does not disqualify a CLOSED
submission but does mean its published row is misleading is exactly a
``CLOSED`` + ``severity="warning"`` issue — and the filter made every
one of them invisible.

Two such families exist:

- the datagen leaf-presence warnings (``report_generator.py``, issue
  #717 era), written to be "worth surfacing but not invalidating", which
  have never appeared in the printed summary;
- the #836 kv_cache collapsed-group and partial_failure warnings, whose
  whole purpose is to stop a row that represents one of three runs, or a
  run that lost 97 % of its ranks, from reading as clean.

A ``WARNING`` line in a reportgen log that carries 183 other warnings is
not a surface a reviewer of a 179-row table can be held to.
"""

from __future__ import annotations

import pytest

from mlpstorage_py.config import PARAM_VALIDATION
from mlpstorage_py.reporting.formatters import ValidationMessageFormatter
from mlpstorage_py.rules.issues import Issue


@pytest.fixture
def formatter():
    """Colorless formatter — assertions are on text, not escape codes."""
    return ValidationMessageFormatter(use_colors=False)


class TestIssuesListVisibility:
    """``show_all=False`` hides noise, not warnings."""

    def test_closed_warning_is_shown(self, formatter):
        """A CLOSED issue marked ``warning`` reaches the printed summary."""
        issues = [
            Issue(
                PARAM_VALIDATION.CLOSED,
                "[WARN] kv_cache row built from 1 of 3 runs in this workload group",
                severity="warning",
            ),
        ]

        rendered = formatter.format_issues_list(issues, show_all=False)

        assert "1 of 3 runs" in rendered, (
            f"the warning was filtered out of the summary: {rendered!r}"
        )
        assert "No actionable issues" not in rendered

    def test_closed_informational_issue_stays_hidden(self, formatter):
        """The filter still suppresses ordinary CLOSED chatter.

        Negative control — without it the fix would just be "show
        everything", which floods the summary with the per-run CLOSED
        qualification messages the verifier emits for every workload.
        """
        issues = [
            Issue(
                PARAM_VALIDATION.CLOSED,
                "All runs satisfy the CLOSED category",
            ),
        ]

        rendered = formatter.format_issues_list(issues, show_all=False)

        assert rendered.strip() == "No actionable issues"

    def test_mixed_list_shows_only_the_actionable(self, formatter):
        """Warnings and non-CLOSED issues show; CLOSED informational does not."""
        issues = [
            Issue(PARAM_VALIDATION.CLOSED, "All runs satisfy the CLOSED category"),
            Issue(
                PARAM_VALIDATION.CLOSED,
                "[WARN] kv_cache run 20260709_215001 recorded partial_failure",
                severity="warning",
            ),
            Issue(PARAM_VALIDATION.INVALID, "metric x is empty in invocation y"),
        ]

        rendered = formatter.format_issues_list(issues, show_all=False)

        assert "partial_failure" in rendered
        assert "metric x is empty" in rendered
        assert "All runs satisfy" not in rendered

    def test_show_all_is_unchanged(self, formatter):
        """``show_all=True`` still renders everything, warnings included."""
        issues = [
            Issue(PARAM_VALIDATION.CLOSED, "All runs satisfy the CLOSED category"),
            Issue(PARAM_VALIDATION.CLOSED, "[WARN] something", severity="warning"),
        ]

        rendered = formatter.format_issues_list(issues, show_all=True)

        assert "All runs satisfy" in rendered
        assert "[WARN] something" in rendered
