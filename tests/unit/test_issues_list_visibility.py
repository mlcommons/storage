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

    def test_warning_is_badged_as_a_warning(self, formatter):
        """A warning must not render under a green ``[CLOSED]`` badge.

        The badge is the first thing read. ``[CLOSED]`` on "row built
        from 1 of 3 runs" reads as approval of the row, which is the
        opposite of the message.
        """
        issues = [
            Issue(
                PARAM_VALIDATION.CLOSED,
                "[WARN] kv_cache row built from 1 of 3 runs",
                severity="warning",
            ),
        ]

        rendered = formatter.format_issues_list(issues, show_all=False)

        assert "[WARN]" in rendered
        assert "[CLOSED]" not in rendered, (
            f"warning rendered under a CLOSED badge: {rendered!r}"
        )

    def test_warn_badge_absorbs_the_messages_own_prefix(self, formatter):
        """No ``[WARN] [WARN]``.

        Warning messages carry their own ``[WARN] `` lead-in — pinned for
        the datagen leaf-presence family by
        ``test_aggregation.py::TestDatagenReportgenValidation`` — so the
        badge must absorb it rather than stack on it.
        """
        issues = [
            Issue(
                PARAM_VALIDATION.CLOSED,
                "[WARN] datagen leaf incomplete (20260101_000000): summary.json",
                severity="warning",
            ),
        ]

        rendered = formatter.format_issues_list(issues, show_all=False)

        assert rendered.count("[WARN]") == 1, f"stacked badges in {rendered!r}"
        assert "datagen leaf incomplete" in rendered

    def test_issue_without_parameter_omits_the_none_placeholder(self, formatter):
        """``parameter`` is optional; absent should not print as ``None:``."""
        issues = [
            Issue(PARAM_VALIDATION.INVALID, "metric x is empty in invocation y"),
        ]

        rendered = formatter.format_issues_list(issues, show_all=False)

        assert "None" not in rendered, f"stray None placeholder in {rendered!r}"
        assert "metric x is empty in invocation y" in rendered

    def test_issue_with_parameter_still_names_it(self, formatter):
        """Negative control — a real ``parameter`` is still rendered."""
        issues = [
            Issue(PARAM_VALIDATION.INVALID, "bad value", parameter="num_processes"),
        ]

        rendered = formatter.format_issues_list(issues, show_all=False)

        assert "num_processes: bad value" in rendered

    def test_show_all_is_unchanged(self, formatter):
        """``show_all=True`` still renders everything, warnings included."""
        issues = [
            Issue(PARAM_VALIDATION.CLOSED, "All runs satisfy the CLOSED category"),
            Issue(PARAM_VALIDATION.CLOSED, "[WARN] something", severity="warning"),
        ]

        rendered = formatter.format_issues_list(issues, show_all=True)

        assert "All runs satisfy" in rendered
        assert "[WARN] something" in rendered
