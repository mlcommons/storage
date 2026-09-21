"""
Allowlist for tests/unit/test_rules_md_declarative.py.

Maps a Rules.md block key (a rule id such as "3.3.8", or "<heading> (prose)"
for text before a section's first rule) to the set of marker names that are
tolerated there *for now*.  Every entry is pre-existing non-declarative text
awaiting migration to RulesCommentary.md / ManPage.md / the tracker.

This list only shrinks.  Do not add entries: if a new sentence trips a
marker, either move the text or (if the sentence is genuinely declarative)
tune the marker in the test module.  A stale entry -- one whose text no
longer hits -- fails test_allowlist_entries_still_hit and must be removed.

Regenerate the current-hits literal with:

    uv run python tests/unit/test_rules_md_declarative.py
"""

ALLOWLIST: dict[str, set[str]] = {}
