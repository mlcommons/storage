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

# Baseline recorded 2026-09-21 (Rules.md as of #874): 64 hits in 36 blocks.
# Sweep B2 §1+§2 (2026-09-21): 51 hits in 29 blocks.
# Sweep B2 §3 (2026-09-21): 41 hits in 22 blocks.
# Sweep B2 §4 (2026-09-21): 33 hits in 18 blocks.
ALLOWLIST: dict[str, set[str]] = {
    "5.4.1": {"intent_phrase"},
    "5.4.2": {"intent_phrase", "tool_behaviour"},
    "5.6. VDB OPEN versus CLOSED Options (prose)": {"code_constant", "code_reference"},
    "5.6.1": {"code_constant"},
    "5.6.3": {"code_constant"},
    "5.6.5": {"code_constant"},
    "6.3. KVCache Run Options (prose)": {"code_constant", "code_reference", "editorial", "tool_behaviour", "tracker_ref"},
    "6.3.1.1": {"code_constant", "code_reference"},
    "6.3.2.2": {"rationale_word"},
    "6.3.3.3": {"code_reference"},
    "6.3.3.4": {"tool_behaviour", "tracker_ref"},
    "6.3.4.1": {"code_reference"},
    "6.3.4.3": {"code_reference", "rationale_word", "tracker_ref"},
    "6.3.5. Example invocations (prose)": {"example"},
    "6.4.1": {"intent_phrase"},
    "6.4.2": {"code_constant", "code_reference"},
    "6.6.2": {"code_constant", "code_reference", "editorial"},
    "6.6.3": {"code_reference"},
}
