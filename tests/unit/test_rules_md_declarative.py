"""
Rules.md declarative lint: rule text is predicates on the package, nothing else.

Rules.md §1 "Scope of this document" says every numbered rule is a declarative
statement decidable from the submission package alone.  Rationale and
derivation belong in RulesCommentary.md (keyed by rule number), run-time tool
behaviour in ManPage.md, editorial notes and PR/issue references in the
tracker.  This module turns that scope note from advice into a gate, the same
way test_help_all_parity.py gates the --help_all reference.

How it works
------------
``scan_rules_md()`` walks Rules.md outside ``` fences and splits it into
*blocks*.  A block is one numbered rule (key = its id, e.g. "3.3.8") from the
rule line to the next rule line or heading; prose before a section's first
rule is keyed "<heading text> (prose)".  Tables and continuation lines inherit
the block they follow.  Each block is matched against MARKERS, a name -> regex
map of words and shapes that do not occur in declarative rule text
("because", "we don't want", "currently", PR/issue numbers, Python symbols,
"mlpstorage warns", "Eg:", ...).

Three invariants:

  1. every (block, marker) hit is listed in rules_md_declarative_allowlist.py
  2. every allowlist entry still hits -- the allowlist only shrinks, so a
     stale entry means a sweep forgot to remove it
  3. every "(Commentary: `RulesCommentary.md` §X.)" pointer in Rules.md
     resolves to a "## X " heading in RulesCommentary.md, and every such
     heading names a live Rules.md rule id

When invariant 1 fails on a sentence that *is* declarative, tune the marker
here; never add an allowlist entry for a genuinely declarative sentence.  When
it fails on text that is not declarative, move the text (RulesCommentary.md /
ManPage.md / tracker) rather than allowlisting it.  Sweep PRs are measured by
how much the allowlist shrinks.

Running this file directly prints the current hits as an allowlist literal:

    uv run python tests/unit/test_rules_md_declarative.py

One paragraph is exempt by construction: the §1 "**Scope of this document.**"
paragraph names the excluded classes ("rationale", "issues or pull
requests") in order to exclude them.  The scanner skips that single
paragraph, and ``test_scope_paragraph_still_present`` keeps the skip honest.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RULES_MD_PATH = PROJECT_ROOT / "Rules.md"
COMMENTARY_PATH = PROJECT_ROOT / "RulesCommentary.md"

if str(PROJECT_ROOT) not in sys.path:  # for the __main__ dumper
    sys.path.insert(0, str(PROJECT_ROOT))
from tests.unit.rules_md_declarative_allowlist import ALLOWLIST  # noqa: E402


# =====================================================================
# Block parser
# =====================================================================

# Wider than rules_coverage's ``[23456]`` pattern on purpose: 1.1 / 1.2 /
# 2.1.18a / 2.1.5.a / 2.29 are rule text too, whatever the coverage tool
# thinks of their numbering.
RULE_LINE_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)+(?:\.?[a-z])?)\.\s+\*\*([A-Za-z][A-Za-z0-9_]*)\*\*"
)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")

# The one paragraph that names the excluded classes in order to exclude them.
SCOPE_PARAGRAPH_PREFIX = "**Scope of this document.**"
# Navigation only; it mirrors the headings and would double-count them.
SKIPPED_HEADINGS = {"Table of Contents"}

COMMENTARY_POINTER_RE = re.compile(
    r"\(Commentary:\s*`RulesCommentary\.md`\s*§(\d+(?:\.\d+)+(?:\.?[a-z])?)\.?\)"
)
COMMENTARY_HEADING_RE = re.compile(
    r"^##\s+(\d+(?:\.\d+)+(?:\.?[a-z])?)\s+(\S+)", re.MULTILINE
)


class Block:
    __slots__ = ("key", "start_line", "lines", "is_heading")

    def __init__(self, key: str, start_line: int, is_heading: bool = False):
        self.key = key
        self.start_line = start_line
        self.lines: list[tuple[int, str]] = []
        self.is_heading = is_heading


def parse_blocks(text: str) -> list[Block]:
    """Split Rules.md into rule / prose blocks, ignoring fenced code."""
    blocks: list[Block] = []
    current: Block | None = None
    in_fence = False
    skipping_section = False
    skipping_paragraph = False

    def open_block(key: str, lineno: int, first_line: str | None, is_heading=False):
        nonlocal current
        current = Block(key, lineno, is_heading)
        blocks.append(current)
        if first_line is not None:
            current.lines.append((lineno, first_line))

    for lineno, line in enumerate(text.splitlines(), start=1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        heading = HEADING_RE.match(line)
        if heading:
            title = heading.group(2)
            skipping_section = title in SKIPPED_HEADINGS
            skipping_paragraph = False
            if skipping_section:
                current = None
                continue
            # The heading line itself is scanned (an "Example ..." heading is
            # a hit), then prose up to the first rule shares its key.
            open_block(f"{title} (prose)", lineno, line, is_heading=True)
            continue
        if skipping_section:
            continue

        stripped = line.strip()
        if skipping_paragraph:
            if stripped == "":
                skipping_paragraph = False
            continue
        if stripped.startswith(SCOPE_PARAGRAPH_PREFIX):
            skipping_paragraph = True
            continue

        rule = RULE_LINE_RE.match(line)
        if rule:
            open_block(rule.group(1), lineno, line)
            continue

        if current is None:
            open_block("(preamble)", lineno, line)
        else:
            current.lines.append((lineno, line))

    return blocks


# =====================================================================
# Markers: shapes that do not occur in declarative rule text
# =====================================================================

MARKERS: dict[str, re.Pattern[str]] = {
    # why the rule is what it is
    "rationale_word": re.compile(
        r"\b(because|rationale|aside)\b|kept as-is", re.IGNORECASE
    ),
    "intent_phrase": re.compile(
        r"\b(we don't want|we need|so that|in order to|to ensure)\b", re.IGNORECASE
    ),
    # editorial chatter
    "editorial": re.compile(
        r"\((?:NB|nb):|maybe remove|not clear|\*\*_\(|\(somehow\)"
        r"|if we can help it|\*\*Caveat:\*\*|Enforcement status"
    ),
    # provenance: PR / issue / plan numbers
    "tracker_ref": re.compile(
        r"(?:\bPR|\bissue|\bIssue|\bPlan|\bPhase)\s*#?\d+|#\d{3,}\b|post-PR"
    ),
    # Python symbols and source files
    "code_reference": re.compile(
        r"`_[a-z_]+`|`\w+\(\)`|`[\w/]+\.py`|\b\w+\.sh\b|`np\.\w+`|\bfmean\b|\bargparse\b"
    ),
    # SHOUTING_CASE identifiers from the code base (REFERENCE_CHECKSUMS,
    # WORKLOAD_PARAMS).  Single-word tokens such as `DISKANN` are package
    # vocabulary (they name directories), hence the required underscore.
    "code_constant": re.compile(r"`[A-Z][A-Z0-9]*_[A-Z0-9_]+`"),
    # what the tool does at run time
    "tool_behaviour": re.compile(
        r"`?mlpstorage`?\s+(command|tool)\s+(must|should|will|applies|warns|probes)"
        r"|`?mlpstorage`?\s+(warns|probes|fails fast)"
        r"|\bcurrently\b"
        r"|today these are enforced",
        re.IGNORECASE,
    ),
    # examples and tutorials
    "example": re.compile(r"\bEg:|Example invocations"),
}
HEADING_MARKERS: dict[str, re.Pattern[str]] = {
    "example": re.compile(r"\bexample", re.IGNORECASE),
}
# Matched text that is a value the package can carry, not a code identifier:
# Milvus index-type names that appear as `index.index_type` values.
PACKAGE_VOCABULARY: dict[str, frozenset[str]] = {
    "code_constant": frozenset({"`IVF_FLAT`", "`IVF_SQ8`"}),
}


class Hit:
    __slots__ = ("key", "marker", "lineno", "excerpt")

    def __init__(self, key, marker, lineno, excerpt):
        self.key, self.marker, self.lineno, self.excerpt = key, marker, lineno, excerpt

    def __repr__(self):
        return f"{self.key} [{self.marker}] line {self.lineno}: {self.excerpt}"


def _excerpt(line: str, match: re.Match, width: int = 50) -> str:
    lo = max(0, match.start() - width)
    hi = min(len(line), match.end() + width)
    return ("…" if lo else "") + line[lo:hi].strip() + ("…" if hi < len(line) else "")


def scan_blocks(blocks: list[Block]) -> list[Hit]:
    hits: list[Hit] = []
    for block in blocks:
        for lineno, line in block.lines:
            markers = MARKERS
            if block.is_heading and lineno == block.start_line:
                markers = {**MARKERS, **HEADING_MARKERS}
            for name, pattern in markers.items():
                vocabulary = PACKAGE_VOCABULARY.get(name, frozenset())
                for m in pattern.finditer(line):
                    if m.group(0) in vocabulary:
                        continue
                    hits.append(Hit(block.key, name, lineno, _excerpt(line, m)))
                    break
    return hits


def scan_rules_md(path: Path = RULES_MD_PATH) -> list[Hit]:
    return scan_blocks(parse_blocks(path.read_text(encoding="utf-8")))


def hits_as_allowlist(hits: list[Hit]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for h in hits:
        out.setdefault(h.key, set()).add(h.marker)
    return out


def format_allowlist(allow: dict[str, set[str]]) -> str:
    lines = ["ALLOWLIST = {"]
    for key in sorted(allow, key=_sort_key):
        markers = ", ".join(f'"{m}"' for m in sorted(allow[key]))
        lines.append(f'    "{key}": {{{markers}}},')
    lines.append("}")
    return "\n".join(lines)


def _sort_key(key: str):
    m = re.match(r"(\d+(?:\.\d+)*)", key)
    if not m:
        return ((), key)
    return (tuple(int(p) for p in m.group(1).split(".")), key)


# =====================================================================
# Tests
# =====================================================================

GUIDANCE = (
    "Rules.md rule text must be declarative (see Rules.md §1 'Scope of this "
    "document').  Move rationale / derivation to RulesCommentary.md under the "
    "same rule number, run-time tool behaviour to ManPage.md, editorial notes "
    "and PR/issue references to the tracker.  If the flagged sentence really "
    "is a predicate on the package, tune the marker in "
    "tests/unit/test_rules_md_declarative.py -- never grow the allowlist."
)


@pytest.fixture(scope="module")
def blocks() -> list[Block]:
    return parse_blocks(RULES_MD_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def hits(blocks) -> list[Hit]:
    return scan_blocks(blocks)


class TestParser:
    def test_every_numbered_rule_is_its_own_block(self, blocks):
        keys = [b.key for b in blocks if not b.is_heading and b.key != "(preamble)"]
        assert len(keys) == len(set(keys)), "duplicate rule ids in Rules.md"
        for expected in ("1.1", "2.1.5.a", "2.1.18a", "2.29", "3.3.8", "6.3.4.5"):
            assert expected in keys, f"rule {expected} not parsed as a block"

    def test_fenced_code_is_not_scanned(self, blocks):
        # 6.3.5's fenced shell examples contain "# CLOSED, 4 clients ..." lines
        # that would otherwise read as headings.
        for b in blocks:
            assert not b.key.startswith("CLOSED, "), b.key
            assert "one-time, per results directory" not in b.key

    def test_scope_paragraph_still_present(self):
        text = RULES_MD_PATH.read_text(encoding="utf-8")
        assert SCOPE_PARAGRAPH_PREFIX in text, (
            "The §1 scope paragraph is the only text the scanner exempts; if "
            "it has been reworded, update SCOPE_PARAGRAPH_PREFIX."
        )


class TestDeclarative:
    def test_no_unlisted_hits(self, hits):
        unlisted = [h for h in hits if h.marker not in ALLOWLIST.get(h.key, set())]
        if unlisted:
            listing = "\n".join(f"  {h!r}" for h in unlisted)
            pytest.fail(
                f"{len(unlisted)} non-declarative marker hit(s) in Rules.md not "
                f"covered by the allowlist:\n{listing}\n\n{GUIDANCE}"
            )

    def test_allowlist_entries_still_hit(self, hits):
        live = hits_as_allowlist(hits)
        stale = [
            (key, marker)
            for key, markers in ALLOWLIST.items()
            for marker in markers
            if marker not in live.get(key, set())
        ]
        assert not stale, (
            "Stale allowlist entries (the text no longer hits): "
            f"{stale}.  Remove them from tests/unit/rules_md_declarative_allowlist.py "
            "-- the allowlist only shrinks."
        )

    def test_allowlist_markers_are_known(self):
        known = set(MARKERS) | set(HEADING_MARKERS)
        unknown = {
            (key, m) for key, ms in ALLOWLIST.items() for m in ms if m not in known
        }
        assert not unknown, f"allowlist names markers that do not exist: {unknown}"


class TestCommentaryCrossReferences:
    def test_every_commentary_pointer_has_a_heading(self):
        rules = RULES_MD_PATH.read_text(encoding="utf-8")
        commentary = COMMENTARY_PATH.read_text(encoding="utf-8")
        headings = {m.group(1) for m in COMMENTARY_HEADING_RE.finditer(commentary)}
        pointers = set(COMMENTARY_POINTER_RE.findall(rules))
        assert pointers, "expected at least one Commentary pointer in Rules.md"
        missing = pointers - headings
        assert not missing, (
            f"Rules.md points at RulesCommentary.md §{sorted(missing)} but no "
            "'## <id> <ruleName>' heading exists there"
        )

    def test_every_commentary_heading_is_a_live_rule(self, blocks):
        commentary = COMMENTARY_PATH.read_text(encoding="utf-8")
        rule_ids = {b.key for b in blocks if not b.is_heading}
        rule_names = {}
        for b in blocks:
            if not b.is_heading and b.lines:
                m = RULE_LINE_RE.match(b.lines[0][1])
                if m:
                    rule_names[m.group(1)] = m.group(2)
        problems = []
        for m in COMMENTARY_HEADING_RE.finditer(commentary):
            rid, name = m.group(1), m.group(2)
            if rid not in rule_ids:
                problems.append(f"§{rid} is not a Rules.md rule")
            elif rule_names.get(rid) != name:
                problems.append(
                    f"§{rid} is named {name} in RulesCommentary.md but "
                    f"{rule_names.get(rid)} in Rules.md"
                )
        assert not problems, problems


if __name__ == "__main__":
    all_hits = scan_rules_md()
    for h in all_hits:
        print(repr(h))
    print()
    print(f"# {len(all_hits)} hits in {len(hits_as_allowlist(all_hits))} blocks")
    print(format_allowlist(hits_as_allowlist(all_hits)))
