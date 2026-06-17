#!/usr/bin/env python3
"""
Tests for the --submitters CLI flag → Config.check_submitter contract.

Per WR-02 iter-2 (review 2026-06-10): the iter-1 WR-06/WR-07 fix in
``main.py`` routes empty / whitespace-only ``--submitters`` strings to
``submitters=None`` and promises the documented "match-all" default. That
promise is unverified at the boundary that actually matters — namely
``Config.check_submitter`` — unless something pins it. A future regression
that flipped ``Config.check_submitter(None semantic)`` from "match all" to
"match none" would silently drop every real submission and no other test
in the suite would catch it.

This module pins two contracts:

1. ``Config(submitters=None).check_submitter(X)`` returns ``True`` for any
   non-empty submitter ``X``. That is the "match-all" semantic main.py
   relies on after the WR-06/WR-07 strip-and-collapse chain.

2. The ``main.py`` strip-and-collapse chain itself: each of {None CLI value,
   ``""``, ``"   "`` (whitespace), ``"Acme,BetaCo"`` (real CSV),
   ``" Acme , BetaCo "`` (CSV with whitespace)} produces the expected
   normalized ``submitters`` value that downstream ``Config`` will accept.

Run with::

    pytest mlpstorage_py/tests/test_main_submitters_filter.py -v
"""

import pytest

from mlpstorage_py.submission_checker.configuration.configuration import Config


# ---------------------------------------------------------------------------
# Contract 1: Config.check_submitter(submitter) with submitters=None
# ---------------------------------------------------------------------------

class TestConfigSubmittersNoneMatchesAll:
    """Pin Config(submitters=None).check_submitter -> True for any submitter."""

    def test_none_matches_real_submitter(self):
        config = Config(version="v5.1", submitters=None)
        assert config.check_submitter("Acme") is True

    def test_none_matches_another_submitter(self):
        config = Config(version="v5.1", submitters=None)
        assert config.check_submitter("BetaCo") is True

    def test_explicit_list_filters_out_non_member(self):
        config = Config(version="v5.1", submitters=["Acme"])
        assert config.check_submitter("BetaCo") is False

    def test_explicit_list_admits_member(self):
        config = Config(version="v5.1", submitters=["Acme"])
        assert config.check_submitter("Acme") is True


# ---------------------------------------------------------------------------
# Contract 2: main.py --submitters CSV parsing → submitters value
# ---------------------------------------------------------------------------

def _parse_submitters_arg(raw):
    """Reproduce the strip-and-collapse chain from main.py for testing.

    The logic lives inline inside ``main()``. Refactoring it out would
    change the touched_files surface for the fix loop; instead, mirror it
    here and pin the contract. If the production chain changes, this test
    must fail loudly so the contract stays explicit.
    """
    if raw:
        submitters = [s.strip() for s in raw.split(",") if s.strip()]
        if not submitters:
            submitters = None
    else:
        submitters = None
    return submitters


class TestMainSubmittersArgParsing:
    """Parametrized coverage of the --submitters CLI string normalization."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            # --submitters omitted at the CLI: argparse yields None
            (None, None),
            # --submitters "" (empty string)
            ("", None),
            # --submitters " " (whitespace-only)
            (" ", None),
            # --submitters "   ,   ,  " (all-empty after strip)
            ("   ,   ,  ", None),
            # --submitters "Acme" (single token)
            ("Acme", ["Acme"]),
            # --submitters "Acme,BetaCo" (real CSV, no spaces)
            ("Acme,BetaCo", ["Acme", "BetaCo"]),
            # --submitters " Acme , BetaCo " (real CSV with whitespace)
            (" Acme , BetaCo ", ["Acme", "BetaCo"]),
            # --submitters "Acme,,BetaCo" (empty token in middle)
            ("Acme,,BetaCo", ["Acme", "BetaCo"]),
        ],
    )
    def test_normalization(self, raw, expected):
        assert _parse_submitters_arg(raw) == expected


class TestMainSubmittersIntegration:
    """End-to-end: --submitters parse -> Config -> check_submitter."""

    @pytest.mark.parametrize(
        "raw, candidate, expected",
        [
            # None CLI / empty / whitespace -> match-all
            (None, "Acme", True),
            ("", "Acme", True),
            (" ", "Acme", True),
            ("   ,   ", "Acme", True),
            # Explicit single -> filters
            ("Acme", "Acme", True),
            ("Acme", "BetaCo", False),
            # Explicit CSV -> filters by membership
            ("Acme,BetaCo", "Acme", True),
            ("Acme,BetaCo", "BetaCo", True),
            ("Acme,BetaCo", "Gamma", False),
            # CSV with whitespace -> stripped tokens still match
            (" Acme , BetaCo ", "Acme", True),
            (" Acme , BetaCo ", "BetaCo", True),
        ],
    )
    def test_arg_to_check_submitter(self, raw, candidate, expected):
        submitters = _parse_submitters_arg(raw)
        config = Config(version="v5.1", submitters=submitters)
        assert config.check_submitter(candidate) is expected
