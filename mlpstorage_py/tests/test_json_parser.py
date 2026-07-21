"""Unit tests for the submission-checker JSONParser.

Covers the ``in`` membership operator (``__contains__``), which referenced a
non-existent ``self.messages`` attribute and raised ``AttributeError`` for any
``key in parser`` test — a latent bug (callers use ``[]`` / ``.get()``).
"""

import json

from mlpstorage_py.submission_checker.parsers.json_parser import JSONParser


def _parser(tmp_path, payload):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return JSONParser(str(path))


def test_contains_top_level_key(tmp_path):
    """``key in parser`` is True for a present top-level key, False otherwise."""
    parser = _parser(tmp_path, {"alpha": 1, "beta": 2})
    assert "alpha" in parser
    assert "beta" in parser
    assert "missing" not in parser


def test_contains_on_normalized_non_dict(tmp_path):
    """A non-dict JSON body is normalized under ``summary`` and is found via ``in``."""
    parser = _parser(tmp_path, [1, 2, 3])
    assert "summary" in parser
    assert "alpha" not in parser
