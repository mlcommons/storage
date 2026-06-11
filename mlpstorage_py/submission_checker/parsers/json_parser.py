import json
import logging
from typing import Literal


class JSONParser:
    """Parse a JSON summary file and provide dict-like access.

    Example:
        p = SummaryParser("summary.json")
        value = p["some_key"]
    """

    def __init__(
        self,
        path,
        name: Literal["Summary", "Metadata", "System"] = "Summary",
    ):
        self.path = path
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.d = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.d = json.load(f)
        except FileNotFoundError:
            self.logger.error("Summary file not found: %s", path)
        except json.JSONDecodeError as exc:
            self.logger.error("Invalid JSON in summary file %s: %s", path, exc)
        if not isinstance(self.d, dict):
            # normalize to dict for consistent API
            self.d = {"summary": self.d}
        self.keys = set(self.d.keys())

    def __getitem__(self, key):
        """Return the value for `key` or None if missing."""
        return self.d.get(key)

    def get(self, key, default=None):
        """Return the value for `key`, or `default` if not present."""
        return self.d.get(key, default)

    def get_dict(self):
        """Return the full parsed JSON as a dict."""
        return self.d

    def get_keys(self):
        """Return a set of top-level keys in the summary."""
        return self.keys

    def __contains__(self, key):
        return key in self.messages

    def __repr__(self):
        return f"<SummaryParser path={self.path!r} keys={len(self.keys)}>"
