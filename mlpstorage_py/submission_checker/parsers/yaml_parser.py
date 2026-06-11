import logging
from typing import Literal
import yaml



class YamlParser:
    """Parse a YAML summary file and provide dict-like access.

    Example:
        p = YamlParser("summary.yaml")
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

        if yaml is None:
            self.logger.error("PyYAML is not installed; cannot parse YAML: %s", path)
        else:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.d = yaml.safe_load(f)
            except FileNotFoundError:
                self.logger.error("YAML file not found: %s", path)
            except Exception as exc:
                # yaml.YAMLError and others
                self.logger.error("Invalid YAML in file %s: %s", path, exc)

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
        """Return the full parsed YAML as a dict."""
        return self.d

    def get_keys(self):
        """Return a set of top-level keys in the summary."""
        return self.keys

    def __contains__(self, key):
        return key in self.d

    def __repr__(self):
        return f"<YamlParser path={self.path!r} keys={len(self.keys)}>"
