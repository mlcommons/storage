"""SystemYamlSchemaCheck — Phase 2 D-A1.

Implements ``schema_validate_all_system_yamls``: walks every
``<division>/<submitter>/systems/<name>.yaml`` under the submission root,
calls ``validate_file`` from ``schema_validator``, and emits one
``log_violation`` per Pydantic error string.  Errors are tagged with the
rule_id from ``SCHEMA_ERROR_RULE_MAP``; unmapped loc strings fall through to
the ``("2.1.7", "systemsDirectoryFiles")`` default (D-A2).

Runs once before the ``for logs in loader.load():`` loop in ``main.py``
(D-A1, mirrors Phase 1's ``SubmissionStructureCheck`` plug-in pattern).
Returns bool; feeds the existing ``errors`` accumulator without aborting
(QUAL-01: accumulate-don't-abort).

Empirical verification — Rule 13 loc string (2026-06-10):
  Pydantic v2 ``model_validator(mode='after')`` on the ``Capabilities``
  model fires at loc = (``system_under_test``, ``solution``, ``capabilities``)
  which stringifies to ``"system_under_test -> solution -> capabilities"``.
  Contrary to RESEARCH.md §Surfaced Gray Areas #4 (which predicted an empty
  tuple / empty string), Pydantic v2 propagates the path to the container
  model through which the validator ran.  Both Rule-13 trigger conditions
  (both_simultaneous AND remap≠0; either_false AND remap==0) produce the
  same loc string.
"""

import os

from .base import BaseCheck
from ..configuration.configuration import Config
from ..utils import list_dir, list_files
from ...system_description.schema_validator import validate_file


class SystemYamlSchemaCheck(BaseCheck):
    """Top-level check that schema-validates every ``systems/<name>.yaml``.

    Runs once before the per-benchmark loader loop (mirrors Phase 1's
    ``SubmissionStructureCheck`` plug-in pattern; D-A1).

    ``SCHEMA_ERROR_RULE_MAP`` maps Pydantic loc strings (formatted as
    ``" -> ".join(str(p) for p in err["loc"])``) to ``(rule_id, rule_name)``
    tuples.  Unmapped loc strings → fallback ``("2.1.7", "systemsDirectoryFiles")``.
    """

    SCHEMA_ERROR_RULE_MAP: dict[str, tuple[str, str]] = {
        # D-A2: locked field paths → (rule_id, rule_name) for violation tagging.
        # Fallback for unmapped paths: ("2.1.7", "systemsDirectoryFiles").

        # Field-level paths for individual capability fields (presence + type).
        "system_under_test -> solution -> capabilities -> remap_time_in_seconds":
            ("4.7.3", "checkpointRemappingTimeReporting"),
        "system_under_test -> solution -> capabilities -> simultaneous_write":
            ("4.7.4", "checkpointSimultaneousRwSupport"),
        "system_under_test -> solution -> capabilities -> simultaneous_read":
            ("4.7.4", "checkpointSimultaneousRwSupport"),
        "system_under_test -> solution -> capabilities -> multi_host":
            ("4.7.4", "checkpointSimultaneousRwSupport"),

        # Rule-13 cross-field entry (Plan 02-02 empirical verification, 2026-06-10):
        # Capabilities.check_remap_time model_validator(mode='after') fires at loc
        # ("system_under_test", "solution", "capabilities") → loc_str below.
        # Observed trigger conditions:
        #   (1) simultaneous_write=True, simultaneous_read=True, remap_time_in_seconds≠0
        #   (2) simultaneous_write=False (or simultaneous_read=False), remap_time_in_seconds==0
        # Both produce loc_str "system_under_test -> solution -> capabilities".
        # Note: RESEARCH.md §Gray Area 4 predicted empty string "" — empirical run
        # showed Pydantic v2 propagates the container path, not an empty tuple.
        "system_under_test -> solution -> capabilities":
            ("4.7.3", "checkpointRemappingTimeReporting"),
    }

    def __init__(self, log, config: Config, root_path: str):
        """Initialize SystemYamlSchemaCheck.

        Args:
            log: Logger instance (same as other check classes).
            config: Config instance (for version and submitter info).
            root_path: Root of the submission tree — e.g. ``args.input``.
        """
        super().__init__(log=log, path=root_path)
        self.config = config
        self.root_path = root_path
        self.name = "system YAML schema checks"
        self.init_checks()

    def init_checks(self):
        """Register check methods.  Called by ``__init__``."""
        self.checks = [self.schema_validate_all_system_yamls]

    def schema_validate_all_system_yamls(self) -> bool:
        """Walk every ``<division>/<submitter>/systems/<name>.yaml`` and validate.

        For each YAML file under ``<root>/{closed,open}/<submitter>/systems/``,
        calls ``validate_file`` and emits one ``log_violation`` per returned
        error string.  The violation rule_id is looked up in
        ``SCHEMA_ERROR_RULE_MAP`` (keyed by the loc string); unmapped loc
        strings fall through to ``("2.1.7", "systemsDirectoryFiles")`` (D-A2).

        Returns:
            True if zero errors were found across all YAMLs.
            False if any error was found (accumulate-don't-abort: every YAML
            and every error within each YAML is checked; reporting does not
            stop on the first failure per QUAL-01 + PITFALLS.md #11).

        Side-effects:
            Calls ``self.log_violation(rule_id, rule_name, yaml_path, '%s', msg)``
            for each error (QUAL-02: lazy-format style, rule-ID prefix).
        """
        valid = True
        if not os.path.isdir(self.root_path):
            return valid  # main.py handles the missing-input case elsewhere

        for division in list_dir(self.root_path):
            if division not in ("closed", "open"):
                continue
            div_path = os.path.join(self.root_path, division)
            if not os.path.isdir(div_path):
                continue
            for submitter in list_dir(div_path):
                systems_path = os.path.join(div_path, submitter, "systems")
                if not os.path.isdir(systems_path):
                    continue  # STRUCT-05 owns the "missing systems/" diagnostic
                for fname in list_files(systems_path):
                    if not fname.endswith(".yaml"):
                        continue
                    yaml_path = os.path.join(systems_path, fname)
                    errors = validate_file(yaml_path)
                    for error_str in errors:
                        if ": " in error_str:
                            loc_str, msg = error_str.split(": ", 1)
                        else:
                            loc_str = ""
                            msg = error_str
                        rule_id, rule_name = self.SCHEMA_ERROR_RULE_MAP.get(
                            loc_str, ("2.1.7", "systemsDirectoryFiles")
                        )
                        if loc_str:
                            self.log_violation(
                                rule_id, rule_name, yaml_path, "%s: %s", loc_str, msg
                            )
                        else:
                            self.log_violation(rule_id, rule_name, yaml_path, "%s", msg)
                        valid = False
        return valid
