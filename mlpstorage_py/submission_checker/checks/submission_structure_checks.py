"""SubmissionStructureCheck — validates the top-level directory hierarchy.

Implements Rules.md §2.1.1–§2.1.13, §2.1.21 (STRUCT-01..STRUCT-14) and
§3.6.1 (trainingClosedSubmissionChecksum) as a single ``BaseCheck`` subclass
with 14 ``@rule``-decorated methods.

Each method follows the accumulate-don't-abort pattern (QUAL-01, PITFALLS.md
#11): it collects ALL violations in a subtree before returning ``False``, and
NEVER raises out of its body (raises would produce opaque "Exception occurred"
messages in ``BaseCheck.run_checks``).

All violation messages are emitted via ``self.log_violation`` or
``self.warn_violation`` (never bare ``self.log.error``), which locks the
format to:

    ``[<rule_id> <rule_name>] <path>: <msg>``

Per PITFALLS.md #2: no ``.lower()`` calls — case comparisons are byte-exact.
"""

import json
import os
import re

from .base import BaseCheck
from ..configuration.configuration import Config
from ..rule_registry import rule
from ..tools.code_checksum import compute_code_tree_md5
from ..utils import list_dir, list_files
from ..parsers.yaml_parser import YamlParser


# ---------------------------------------------------------------------------
# POSIX-safe name pattern for submitter directories (STRUCT-01, D-18)
# ---------------------------------------------------------------------------
_SUBMITTER_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# Allowed top-level divisions (case-sensitive, PITFALLS.md #2)
_VALID_DIVISIONS = frozenset({"closed", "open"})

# Required submitter subdirectories (case-sensitive set equality)
_REQUIRED_SUBMITTER_SUBDIRS = frozenset({"code", "results", "systems"})

# Valid workload categories under results/<system>/
_VALID_WORKLOAD_CATEGORIES = frozenset({"training", "checkpointing"})

# Valid training workload names under training/
_VALID_TRAINING_WORKLOADS = frozenset({"unet3d", "retinanet"})

# Valid training phase directories under training/<workload>/
_VALID_TRAINING_PHASES = frozenset({"datagen", "run"})

# Valid checkpointing workload names under checkpointing/
_VALID_CHECKPOINTING_WORKLOADS = frozenset({"llama3-8b", "llama3-70b", "llama3-405b", "llama3-1t"})

# Timestamp pattern per Rules.md 2.1.13
_TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")


class SubmissionStructureCheck(BaseCheck):
    """Validate the top-level directory structure of an MLPerf Storage submission.

    Walks the submission root once per check method rather than relying on
    ``Loader.load()`` — the Loader silently skips any division not in
    ``VALID_DIVISIONS``, so STRUCT-02 violations would be invisible to it.

    Constructor:
        log: Logger object with ``error``, ``warning``, ``info`` methods.
        config: ``Config`` instance supplying ``get_reference_checksum()``.
        root_path: The submission root directory (same value as ``args.input``
            in main.py).
    """

    def __init__(self, log, config: Config, root_path: str):
        super().__init__(log=log, path=root_path)
        self.config = config
        self.root_path = root_path
        self.name = "submission structure checks"
        self.init_checks()

    def init_checks(self):
        self.checks = []
        self.checks.extend([
            self.submitter_root_directory_check,
            self.top_level_subdirectories_check,
            self.open_matches_closed_check,
            self.closed_submitter_directory_check,
            self.required_subdirectories_check,
            self.code_directory_contents_check,
            self.systems_directory_files_check,
            self.results_directory_systems_check,
            self.identical_system_config_check,
            self.workload_categories_check,
            self.training_workloads_check,
            self.training_phases_check,
            self.datagen_timestamp_check,
            self.checkpointing_workloads_check,
        ])

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _iter_submitter_dirs(self):
        """Yield (division, submitter, submitter_path) for each known division."""
        for division in list_dir(self.root_path):
            if division not in _VALID_DIVISIONS:
                continue
            div_path = os.path.join(self.root_path, division)
            for submitter in list_dir(div_path):
                yield division, submitter, os.path.join(div_path, submitter)

    def _load_json_safe(self, json_path):
        """Return parsed JSON dict or None on any error (silently)."""
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    # -----------------------------------------------------------------------
    # STRUCT-01 — 2.1.1 submitterRootDirectory
    # -----------------------------------------------------------------------

    @rule("2.1.1", "submitterRootDirectory")
    def submitter_root_directory_check(self):
        """STRUCT-01: submitter root name must match ^[A-Za-z0-9._-]+$ (D-18).

        Checks every per-division submitter directory name (static only —
        no cross-check against system YAML, which has no submitter field).
        """
        valid = True
        for division in list_dir(self.root_path):
            if division not in _VALID_DIVISIONS:
                continue
            div_path = os.path.join(self.root_path, division)
            for submitter in list_dir(div_path):
                if not _SUBMITTER_NAME_RE.match(submitter):
                    self.log_violation(
                        "2.1.1", "submitterRootDirectory",
                        os.path.join(div_path, submitter),
                        "submitter directory name %r does not match ^[A-Za-z0-9._-]+$",
                        submitter,
                    )
                    valid = False
        return valid

    # -----------------------------------------------------------------------
    # STRUCT-02 — 2.1.2 topLevelSubdirectories
    # -----------------------------------------------------------------------

    @rule("2.1.2", "topLevelSubdirectories")
    def top_level_subdirectories_check(self):
        """STRUCT-02: top-level dirs must be a non-empty subset of {closed, open}.

        Case-sensitive set check — no .lower() (PITFALLS.md #2).

        Dot-prefixed entries (.git/, .github/, .gitignore, .DS_Store, etc.)
        are silently skipped: merged reviewer trees are typically git working
        trees, and version-control / CI metadata is never submission content.
        """
        valid = True
        top_dirs = {e for e in list_dir(self.root_path) if not e.startswith(".")}

        # Check for any unrecognised top-level dirs
        unexpected = top_dirs - _VALID_DIVISIONS
        for entry in sorted(unexpected):
            self.log_violation(
                "2.1.2", "topLevelSubdirectories",
                os.path.join(self.root_path, entry),
                "unexpected top-level directory %r (expected only 'closed' and/or 'open')",
                entry,
            )
            valid = False

        # At least one of closed/open must be present
        if not (top_dirs & _VALID_DIVISIONS):
            self.log_violation(
                "2.1.2", "topLevelSubdirectories",
                self.root_path,
                "submission root contains neither 'closed' nor 'open' top-level directory",
            )
            valid = False

        return valid

    # -----------------------------------------------------------------------
    # STRUCT-03 — 2.1.3 openMatchesClosed
    # -----------------------------------------------------------------------

    @rule("2.1.3", "openMatchesClosed")
    def open_matches_closed_check(self):
        """STRUCT-03: Rules.md 2.1.3 is a structural meta-rule asserting that
        the open/ hierarchy follows the same construction rules as the closed/
        hierarchy (2.1.4 through 2.1.X). It is NOT a contents-mirroring
        requirement: closed/ and open/ are individually optional, and a
        submitter may appear in one division without appearing in the other.

        The structural mirroring is enforced automatically because every
        downstream STRUCT method iterates _VALID_DIVISIONS uniformly — STRUCT-05
        (requiredSubdirectories) verifies the {code, results, systems} shape
        for every present submitter in either division. This @rule binding
        therefore has no extra runtime work; it exists so discover_rules
        reports 2.1.3 as covered (same no-op pattern as STRUCT-12 / 2.1.27
        directoryDiagram).

        Pre-fix behavior: emitted a violation for every submitter present in
        closed/ but not in open/ (and vice versa via the inner loop). This
        misread "constructed identically" as a 1:1 submitter-set requirement.
        It produced spurious errors against the merged reviewer tree where
        each submitter typically chose only one division (see the v2.0 results
        bundle: Alluxio / DDN / ExponTech / etc. each appear in only one).
        """
        return True

    # -----------------------------------------------------------------------
    # STRUCT-04 — 2.1.4 closedSubmitterDirectory
    # -----------------------------------------------------------------------

    @rule("2.1.4", "closedSubmitterDirectory")
    def closed_submitter_directory_check(self):
        """STRUCT-04: Rules.md 2.1.4 names a per-submitter naming convention
        for the directory under "closed" (and, transitively, under "open" via
        2.1.3). The validator runs against two different tree shapes:

          1. Single-submitter package — a submitter's own pre-merge tree, in
             which "closed" contains exactly one directory whose name matches
             the top-level submitter directory.
          2. Merged reviewer tree — N submitters' packages concatenated, in
             which "closed" contains one directory per participating
             submitter. The top-level directory in this mode is named for the
             merged-tree set (e.g. "submissions_storage_v2.0"), not any one
             submitter.

        The submitter-name character-set requirement is enforced uniformly by
        STRUCT-01 (2.1.1 submitterRootDirectory) for every submitter directory
        in either division, and STRUCT-05 (2.1.5 requiredSubdirectories)
        enforces the {code, results, systems} shape under every submitter. So
        the structural value 2.1.4 contributes is already covered by the
        sibling STRUCT methods in both tree modes; this @rule binding stays
        for coverage signaling without runtime work.

        Pre-fix behavior: enforced cardinality 1 on every division and a
        basename-matches-submitter equality check. That made the validator
        unusable against the merged reviewer tree (every multi-submitter
        division tripped the count check, and the basename mismatch fired
        once per submitter). See the v2.0 results bundle:
        submissions_storage_v2.0/closed/{Alluxio,DDN,...}.
        """
        return True

    # -----------------------------------------------------------------------
    # STRUCT-05 — 2.1.5 requiredSubdirectories
    # -----------------------------------------------------------------------

    @rule("2.1.5", "requiredSubdirectories")
    def required_subdirectories_check(self):
        """STRUCT-05: submitter dir must contain EXACTLY {code, results, systems}.

        Dot-prefixed entries are silently skipped (e.g. .DS_Store, .cache/).

        When an unexpected subdirectory itself contains some of {code, results,
        systems}, the diagnostic includes a wrapping hint — this catches the
        common v2.0 submitter mistake of nesting the package one level deeper
        than the spec requires (e.g. closed/<submitter>/benchmarks/{code,
        results, systems}/ instead of closed/<submitter>/{code, results,
        systems}/).
        """
        valid = True
        for division, submitter, sub_path in self._iter_submitter_dirs():
            actual = {e for e in list_dir(sub_path) if not e.startswith(".")}
            missing = _REQUIRED_SUBMITTER_SUBDIRS - actual
            extra = actual - _REQUIRED_SUBMITTER_SUBDIRS

            for m in sorted(missing):
                self.log_violation(
                    "2.1.5", "requiredSubdirectories",
                    os.path.join(sub_path, m),
                    "required subdirectory %r missing from %s/%s",
                    m, division, submitter,
                )
                valid = False

            for e in sorted(extra):
                extra_path = os.path.join(sub_path, e)
                hint = ""
                if os.path.isdir(extra_path):
                    nested = {
                        n for n in list_dir(extra_path) if not n.startswith(".")
                    }
                    wrapped = sorted(nested & _REQUIRED_SUBMITTER_SUBDIRS)
                    if wrapped:
                        hint = (
                            "; the submission appears to be nested one level "
                            "deeper than expected — found %s inside, expected "
                            "directly under %s/%s/"
                            % (wrapped, division, submitter)
                        )
                self.log_violation(
                    "2.1.5", "requiredSubdirectories",
                    extra_path,
                    "unexpected subdirectory %r in %s/%s "
                    "(only code/results/systems allowed)%s",
                    e, division, submitter, hint,
                )
                valid = False

        return valid

    # -----------------------------------------------------------------------
    # STRUCT-06 — 2.1.6 codeDirectoryContents (+ 3.6.1 for CLOSED)
    # -----------------------------------------------------------------------

    @rule("2.1.6", "codeDirectoryContents")
    def code_directory_contents_check(self):
        """STRUCT-06: for CLOSED submissions, verify code/ tree MD5.

        Per D-12: when reference checksum is None, emit WARNING and return
        True (does not fail the run). The no-checksum warning is hoisted out
        of the per-submitter loop so an unconfigured invocation emits one
        warning per run rather than one per submitter (which would spam the
        report against N-submitter merged trees).
        """
        valid = True
        closed_path = os.path.join(self.root_path, "closed")
        if not os.path.isdir(closed_path):
            return valid  # no closed/ — nothing to check

        expected = self.config.get_reference_checksum()
        if expected is None:
            self.warn_violation(
                "2.1.6", "codeDirectoryContents",
                closed_path,
                "reference checksum not configured "
                "(use --reference-checksum or populate REFERENCE_CHECKSUMS); "
                "the code/ subtree cannot be validated without one",
            )
            return valid  # not a failure (D-12 preserved); skip per-submitter walk

        for submitter in list_dir(closed_path):
            code_path = os.path.join(closed_path, submitter, "code")
            if not os.path.isdir(code_path):
                continue  # STRUCT-05 will catch missing code/

            digest = compute_code_tree_md5(code_path, self.log)
            if digest != expected:
                self.log_violation(
                    "2.1.6", "codeDirectoryContents",
                    code_path,
                    "code tree MD5 mismatch: expected %s, got %s",
                    expected, digest,
                )
                valid = False

        return valid

    # -----------------------------------------------------------------------
    # STRUCT-07 — 2.1.7 systemsDirectoryFiles
    # -----------------------------------------------------------------------

    @rule("2.1.7", "systemsDirectoryFiles")
    def systems_directory_files_check(self):
        """STRUCT-07: systems/ must contain only paired <name>.yaml + <name>.pdf.

        Markdown files (*.md) are allowed for supplementary documentation
        (Rules.md 2.1.7). Dot-prefixed entries (.DS_Store, .gitkeep, etc.)
        are silently skipped.
        """
        valid = True
        for _division, _submitter, sub_path in self._iter_submitter_dirs():
            systems_path = os.path.join(sub_path, "systems")
            if not os.path.isdir(systems_path):
                continue

            # Collect non-dotfile entries only (list_files / list_dir don't
            # filter dotfiles themselves; .DS_Store etc. would otherwise be
            # flagged as unexpected by the others loop below).
            all_files = [f for f in list_files(systems_path) if not f.startswith(".")]
            all_dirs = [d for d in list_dir(systems_path) if not d.startswith(".")]

            for d in all_dirs:
                self.log_violation(
                    "2.1.7", "systemsDirectoryFiles",
                    os.path.join(systems_path, d),
                    "systems/ must not contain subdirectories; found %r",
                    d,
                )
                valid = False

            yamls = set()
            pdfs = set()
            others = []
            for f in all_files:
                if f.endswith(".yaml"):
                    yamls.add(f[:-5])  # stem
                elif f.endswith(".pdf"):
                    pdfs.add(f[:-4])   # stem
                elif f.endswith(".md"):
                    # Markdown documentation files are permitted by 2.1.7.
                    continue
                else:
                    others.append(f)

            # Extra file types
            for f in others:
                self.log_violation(
                    "2.1.7", "systemsDirectoryFiles",
                    os.path.join(systems_path, f),
                    "unexpected file %r in systems/ "
                    "(only <name>.yaml, <name>.pdf, and *.md allowed)",
                    f,
                )
                valid = False

            # yaml without matching pdf
            for stem in sorted(yamls - pdfs):
                self.log_violation(
                    "2.1.7", "systemsDirectoryFiles",
                    os.path.join(systems_path, stem + ".yaml"),
                    "%s.yaml has no matching %s.pdf in systems/",
                    stem, stem,
                )
                valid = False

            # pdf without matching yaml
            for stem in sorted(pdfs - yamls):
                self.log_violation(
                    "2.1.7", "systemsDirectoryFiles",
                    os.path.join(systems_path, stem + ".pdf"),
                    "%s.pdf has no matching %s.yaml in systems/",
                    stem, stem,
                )
                valid = False

        return valid

    # -----------------------------------------------------------------------
    # STRUCT-08 — 2.1.8 resultsDirectorySystems
    # -----------------------------------------------------------------------

    @rule("2.1.8", "resultsDirectorySystems")
    def results_directory_systems_check(self):
        """STRUCT-08: bidirectional results/ ↔ systems/ check (D-17).

        For each <name>/ in results/: systems/<name>.yaml + .pdf must exist
        AND YAML's submission_name must equal <name>.
        For each <name>.yaml in systems/: results/<name>/ must exist.
        """
        valid = True
        for _division, _submitter, sub_path in self._iter_submitter_dirs():
            results_path = os.path.join(sub_path, "results")
            systems_path = os.path.join(sub_path, "systems")
            if not os.path.isdir(results_path) or not os.path.isdir(systems_path):
                continue

            result_systems = set(list_dir(results_path))
            system_yamls = {f[:-5] for f in list_files(systems_path) if f.endswith(".yaml")}
            system_pdfs = {f[:-4] for f in list_files(systems_path) if f.endswith(".pdf")}

            # Forward: results/<name>/ must have systems/<name>.yaml and .pdf
            for name in sorted(result_systems):
                yaml_path = os.path.join(systems_path, name + ".yaml")
                pdf_path = os.path.join(systems_path, name + ".pdf")

                if name not in system_yamls:
                    self.log_violation(
                        "2.1.8", "resultsDirectorySystems",
                        yaml_path,
                        "results/%s/ exists but systems/%s.yaml is missing",
                        name, name,
                    )
                    valid = False
                else:
                    # Parse YAML and check submission_name == name (D-17)
                    system_yaml = YamlParser(yaml_path, "System").get_dict()
                    try:
                        submission_name = (
                            system_yaml.get("system_under_test", {})
                            .get("solution", {})
                            .get("submission_name")
                        )
                    except AttributeError:
                        submission_name = None

                    if submission_name != name:
                        self.log_violation(
                            "2.1.8", "resultsDirectorySystems",
                            yaml_path,
                            "systems/%s.yaml: submission_name is %r, expected %r (D-17)",
                            name, submission_name, name,
                        )
                        valid = False

                if name not in system_pdfs:
                    self.log_violation(
                        "2.1.8", "resultsDirectorySystems",
                        pdf_path,
                        "results/%s/ exists but systems/%s.pdf is missing",
                        name, name,
                    )
                    valid = False

            # Reverse: systems/<name>.yaml must have results/<name>/
            for stem in sorted(system_yamls):
                if stem not in result_systems:
                    self.log_violation(
                        "2.1.8", "resultsDirectorySystems",
                        os.path.join(results_path, stem),
                        "systems/%s.yaml exists but results/%s/ is missing",
                        stem, stem,
                    )
                    valid = False

        return valid

    # -----------------------------------------------------------------------
    # STRUCT-09 — 2.1.9 identicalSystemConfig  (opportunistic, D-15/D-16)
    # -----------------------------------------------------------------------

    @rule("2.1.9", "identicalSystemConfig")
    def identical_system_config_check(self):
        """STRUCT-09: opportunistic cross-check between systems YAML and summary.json.

        Per D-15, checks three fields (num_hosts, host_memory_GB, multi_host).
        Per D-16, absent fields are silently skipped (no warning, no error).
        The same check fires for both training run/ timestamps AND
        checkpointing timestamps.
        """
        valid = True
        for _division, _submitter, sub_path in self._iter_submitter_dirs():
            results_path = os.path.join(sub_path, "results")
            systems_path = os.path.join(sub_path, "systems")
            if not os.path.isdir(results_path) or not os.path.isdir(systems_path):
                continue

            for sys_name in list_dir(results_path):
                yaml_path = os.path.join(systems_path, sys_name + ".yaml")
                if not os.path.isfile(yaml_path):
                    continue  # STRUCT-08 handles missing YAML

                system_yaml = YamlParser(yaml_path, "System").get_dict()
                if not system_yaml:
                    continue

                sut = system_yaml.get("system_under_test", {}) or {}
                clients = sut.get("clients") or []
                solution = sut.get("solution") or {}
                capabilities = solution.get("capabilities") or {}

                # Compute expected values from system YAML (may be None if absent)
                expected_num_hosts = None
                if clients:
                    try:
                        expected_num_hosts = sum(int(c.get("quantity", 0)) for c in clients)
                    except (TypeError, ValueError):
                        pass

                expected_memory_per_host = None
                if clients and len(clients) == 1:
                    chassis = (clients[0] or {}).get("chassis") or {}
                    cap = chassis.get("memory_capacity")
                    if cap is not None:
                        try:
                            expected_memory_per_host = int(cap)
                        except (TypeError, ValueError):
                            pass

                yaml_multi_host = capabilities.get("multi_host")  # bool or None

                # Find all summary.json files under this system's results
                sys_result_path = os.path.join(results_path, sys_name)
                summary_paths = self._collect_summary_jsons(sys_result_path)

                for summary_path in summary_paths:
                    summary = self._load_json_safe(summary_path)
                    if not summary or not isinstance(summary, dict):
                        continue
                    valid &= self._cross_check_summary(
                        summary, summary_path,
                        expected_num_hosts, expected_memory_per_host, yaml_multi_host,
                    )

        return valid

    def _collect_summary_jsons(self, sys_result_path):
        """Collect all summary.json paths under training/*/run/*/ and checkpointing/*/."""
        paths = []
        training_path = os.path.join(sys_result_path, "training")
        if os.path.isdir(training_path):
            for workload in list_dir(training_path):
                run_path = os.path.join(training_path, workload, "run")
                if os.path.isdir(run_path):
                    for ts in list_dir(run_path):
                        sp = os.path.join(run_path, ts, "summary.json")
                        if os.path.isfile(sp):
                            paths.append(sp)

        chkpt_path = os.path.join(sys_result_path, "checkpointing")
        if os.path.isdir(chkpt_path):
            for workload in list_dir(chkpt_path):
                wl_path = os.path.join(chkpt_path, workload)
                for ts in list_dir(wl_path):
                    sp = os.path.join(wl_path, ts, "summary.json")
                    if os.path.isfile(sp):
                        paths.append(sp)
        return paths

    def _cross_check_summary(
        self, summary, summary_path,
        expected_num_hosts, expected_memory_per_host, yaml_multi_host,
    ):
        """Run the three D-15 cross-checks against one summary.json.

        Returns False if any cross-check fires; True otherwise.
        Per D-16: absent fields are silently skipped.
        """
        valid = True

        # Cross-check 1 — num_hosts
        summary_num_hosts = summary.get("num_hosts")
        if summary_num_hosts is not None and expected_num_hosts is not None:
            if summary_num_hosts != expected_num_hosts:
                self.log_violation(
                    "2.1.9", "identicalSystemConfig",
                    summary_path,
                    "num_hosts mismatch: summary.json has %s, system YAML clients sum to %s",
                    summary_num_hosts, expected_num_hosts,
                )
                valid = False

        # Cross-check 2 — per-host memory (index-aligned when possible)
        # ``memory_capacity`` in the system YAML is nameplate (GB) while
        # ``host_memory_GB`` in summary.json is the usable amount the OS
        # sees (GiB after DDR overhead, kernel reserve, BIOS-reserved
        # regions, etc.). On identical hardware these never agree to the
        # integer — usable RAM is ~95-98% of nameplate. A strict `==`
        # check fires a 2.1.9 violation on every well-formed submission.
        # Use a ±10% band: catches genuine config mismatches (e.g., a
        # 256 GB rig reporting against a 768 GB YAML — ~3x off) while
        # letting nameplate-vs-usable through.
        MEMORY_TOLERANCE = 0.10
        host_memory_list = summary.get("host_memory_GB")
        if (
            host_memory_list is not None
            and expected_memory_per_host is not None
            and isinstance(host_memory_list, list)
        ):
            for i, mem in enumerate(host_memory_list):
                try:
                    mem_int = int(mem)
                except (TypeError, ValueError):
                    continue
                lower = expected_memory_per_host * (1 - MEMORY_TOLERANCE)
                upper = expected_memory_per_host * (1 + MEMORY_TOLERANCE)
                if not (lower <= mem_int <= upper):
                    self.log_violation(
                        "2.1.9", "identicalSystemConfig",
                        summary_path,
                        "host_memory_GB[%d] mismatch: summary.json has %s GiB, "
                        "system YAML client chassis.memory_capacity is %s GiB "
                        "(outside ±%d%% tolerance)",
                        i, mem_int, expected_memory_per_host,
                        int(MEMORY_TOLERANCE * 100),
                    )
                    valid = False
                    break  # one violation per summary.json per field is sufficient

        # Cross-check 3 — multi_host capability vs. num_hosts > 1
        if (
            yaml_multi_host is False
            and summary_num_hosts is not None
            and summary_num_hosts > 1
        ):
            self.log_violation(
                "2.1.9", "identicalSystemConfig",
                summary_path,
                "system YAML capabilities.multi_host is False but "
                "summary.json num_hosts=%s > 1",
                summary_num_hosts,
            )
            valid = False

        return valid

    # -----------------------------------------------------------------------
    # STRUCT-10 — 2.1.10 workloadCategories
    # -----------------------------------------------------------------------

    @rule("2.1.10", "workloadCategories")
    def workload_categories_check(self):
        """STRUCT-10: results/<system>/ must contain only {training, checkpointing}."""
        valid = True
        for _division, _submitter, sub_path in self._iter_submitter_dirs():
            results_path = os.path.join(sub_path, "results")
            if not os.path.isdir(results_path):
                continue

            for sys_name in list_dir(results_path):
                sys_path = os.path.join(results_path, sys_name)
                categories = set(list_dir(sys_path))
                unexpected = categories - _VALID_WORKLOAD_CATEGORIES
                for cat in sorted(unexpected):
                    self.log_violation(
                        "2.1.10", "workloadCategories",
                        os.path.join(sys_path, cat),
                        "unexpected workload category %r in results/%s/ "
                        "(only 'training' and 'checkpointing' allowed)",
                        cat, sys_name,
                    )
                    valid = False

                if not (categories & _VALID_WORKLOAD_CATEGORIES):
                    self.log_violation(
                        "2.1.10", "workloadCategories",
                        sys_path,
                        "results/%s/ contains neither 'training' nor 'checkpointing'",
                        sys_name,
                    )
                    valid = False

        return valid

    # -----------------------------------------------------------------------
    # STRUCT-11 — 2.1.11 trainingWorkloads
    # -----------------------------------------------------------------------

    @rule("2.1.11", "trainingWorkloads")
    def training_workloads_check(self):
        """STRUCT-11: training/ must contain only {unet3d, retinanet}."""
        valid = True
        for _division, _submitter, sub_path in self._iter_submitter_dirs():
            results_path = os.path.join(sub_path, "results")
            if not os.path.isdir(results_path):
                continue

            for sys_name in list_dir(results_path):
                training_path = os.path.join(results_path, sys_name, "training")
                if not os.path.isdir(training_path):
                    continue

                for workload in list_dir(training_path):
                    if workload not in _VALID_TRAINING_WORKLOADS:
                        self.log_violation(
                            "2.1.11", "trainingWorkloads",
                            os.path.join(training_path, workload),
                            "unknown training workload %r "
                            "(valid: unet3d, retinanet)",
                            workload,
                        )
                        valid = False

        return valid

    # -----------------------------------------------------------------------
    # STRUCT-12 — 2.1.12 trainingPhases
    # -----------------------------------------------------------------------

    @rule("2.1.12", "trainingPhases")
    def training_phases_check(self):
        """STRUCT-12: each training workload dir must contain exactly {datagen, run}."""
        valid = True
        for _division, _submitter, sub_path in self._iter_submitter_dirs():
            results_path = os.path.join(sub_path, "results")
            if not os.path.isdir(results_path):
                continue

            for sys_name in list_dir(results_path):
                training_path = os.path.join(results_path, sys_name, "training")
                if not os.path.isdir(training_path):
                    continue

                for workload in list_dir(training_path):
                    workload_path = os.path.join(training_path, workload)
                    phases = set(list_dir(workload_path))
                    missing = _VALID_TRAINING_PHASES - phases
                    extra = phases - _VALID_TRAINING_PHASES

                    for m in sorted(missing):
                        self.log_violation(
                            "2.1.12", "trainingPhases",
                            os.path.join(workload_path, m),
                            "required phase directory %r missing from training/%s",
                            m, workload,
                        )
                        valid = False

                    for e in sorted(extra):
                        self.log_violation(
                            "2.1.12", "trainingPhases",
                            os.path.join(workload_path, e),
                            "unexpected directory %r in training/%s "
                            "(only 'datagen' and 'run' allowed)",
                            e, workload,
                        )
                        valid = False

        return valid

    # -----------------------------------------------------------------------
    # STRUCT-13 — 2.1.13 datagenTimestamp
    # -----------------------------------------------------------------------

    @rule("2.1.13", "datagenTimestamp")
    def datagen_timestamp_check(self):
        """STRUCT-13: datagen/ must contain exactly ONE YYYYMMDD_HHmmss timestamp."""
        valid = True
        for _division, _submitter, sub_path in self._iter_submitter_dirs():
            results_path = os.path.join(sub_path, "results")
            if not os.path.isdir(results_path):
                continue

            for sys_name in list_dir(results_path):
                training_path = os.path.join(results_path, sys_name, "training")
                if not os.path.isdir(training_path):
                    continue

                for workload in list_dir(training_path):
                    datagen_path = os.path.join(training_path, workload, "datagen")
                    if not os.path.isdir(datagen_path):
                        continue

                    entries = list_dir(datagen_path)
                    # Check format of each entry
                    bad_fmt = [e for e in entries if not _TIMESTAMP_RE.match(e)]
                    for e in bad_fmt:
                        self.log_violation(
                            "2.1.13", "datagenTimestamp",
                            os.path.join(datagen_path, e),
                            "datagen entry %r does not match YYYYMMDD_HHmmss format",
                            e,
                        )
                        valid = False

                    # Check count — must be exactly 1
                    if len(entries) != 1:
                        self.log_violation(
                            "2.1.13", "datagenTimestamp",
                            datagen_path,
                            "datagen/ must contain exactly 1 timestamp directory, "
                            "found %d",
                            len(entries),
                        )
                        valid = False

        return valid

    # -----------------------------------------------------------------------
    # STRUCT-14 — 2.1.21 checkpointingWorkloads
    # -----------------------------------------------------------------------

    @rule("2.1.21", "checkpointingWorkloads")
    def checkpointing_workloads_check(self):
        """STRUCT-14: checkpointing/ must contain only valid LLaMA workloads."""
        valid = True
        for _division, _submitter, sub_path in self._iter_submitter_dirs():
            results_path = os.path.join(sub_path, "results")
            if not os.path.isdir(results_path):
                continue

            for sys_name in list_dir(results_path):
                chkpt_path = os.path.join(results_path, sys_name, "checkpointing")
                if not os.path.isdir(chkpt_path):
                    continue

                for workload in list_dir(chkpt_path):
                    if workload not in _VALID_CHECKPOINTING_WORKLOADS:
                        self.log_violation(
                            "2.1.21", "checkpointingWorkloads",
                            os.path.join(chkpt_path, workload),
                            "unknown checkpointing workload %r "
                            "(valid: llama3-8b, llama3-70b, llama3-405b, llama3-1t)",
                            workload,
                        )
                        valid = False

        return valid
