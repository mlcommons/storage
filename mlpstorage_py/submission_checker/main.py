import argparse
import logging
import os
import sys

# Constants
from .constants import *

from mlpstorage_py.editions import UncheckableEditionError, load_editions
from mlpstorage_py.provenance import (
    MANIFEST_FILENAME,
    UNKNOWN,
    ProvenanceError,
    read_submission_manifest,
)

# Import config
from .configuration.configuration import Config

# Import loader
from .loader import Loader

# Import checkers
from .checks.checkpointing_checks import CheckpointingCheck
from .checks.directory_checks import DirectoryCheck
from .checks.kvcache_checks import KVCacheCheck
from .checks.pool_structure_checks import PoolStructureCheck
from .checks.provenance_checks import ProvenanceCheck
from .checks.edition_checks import EditionCheck
from .checks.submission_structure_checks import SubmissionStructureCheck
from .checks.system_yaml_schema_checks import SystemYamlSchemaCheck
from .checks.training_checks import TrainingCheck
from .checks.vdb_checks import VdbCheck


# Import result exporter
from .results import ResultExporter

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger("main")

# Per CR-02 (review 2026-06-10): the checker list is mode-routed so that each
# per-mode Check class only runs against submissions whose loader mode matches.
# The previous flat list ran every checker against every submission. That was
# benign while VdbCheck/KVCacheCheck were no-op stubs, but locked in a
# regression for the next contributor — the first real @rule-decorated method
# added to VdbCheck would fire against training/checkpointing submissions and
# emit a §5 rule_id bound to the wrong submission. Gating at the loop keeps
# the stubs (and any future real checks) on the right submissions without each
# rule method needing to guard with `if self.mode != ...`.
#
# Per WR-01 iter-2 (review 2026-06-10): hoisted to module scope (was inside
# main()) for discoverability — contributors adding a new mode now find this
# at the same scope as the loader's `for mode in list_dir(...)` walk and the
# rules_coverage tool's check-class list, rather than buried in a function
# closure. Module scope also enables test-time monkeypatch.setattr(
# "...main.MODE_TO_CHECKERS", {...}) injection of mock checkers, mirroring
# the STUB_COVERAGE/OUT_OF_SCOPE_RULES pattern in test_rules_coverage.py.
MODE_TO_CHECKERS = {
    "training":         [DirectoryCheck, TrainingCheck],
    "checkpointing":    [DirectoryCheck, CheckpointingCheck],
    # Keys must match the on-disk directory names produced by
    # ``mlpstorage_py.rules.utils.generate_output_location`` —
    # ``BENCHMARK_TYPES.name`` for vdb / kvcache is ``vector_database`` /
    # ``kv_cache``. Pre-fix these were keyed ``vectordb`` / ``kvcache``,
    # so the loader's ``mode = list_dir(system_path)[i]`` (which yields
    # the disk-canonical name) never matched and every vdb / kvcache
    # submission tripped the [2.1.10 workloadCategories] unrecognized-mode
    # error at line 174 below (issue #612).
    "vector_database":  [VdbCheck],
    "kv_cache":         [KVCacheCheck],
}

def get_args():
    """Parse command-line arguments for the submission checker.

    Sets up an ArgumentParser with options for input directory, version,
    filtering, output files, and various skip flags for different checks.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="submission directory")
    parser.add_argument("--submitters", help="Comma separated submitters to run the checker")
    parser.add_argument(
        "--csv",
        default="summary.csv",
        help="csv file with results")
    parser.add_argument(
        "--skip-output-file",
        action="store_true",
        help="Skip check output file"
    )
    args = parser.parse_args()
    return args

def config_for_submission(root, division, submitter, tree_config, cache):
    """The Config a submission's workload checks run with.

    The rules edition comes from ``<division>/<submitter>/submission.yaml``
    (``rules_edition``). No manifest, an edition of ``unknown``, a manifest
    PROV-02 cannot parse, or an edition EDN-01 does not know all fall back to
    the tree-wide Config (the current edition) so those rules stay the single
    report. A known edition this tool cannot check returns ``None`` (EDN-04
    reported it pre-loop). Cached per (division, submitter).
    """
    key = (division, submitter)
    if key in cache:
        return cache[key]
    result = tree_config
    manifest_path = os.path.join(root, division, submitter, MANIFEST_FILENAME)
    if os.path.isfile(manifest_path):
        try:
            declared = read_submission_manifest(manifest_path).get("rules_edition")
        except ProvenanceError:
            declared = None
        if declared is not None and str(declared) != UNKNOWN:
            try:
                result = tree_config.for_edition(declared)
            except UncheckableEditionError:
                result = None if load_editions().edition(declared) is not None else tree_config
    cache[key] = result
    return result


def run(args):
    """Run the MLPerf submission checker against a parsed argument namespace.

    Public entry for in-process invocation (e.g. the top-level
    ``mlpstorage validate`` CLI shim, or tests). The standalone
    ``__main__`` entry below builds the namespace via ``get_args()``
    and delegates here.

    Args:
        args: ``argparse.Namespace`` with attributes ``input``,
            ``submitters``, ``csv``, ``skip_output_file``, and
            ``reference_checksum``. The rules edition is not an argument:
            each submission declares its own in ``submission.yaml`` (see
            ``config_for_submission``).

    Returns:
        int: 0 if all submissions pass checks, 1 if any errors found.
    """
    # When --submitters is not supplied, pass None (not ["None"]) to Config so
    # Config.check_submitter returns True for every submitter (the documented
    # "match all" default). The previous str(None).split(",") produced ["None"]
    # which silently filtered out every real submitter, leaving the loader loop
    # empty — a pre-existing bug surfaced by the Phase-3 Definition-of-Done test.
    #
    # Per WR-06 + WR-07 (review 2026-06-10): strip whitespace from each CSV
    # token and drop empty-after-strip entries. `--submitters "Acme, BetaCo"`
    # previously produced `["Acme", " BetaCo"]` and the leading space silently
    # filtered out BetaCo. Likewise `--submitters ""` and `--submitters " "`
    # now route to None (match-all) consistently rather than producing `[""]`
    # which `Config.check_submitter` would reject for every submitter.
    if args.submitters:
        submitters = [s.strip() for s in args.submitters.split(",") if s.strip()]
        if not submitters:
            submitters = None
    else:
        submitters = None
    # Tree-wide Config: the current rules edition. The pre-loop structural
    # checks and the exporter use it; each submission in the loop gets its own
    # (config_for_submission) for the edition its manifest declares.
    config = Config(
        submitters=submitters,
        skip_output_file=args.skip_output_file,
    )

    loader = Loader(args.input, config)
    exporter = ResultExporter(args.csv, config)


    results = {}
    systems = {}
    errors = []

    # Per PLAN.md 01-03 D-02: run structural hierarchy checks ONCE before the
    # per-benchmark loader loop. Failures are accumulated into `errors` but do
    # NOT short-circuit the loop — every benchmark still gets its own checks.
    structure_check = SubmissionStructureCheck(log, config, args.input)
    if not structure_check():
        errors.append(args.input)

    # Per Phase 2 D-A1: schema-validate every systems/<name>.yaml ONCE before
    # the per-benchmark loader loop. Runs after SubmissionStructureCheck and
    # before the for-loop so schema errors surface once per YAML (not once per
    # workload). Failures accumulated into `errors` but do NOT abort the loop.
    schema_check = SystemYamlSchemaCheck(log, config, args.input)
    if not schema_check():
        errors.append(args.input)

    # Per Phase 8 D-82: CHECK-01..04 run as pre-loop checks, same pattern as
    # SubmissionStructureCheck and SystemYamlSchemaCheck. Verifies v1.1 pool
    # layout: pointer resolution, self-consistency, orphan detection, legacy
    # detection. Failures accumulated into errors but do NOT abort the loop.
    pool_check = PoolStructureCheck(log, config, args.input)
    if not pool_check():
        errors.append(args.input)

    # PROV-01/02: per-leaf provenance.json stamps and per-org submission.yaml
    # manifests (results-dir hygiene PR5). Silent on trees that predate them,
    # so the frozen v3.0 tree validates unchanged.
    provenance_check = ProvenanceCheck(log, config, args.input)
    if not provenance_check():
        errors.append(args.input)

    # EDN-01/02/03: declared rules editions and comparability classes against
    # mlpstorage_py/rules/editions.yaml. Silent on derived (pre-stamp) leaves.
    edition_check = EditionCheck(log, config, args.input)
    if not edition_check():
        errors.append(args.input)

    # Main loop over all the submissions
    sub_configs = {}
    for logs in loader.load():
        md = logs.loader_metadata
        sub_config = config_for_submission(args.input, md.division, md.submitter, config, sub_configs)
        if sub_config is None:
            # EditionCheck already reported EDN-04 for this manifest once.
            log.debug("skipping workload checks for %s/%s: its submission.yaml declares a "
                      "rules edition this tool cannot check", md.division, md.submitter)
            errors.append(md.folder)
            continue
        mode = getattr(md, "mode", None)
        checkers = MODE_TO_CHECKERS.get(mode, None)
        # Per CR-01 iter-2 (review 2026-06-10): an unmapped mode is a §2.1.10
        # workloadCategories violation, NOT a silent pass. Pre-CR-02 every
        # submission ran DirectoryCheck + per-mode classes; an unknown mode
        # under results/<sys>/ would have tripped DirectoryCheck's structural
        # checks. With the mode-routed dict, an empty checker list would skip
        # all validation and call exporter.add_result(logs) — recording an
        # unvalidated submission as a clean pass. Surface as an ERROR with the
        # locked [<id> <name>] prefix so the DoD test's error harness catches
        # it, accumulate into `errors` (don't abort), and continue.
        if checkers is None:
            log.error(
                "[2.1.10 workloadCategories] %s: unrecognized mode directory "
                "%r (expected one of %s)",
                logs.loader_metadata.folder, mode,
                sorted(MODE_TO_CHECKERS.keys()),
            )
            errors.append(logs.loader_metadata.folder)
            continue
        valid = True
        for checker in checkers:
            valid &= checker(log, sub_config, logs)()

        # TODO: Add results to summary
        if valid:
            exporter.add_result(logs)
        else:
            errors.append(logs.loader_metadata.folder)
    
    # Export results
    exporter.export()

    if len(errors) > 0:
        log.error("SUMMARY: submission has errors")
        return 1
    else:
        log.info("SUMMARY: submission looks OK")
        return 0


def main():
    """Standalone entry point: parse argv, then delegate to ``run``."""
    return run(get_args())


if __name__ == "__main__":
    sys.exit(main())