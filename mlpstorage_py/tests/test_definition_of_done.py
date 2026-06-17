"""Definition-of-Done end-to-end test (Phase 3 Plan 03-05).

Subprocess-invokes ``python -m mlpstorage_py.submission_checker`` against
synthetic submission fixtures built via the conftest ``build_submission``
factory, exercising the full validator pipeline end-to-end.

Closes ROADMAP Phase 3 success criteria #3 and #4 with the **Path-A
relaxation** approved at the 03-05 user checkpoint (2026-06-10):

* **#3 (good fixture, observation-style):** a vanilla
  ``build_submission(tmp_path)`` tree is invoked end-to-end and the rule-ID
  prefixed ERROR records emitted are *observed* (counted, logged into the
  assertion message). The strict assertion is narrowed from "zero ERROR
  records" to "none of the four bad-fixture-locked rule IDs (2.1.2, 3.3.4,
  4.6.4, 4.7.3) appear in the good fixture." Rationale: after the Phase 1
  and Phase 2 retrofits expanded the per-benchmark check surface, the
  vanilla fixture trips ~30 pre-existing structural ERROR records unrelated
  to the four engineered bad-fixture violations. The original "exit 0 / no
  ERROR records" lock cannot be satisfied until the fixture is upgraded.
  Cleaning the vanilla fixture and re-tightening this assertion is tracked
  as a deferred item for Phase 4 (see 03-05-SUMMARY.md → Known Stubs).
* **#4 (bad fixture, subset-style):** a tree with four locked violations
  (one per check class + one schema check) exits 1 and emits *at least*
  the rule_id set ``{2.1.2, 3.3.4, 4.6.4, 4.7.3}`` at ERROR level. The
  assertion is relaxed from set equality to subset for the same reason:
  the vanilla fixture's pre-existing noise surfaces alongside the
  engineered violations. The bad-fixture origin-context binding test
  (Gemini suggestion #3) remains a hard exact-match assertion.

The origin-context binding assertion is **non-negotiable** — each
bad-fixture rule_id binds to the path attribute that proves it fired from
the expected check class: 2.1.2 → submission-root / stray_dir; 3.3.4 →
unet3d training path; 4.6.4 → llama3-70b checkpointing path; 4.7.3 →
schema yaml path.

Run with::

    pytest mlpstorage_py/tests/test_definition_of_done.py -v -m integration

Skipped by default in the fast unit-test suite::

    pytest mlpstorage_py/tests -v -m "not integration"
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from mlpstorage_py.tests.conftest import build_submission


# ---------------------------------------------------------------------------
# Locked origin-context substring map (Gemini suggestion #3)
# ---------------------------------------------------------------------------
#
# For each bad-fixture rule_id, the validator's ``log_violation`` call carries
# a ``path`` argument that names the origin context of the violation:
#
# * 2.1.2 ← SubmissionStructureCheck.top_level_subdirectories_check
#           path = os.path.join(root_path, "stray_dir")  (pre-loader)
# * 3.3.4 ← TrainingCheck.single_host_client_limit
#           path = self.path = loader_metadata.folder = .../results/<sys>/training/unet3d
# * 4.6.4 ← CheckpointingCheck.open_mpi_processes
#           path = self.path = .../results/<sys>/checkpointing/llama3-70b
# * 4.7.3 ← SystemYamlSchemaCheck.schema_validate_all_system_yamls
#           path = yaml_path = .../closed/Acme/systems/acme-storage-v1.yaml
#
# A failure of this assertion proves the loader bound a per-benchmark log to
# the wrong Check class, or a pre-loader check moved into the loader loop.
_EXPECTED_PATH_SUBSTRINGS = {
    "2.1.2": ["stray_dir"],
    "3.3.4": ["unet3d", "training"],
    "4.6.4": ["llama3-70b", "checkpointing"],
    "4.7.3": ["acme-storage-v1.yaml"],
}

_EXPECTED_BAD_FIXTURE_RULE_IDS = {"2.1.2", "3.3.4", "4.6.4", "4.7.3"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_validator(root):
    """Invoke the validator CLI as a subprocess and return the CompletedProcess.

    Matches success criteria #3/#4 verbatim — uses ``-m
    mlpstorage_py.submission_checker``, ``--input <root>``, ``--version v2.0``.
    """
    return subprocess.run(
        [
            sys.executable,
            "-m", "mlpstorage_py.submission_checker",
            "--input", str(root),
            "--version", "v2.0",
        ],
        capture_output=True,
        text=True,
        # Per WR-03 (review 2026-06-10): a regression that hangs the
        # validator (infinite loop in a future check, yaml-load deadlock,
        # etc.) would hang the DoD test indefinitely and stall CI. 60s is
        # generous for fixture-tree validation.
        timeout=60,
    )


def _combined_output(result):
    """Return stdout + stderr joined for logline scanning."""
    return (result.stdout or "") + "\n" + (result.stderr or "")


def _extract_error_rule_ids(combined: str) -> set[str]:
    """Extract rule IDs from ERROR-level log lines carrying a ``[X.Y.Z ...]`` prefix.

    Filters by both the log-level marker (``"ERROR]"`` — the locked
    ``%(levelname)s`` slot in main.py's logging.basicConfig format) and the
    rule-ID bracket prefix (``[X.Y.Z `` from BaseCheck.log_violation).
    """
    ids: set[str] = set()
    for line in combined.splitlines():
        if "ERROR]" not in line:
            continue
        m = re.search(r"\[([234]\.\d+\.\d+) ", line)
        if m:
            ids.add(m.group(1))
    return ids


def _extract_error_rule_id_path_pairs(combined: str) -> list[tuple[str, str]]:
    """Extract ``(rule_id, path)`` pairs from ERROR-level rule-prefixed lines.

    Matches the locked ``BaseCheck.log_violation`` format
    ``[<rule_id> <rule_name>] <path>: <msg>`` per checks/base.py:36.

    Per WR-04 (review 2026-06-10): pin the path capture to the **first**
    ``: `` (colon-space) combination rather than ``[^:]+``. The locked
    BaseCheck format always emits ``<path>: <msg>`` with a single space
    after the colon, so a non-greedy match up to the first ``: `` is
    unambiguous AND tolerates bare ``:`` characters inside the path
    (e.g., Windows drive letter, or a future fixture-generated dirname
    containing a colon).
    """
    pairs: list[tuple[str, str]] = []
    pattern = re.compile(r"\[([234]\.\d+\.\d+) [a-zA-Z][a-zA-Z0-9]*\] (.+?): ")
    for line in combined.splitlines():
        if "ERROR]" not in line:
            continue
        m = pattern.search(line)
        if m:
            pairs.append((m.group(1), m.group(2)))
    return pairs


# ---------------------------------------------------------------------------
# Good-fixture test class (D-E4 / ROADMAP success criterion #3)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestDefinitionOfDoneGood:
    """ROADMAP #3 (Path-A relaxed): a vanilla submission tree does not trip
    any of the four bad-fixture-locked rule IDs.

    The original lock — "exit 0 with no ERROR records" — is not satisfiable
    against the current vanilla ``build_submission`` factory because Phase
    1 and Phase 2 retrofits expanded the per-benchmark check surface, and
    the fixture's vanilla defaults trip ~30 pre-existing structural ERROR
    records unrelated to the four engineered bad-fixture violations.

    Path-A (approved at the 03-05 user checkpoint, 2026-06-10) narrows the
    assertion to "none of the four bad-fixture-locked rule IDs (2.1.2,
    3.3.4, 4.6.4, 4.7.3) appear in the good fixture." This preserves the
    core invariant that the bad fixture trips violations the good fixture
    does not, while deferring fixture cleanup to Phase 4. The vanilla
    fixture's observed ERROR count is recorded in the assertion message
    for diagnostic value.
    """

    def test_good_fixture_does_not_trip_bad_fixture_rule_ids(self, tmp_path):
        """A vanilla ``build_submission`` tree may emit pre-existing
        structural ERROR records, but it MUST NOT trip any of the four
        rule IDs the bad fixture is engineered to trigger.

        Path-A relaxation rationale: see class docstring. Tracked for
        re-tightening in Phase 4 (see 03-05-SUMMARY.md → Known Stubs).
        """
        root = build_submission(tmp_path)

        result = _run_validator(root)
        combined = _combined_output(result)

        observed_rule_ids = _extract_error_rule_ids(combined)

        forbidden = observed_rule_ids & _EXPECTED_BAD_FIXTURE_RULE_IDS
        assert not forbidden, (
            f"Good fixture tripped bad-fixture-locked rule IDs (expected none): "
            f"{sorted(forbidden)}.\n"
            f"All observed rule IDs in good fixture: {sorted(observed_rule_ids)} "
            f"(noise count: {len(observed_rule_ids)} — tracked for Phase 4 fixture "
            f"cleanup; see 03-05-SUMMARY.md → Known Stubs).\n"
            f"Exit code: {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}\n"
        )


# ---------------------------------------------------------------------------
# Bad-fixture test class (D-E5 / D-E6 / ROADMAP success criterion #4)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestDefinitionOfDoneBad:
    """ROADMAP #4 (Path-A relaxed): a bad submission tree exits 1 and emits
    at least the four locked rule IDs as ERROR records, each from its
    expected origin context.

    The original lock asserted set equality (``observed == expected``); the
    Path-A relaxation (approved 2026-06-10) narrows it to subset
    (``expected ⊆ observed``) for the same fixture-noise reason documented
    on the good-fixture class. The origin-context binding test below
    remains a hard exact-match assertion: each of the four locked rule IDs
    must bind to its expected origin path substring.
    """

    @pytest.fixture(scope="class")
    def bad_fixture_result(self, tmp_path_factory):
        """Build the bad fixture once and share the subprocess result across
        every test in this class (saves wall time + avoids divergence)."""
        root = tmp_path_factory.mktemp("dod_bad")
        root = build_submission(
            root,
            # 2.1.2 topLevelSubdirectories — stray top-level dir.
            # W-01 lock: NOT top_level_capitalcase=True (which would prevent
            # SystemYamlSchemaCheck from walking closed/Acme/systems/ and
            # silently mask the 4.7.3 violation).  See plan <interfaces>.
            extra_top_level="stray_dir",
            # 3.3.4 trainingSingleHostClientLimit — single-host run with
            # multiple client hosts declared in args.hosts.
            run_metadata_hosts=["host1", "host2"],
            # 4.6.4 checkpointOpenSubmissionScaling — OPEN llama3-70b with
            # num_processes (10) not a multiple of TP*PP (= 8).
            chkpt_open_num_processes=10,
            chkpt_model="llama3-70b",
            # 4.7.3 checkpointRemappingTimeReporting — Pydantic Rule-13
            # violation: simultaneous flags both True + non-zero remap time.
            system_yaml_rule13_violation=True,
        )
        # Path-A workaround: post-`build_submission` patch to force
        # ``summary.num_hosts = 1`` in every training run timestamp's
        # summary.json. The conftest default is `num_hosts=2`, but rule 3.3.4
        # (`trainingSingleHostClientLimit`) only fires when
        # `summary.num_hosts == 1 AND len(args.hosts) > 1`. The plan's
        # <interfaces> block assumed the rule read from per-timestamp
        # metadata, but training_checks.py:355 actually reads
        # `summary.get("num_hosts", 1)`. No conftest kwarg exists to
        # selectively override this to 1 (the existing kwargs only set 2 or 3).
        # The user directive (2026-06-10 03-05 checkpoint) forbids conftest
        # modifications. Per-test post-fixture patching is the bounded
        # workaround that keeps the origin-context binding test sound
        # (non-negotiable per the same directive). Tracked for cleanup in
        # Phase 4 alongside the broader fixture-noise stub (see
        # 03-05-SUMMARY.md → Known Stubs).
        for summary_path in Path(root).rglob("training/unet3d/run/*/summary.json"):
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            data["num_hosts"] = 1
            summary_path.write_text(json.dumps(data), encoding="utf-8")
        return _run_validator(root), root

    def test_bad_fixture_exits_1_with_locked_rule_ids_present(self, bad_fixture_result):
        """Bad fixture exits 1 and emits AT LEAST the four locked rule IDs
        as rule-ID-prefixed ERROR records (Path-A subset assertion).

        Path-A relaxation: switched from exact set equality to subset
        because pre-existing vanilla-fixture noise surfaces alongside the
        engineered violations. See class docstring.
        """
        result, _root = bad_fixture_result
        combined = _combined_output(result)

        assert result.returncode == 1, (
            f"Expected exit 1 for bad fixture, got {result.returncode}.\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}\n"
        )

        extracted = _extract_error_rule_ids(combined)
        missing = _EXPECTED_BAD_FIXTURE_RULE_IDS - extracted
        assert _EXPECTED_BAD_FIXTURE_RULE_IDS <= extracted, (
            f"Bad fixture missing one or more locked rule IDs.\n"
            f"  expected (subset): {sorted(_EXPECTED_BAD_FIXTURE_RULE_IDS)}\n"
            f"  observed:          {sorted(extracted)}\n"
            f"  missing:           {sorted(missing)}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}\n"
        )

    def test_bad_fixture_rule_ids_bind_to_expected_origin_contexts(self, bad_fixture_result):
        """Each bad-fixture rule_id binds to the path attribute that proves
        it fired from the expected origin context (Gemini suggestion #3).

        This assertion is NOT relaxed — it remains a hard exact-match check
        because it proves Phase-1/2 wiring correctness (loader binds
        per-benchmark logs to the correct Check class; pre-loader checks
        fire in the pre-loader phase, not from inside the loader loop).
        Failure of this assertion is a wiring regression that must be
        fixed, not a fixture-noise artifact.
        """
        result, _root = bad_fixture_result
        combined = _combined_output(result)

        pairs = _extract_error_rule_id_path_pairs(combined)
        paths_by_rule: dict[str, list[str]] = {}
        for rid, path in pairs:
            paths_by_rule.setdefault(rid, []).append(path)

        for rid, expected_substrings in _EXPECTED_PATH_SUBSTRINGS.items():
            observed = paths_by_rule.get(rid, [])
            matching = [
                p for p in observed
                if all(sub in p for sub in expected_substrings)
            ]
            assert matching, (
                f"rule {rid}: no observed path contains all expected "
                f"substrings {expected_substrings!r}. observed paths for "
                f"{rid}: {observed!r}. This means the loader bound {rid} "
                f"to the wrong origin context (per-benchmark vs pre-loader) "
                f"or the path field was empty.\n"
                f"--- stdout ---\n{result.stdout}\n"
                f"--- stderr ---\n{result.stderr}\n"
            )

    def test_bad_fixture_struct_violation_fires_before_loader_loop(self, bad_fixture_result):
        """D-E6 accumulate-don't-abort across the pre-loader / in-loop
        boundary: the 2.1.2 topLevelSubdirectories ERROR record appears in
        the output BEFORE the first per-benchmark ``Starting`` log line.

        Proves the pre-loader STRUCT pass does not short-circuit the
        loader loop — both fire in one run.
        """
        result, _root = bad_fixture_result
        combined = _combined_output(result)
        lines = combined.splitlines()

        struct_212_idx = None
        first_benchmark_start_idx = None
        # Sentinels for "the loader loop ran": any record from a rule
        # in §3 (training) or §4 (checkpointing) — those checks only fire
        # from inside the per-benchmark loop. We accept ERROR or INFO so
        # the sentinel is robust to the bad fixture's specific failures.
        IN_LOOP_RULE_PREFIXES = ("[3.", "[4.")
        for idx, line in enumerate(lines):
            if struct_212_idx is None and "ERROR]" in line and "[2.1.2 " in line:
                struct_212_idx = idx
            if first_benchmark_start_idx is None and any(
                prefix in line for prefix in IN_LOOP_RULE_PREFIXES
            ):
                first_benchmark_start_idx = idx

        assert struct_212_idx is not None, (
            "No [2.1.2 ...] ERROR record found in output.\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}\n"
        )
        assert first_benchmark_start_idx is not None, (
            "No in-loop rule record found (looked for any [3.* / [4.* "
            "prefix) — loader loop did not run.\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}\n"
        )
        assert struct_212_idx < first_benchmark_start_idx, (
            f"[2.1.2 ...] ERROR (line {struct_212_idx}) fired AFTER the "
            f"first per-benchmark 'Starting' line (line "
            f"{first_benchmark_start_idx}). Pre-loader STRUCT pass must "
            f"run before the loader loop (D-E6).\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}\n"
        )
