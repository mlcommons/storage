"""Integration coverage for SC-3 (source-change captures new image) and UX-01
negative-grep on the retired reject strings — Phase 6 Plan 06-04 Task 3.

<!-- planner-discipline-allow: changes to the codebase are not allowed -->
<!-- planner-discipline-allow: all runs of this type must use the same codebase -->

Exercises the ROADMAP SC-3 assertion end-to-end against the already-landed
`capture_or_verify_code_image` rewrite from Plan 06-02:

* A source-tree change between two invocations captures a NEW pool image
  (CAPVER-02) — the org root now contains two `code-<hash8>/` directories with
  distinct hashes.
* The retired Phase-5 reject strings (`"changes to the codebase are not
  allowed"` and `"all runs of this type must use the same codebase"`) do NOT
  appear anywhere in the captured logger output on the source-change path
  (UX-01: source-change is a SUCCESS path in Phase 6, not a reject path).
* No exception is raised — `capture_or_verify_code_image` returns cleanly.

The two `planner-discipline-allow` markers above scope the literal reject
strings against the negative-grep gate — those exact strings appear inside
this file's assertion literals below.

Scope note (per SC#7): tests call `capture_or_verify_code_image` DIRECTLY
(integration-scope: real files, real hashing).

Refs: 06-04-PLAN.md Task 3, 06-CONTEXT.md CAPVER-02, CAPVER-03, UX-01.
"""

from __future__ import annotations

import json
from pathlib import Path

from mlpstorage_py.submission_checker.tools.code_image import (
    CodeImageError,
    capture_or_verify_code_image,
)

from tests.integration.conftest import pool_dirs


# planner-discipline-allow: changes to the codebase are not allowed
_RETIRED_CLOSED_REJECT = "changes to the codebase are not allowed"
# planner-discipline-allow: all runs of this type must use the same codebase
_RETIRED_OPEN_REJECT = "all runs of this type must use the same codebase"


class TestSourceChangeCapturesNewImage:
    """SC-3: source change between calls captures a new pool image."""

    def test_source_change_captures_second_pool_image(
        self, tmp_path, fake_source_root, capture_args_factory, log
    ):
        """SC-3 primary: mutate the fake source tree between calls — the second
        call captures a NEW `code-<hash8>/` (2 pool dirs total, distinct hashes).
        """
        rd = tmp_path / "results"
        rd.mkdir()

        args1 = capture_args_factory(
            results_dir=rd, mode="closed", orgname="Acme",
            benchmark="training", command="run", model="unet3d",
        )
        pool_dir_1 = capture_or_verify_code_image(args1, {}, log)

        # Add a marker file to the fake source tree — this changes the
        # md5-tree-v2 hash deterministically.
        (fake_source_root / "mlpstorage_py" / "phase_6_marker.py").write_text(
            "# marker for SC-3 source-change coverage\n"
        )

        args2 = capture_args_factory(
            results_dir=rd, mode="closed", orgname="Acme",
            benchmark="training", command="run", model="unet3d",
        )
        pool_dir_2 = capture_or_verify_code_image(args2, {}, log)

        # SC-3: two DISTINCT pool dirs exist under <rd>/Acme/.
        org_root = rd / "Acme"
        pools = pool_dirs(org_root)
        assert len(pools) == 2, (
            f"SC-3 violated: expected 2 pool dirs after source change, got {pools}"
        )
        assert Path(pool_dir_1) != Path(pool_dir_2)
        assert set(pools) == {Path(pool_dir_1), Path(pool_dir_2)}

        # Distinct hashes recorded in each sidecar.
        h1 = json.loads((Path(pool_dir_1) / ".code-hash.json").read_text())["hash"]
        h2 = json.loads((Path(pool_dir_2) / ".code-hash.json").read_text())["hash"]
        assert h1 != h2, f"expected distinct hashes, both were {h1!r}"

    def test_source_change_does_NOT_emit_retired_reject_string(
        self, tmp_path, fake_source_root, capture_args_factory, log
    ):
        """SC-3 negative + UX-01: source-change is a SUCCESS path in Phase 6,
        not a reject path. The two retired Phase-5 reject strings MUST NOT
        appear in any log level's captured messages."""
        rd = tmp_path / "results"
        rd.mkdir()

        args1 = capture_args_factory(
            results_dir=rd, mode="closed", orgname="Acme",
            benchmark="training", command="run", model="unet3d",
        )
        capture_or_verify_code_image(args1, {}, log)

        (fake_source_root / "mlpstorage_py" / "phase_6_marker.py").write_text(
            "# marker triggers source-change hash divergence\n"
        )

        args2 = capture_args_factory(
            results_dir=rd, mode="closed", orgname="Acme",
            benchmark="training", command="run", model="unet3d",
        )
        capture_or_verify_code_image(args2, {}, log)

        # Aggregate every captured message across every level.
        all_messages = (
            log.errors + log.statuses + log.warnings + log.infos + log.debugs
        )
        joined = "\n".join(all_messages)

        assert _RETIRED_CLOSED_REJECT not in joined, (
            f"UX-01 violated: retired CLOSED reject string emitted on "
            f"source-change success path — messages: {all_messages}"
        )
        assert _RETIRED_OPEN_REJECT not in joined, (
            f"UX-01 violated: retired OPEN reject string emitted on "
            f"source-change success path — messages: {all_messages}"
        )

    def test_source_change_second_call_returns_success(
        self, tmp_path, fake_source_root, capture_args_factory, log
    ):
        """SC-3 companion: the second call (after source change) does not raise
        `CodeImageError` — source change is a success path, not an error."""
        rd = tmp_path / "results"
        rd.mkdir()

        args1 = capture_args_factory(
            results_dir=rd, mode="closed", orgname="Acme",
            benchmark="training", command="run", model="unet3d",
        )
        capture_or_verify_code_image(args1, {}, log)

        (fake_source_root / "mlpstorage_py" / "phase_6_marker.py").write_text(
            "# marker\n"
        )

        args2 = capture_args_factory(
            results_dir=rd, mode="closed", orgname="Acme",
            benchmark="training", command="run", model="unet3d",
        )
        try:
            result = capture_or_verify_code_image(args2, {}, log)
        except CodeImageError as e:
            raise AssertionError(
                f"SC-3 violated: source-change second call raised "
                f"CodeImageError — {e!r}"
            )
        assert result is not None
        assert Path(result).is_dir()
