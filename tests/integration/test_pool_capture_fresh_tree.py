"""Integration coverage for SC-1 (fresh-tree pool + pointer) and SC#11
(D-63 legacy layout refuse) — Phase 6 Plan 06-04 Task 1.

Exercises the ROADMAP SC-1 assertion in an end-to-end integration scope
against the already-landed `capture_or_verify_code_image` rewrite from
Plan 06-02:

* A fresh (empty) `--results-dir` invocation writes exactly one pool image at
  `<rd>/<orgname>/code-<hash8>/` (mode-agnostic per D-64) with a valid
  `.code-hash.json` sidecar (POOL-01, POOL-02).
* The run leaf receives an atomic `.mlps-code-image` pointer whose content is
  `md5-tree-v2:<full-32-hex>` (PTR-01, D-61).

Also covers SC#11 (D-63 refuse) — see plan `<must_haves> SC#11`:
a pre-existing legacy `<rd>/{closed,open}/<org>/code/` layout is refused with
`LegacyLayoutDetected` before any pool write is attempted.

Scope note (per SC#7): tests call `capture_or_verify_code_image` DIRECTLY
(integration-scope: real files, real hashing) rather than driving through
`main._main_impl`. This keeps runtime under 60s per file and avoids the
DLIO/MPI dependency chain, while still exercising every capture-layer
invariant end-to-end.

Refs: 06-04-PLAN.md Task 1, 06-CONTEXT.md D-60..D-67, POOL-01..04, PTR-01.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlpstorage_py.submission_checker.tools.code_image import (
    LegacyLayoutDetected,
    capture_or_verify_code_image,
)


class TestFreshTreePoolCapture:
    """SC-1: fresh `--results-dir` produces pool image + pointer."""

    def test_fresh_tree_writes_pool_and_pointer(
        self, tmp_path, fake_source_root, capture_args_factory, log
    ):
        """SC-1 primary: exactly one `code-<hash8>/` at `<rd>/<org>/`, with a
        sidecar `.code-hash.json`, plus a `.mlps-code-image` pointer at the
        run leaf whose content is `md5-tree-v2:<full-hash>`."""
        rd = tmp_path / "results"
        rd.mkdir()

        args = capture_args_factory(
            results_dir=rd, mode="closed", orgname="Acme",
            benchmark="training", command="run", model="unet3d",
        )

        pool_dir = capture_or_verify_code_image(args, {}, log)

        # 1. Pool image exists at <rd>/Acme/code-<hash8>/ (POOL-01, D-64 mode-agnostic).
        assert pool_dir is not None
        assert Path(pool_dir).is_dir()
        assert Path(pool_dir).parent == rd / "Acme", (
            f"expected pool under <rd>/Acme/, got parent {Path(pool_dir).parent}"
        )

        # Exactly ONE pool dir after a single fresh-tree capture.
        org_root = rd / "Acme"
        pool_glob = sorted(org_root.glob("code-*"))
        assert len(pool_glob) == 1, pool_glob
        assert pool_glob[0] == Path(pool_dir)

        # 2. Sidecar .code-hash.json is present and valid JSON.
        sidecar = Path(pool_dir) / ".code-hash.json"
        assert sidecar.is_file(), f"expected .code-hash.json at {sidecar}"
        data = json.loads(sidecar.read_text())

        # 3. POOL-02: dir name is `code-<first-8-of-full-hash>`.
        assert Path(pool_dir).name == f"code-{data['hash'][:8]}", (
            f"pool name {Path(pool_dir).name!r} != code-{data['hash'][:8]}"
        )

        # 4. PTR-01 + D-61: pointer file exists inside the run leaf with content
        # `md5-tree-v2:<full-32-hex>`. Locate the pointer via rglob because
        # the leaf datetime is generated at call time.
        pointers = list(rd.rglob(".mlps-code-image"))
        assert len(pointers) == 1, pointers
        pointer = pointers[0]
        assert pointer.read_text().strip() == f"md5-tree-v2:{data['hash']}", (
            f"pointer content {pointer.read_text()!r} != md5-tree-v2:{data['hash']}"
        )

        # 5. Pointer lives inside a run leaf under `<rd>/closed/Acme/results/.../`
        # (Rules.md §2.1 canonical shape).
        assert "closed" in pointer.parts and "Acme" in pointer.parts, pointer

    def test_fresh_tree_hash_json_schema_fields_all_present(
        self, tmp_path, fake_source_root, capture_args_factory, log
    ):
        """POOL-02 + D-07 schema: sidecar has hash, algorithm, captured_at,
        mlpstorage_version, git_sha fields (git_sha may be None)."""
        rd = tmp_path / "results"
        rd.mkdir()
        args = capture_args_factory(
            results_dir=rd, mode="closed", orgname="Acme",
            benchmark="training", command="run", model="unet3d",
        )
        pool_dir = capture_or_verify_code_image(args, {}, log)

        data = json.loads((Path(pool_dir) / ".code-hash.json").read_text())
        for field in ("hash", "algorithm", "captured_at", "mlpstorage_version", "git_sha"):
            assert field in data, f"missing schema field {field!r} in .code-hash.json"
        assert data["algorithm"] == "md5-tree-v2"
        # hash: 32 lowercase hex
        assert len(data["hash"]) == 32
        assert all(c in "0123456789abcdef" for c in data["hash"])


class TestFreshTreeLegacyRefuse:
    """SC#11 / D-63: legacy `code/` layout is refused before any pool write.

    Precise contract: `capture_or_verify_code_image` scans
    `<results_dir>/{closed,open}/<orgname>/code/` — if either is a directory,
    `LegacyLayoutDetected` is raised BEFORE any pool write is attempted.
    Phase 7 owns the migration; Phase 6 refuses. (`_scan_legacy_layout` in
    `mlpstorage_py/submission_checker/tools/code_image.py:665`.)
    """

    def test_fresh_tree_refuses_when_closed_legacy_code_present(
        self, tmp_path, fake_source_root, capture_args_factory, log
    ):
        """D-63: pre-existing `<rd>/closed/Acme/code/` triggers refuse with the
        canonical error substring `Legacy code-image layout detected`."""
        rd = tmp_path / "results"
        rd.mkdir()

        # Pre-create a legacy CLOSED code/ subtree — this is the D-63 trigger.
        legacy = rd / "closed" / "Acme" / "code"
        legacy.mkdir(parents=True)
        # Include a dummy .code-hash.json so the tree looks realistic (not
        # required for D-63 to fire — the mere presence of code/ triggers it).
        (legacy / ".code-hash.json").write_text('{"hash": "0" * 32}')

        args = capture_args_factory(
            results_dir=rd, mode="closed", orgname="Acme",
            benchmark="training", command="run", model="unet3d",
        )

        with pytest.raises(LegacyLayoutDetected) as exc_info:
            capture_or_verify_code_image(args, {}, log)

        # Error message contains the D-63 canonical substring.
        assert "Legacy code-image layout detected" in str(exc_info.value), (
            f"expected D-63 substring in error message; got {exc_info.value!r}"
        )

        # No pool image was written under <rd>/Acme/ — the refuse is BEFORE
        # any writes per D-63 contract.
        org_root = rd / "Acme"
        pool_glob = list(org_root.glob("code-*")) if org_root.is_dir() else []
        assert pool_glob == [], (
            f"D-63 violated: pool image written despite legacy layout — {pool_glob}"
        )

    def test_fresh_tree_refuses_when_open_legacy_code_present(
        self, tmp_path, fake_source_root, capture_args_factory, log
    ):
        """D-63 symmetric: pre-existing `<rd>/open/Acme/code/` also triggers refuse.

        Even when the current invocation is a CLOSED-mode command, an OPEN
        legacy layout is refused — `_scan_legacy_layout` walks BOTH mode
        subtrees regardless of the current args.mode (see code_image.py:693).
        """
        rd = tmp_path / "results"
        rd.mkdir()

        legacy = rd / "open" / "Acme" / "code"
        legacy.mkdir(parents=True)

        args = capture_args_factory(
            results_dir=rd, mode="closed", orgname="Acme",
            benchmark="training", command="run", model="unet3d",
        )

        with pytest.raises(LegacyLayoutDetected) as exc_info:
            capture_or_verify_code_image(args, {}, log)

        assert "Legacy code-image layout detected" in str(exc_info.value)
        # No pool image written.
        org_root = rd / "Acme"
        pool_glob = list(org_root.glob("code-*")) if org_root.is_dir() else []
        assert pool_glob == []
