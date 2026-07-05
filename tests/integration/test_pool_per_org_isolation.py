"""Integration coverage for SC-5 (per-org pool isolation) — Phase 6 Plan 06-04 Task 5.

Exercises the ROADMAP SC-5 assertion end-to-end against the already-landed
`capture_or_verify_code_image` rewrite from Plan 06-02:

* Two different orgs sharing the same `--results-dir` maintain SEPARATE pool
  sets — `<rd>/Acme/code-*/` and `<rd>/Beta/code-*/` are independent trees
  (POOL-01, LAY-03).
* Even with identical source contents (identical live hashes), each org's
  pool root contains its own copy — no cross-org pool sharing.
* Source-tree changes propagate independently per org — Beta's second capture
  after a source mutation writes a NEW pool image under `<rd>/Beta/`, without
  disturbing Acme's pool.

Scope note (per SC#7): tests call `capture_or_verify_code_image` DIRECTLY
(integration-scope: real files, real hashing).

Refs: 06-04-PLAN.md Task 5, 06-CONTEXT.md POOL-01, LAY-03.
"""

from __future__ import annotations

import json
from pathlib import Path

from mlpstorage_py.submission_checker.tools.code_image import (
    capture_or_verify_code_image,
)

from tests.integration.conftest import pool_dirs


class TestPerOrgIsolation:
    """SC-5: two orgs sharing a results-dir maintain separate pool sets."""

    def test_two_orgs_maintain_separate_pool_sets(
        self, tmp_path, fake_source_root, capture_args_factory, log
    ):
        """SC-5 primary: identical source hashed twice under two orgs → each
        org has its own single pool dir; both dirs share the same hash suffix
        (identical source), but they live under distinct org roots."""
        rd = tmp_path / "results"
        rd.mkdir()

        args_acme = capture_args_factory(
            results_dir=rd, mode="closed", orgname="Acme",
            benchmark="training", command="run", model="unet3d",
        )
        pool_acme = capture_or_verify_code_image(args_acme, {}, log)

        args_beta = capture_args_factory(
            results_dir=rd, mode="closed", orgname="Beta",
            benchmark="training", command="run", model="unet3d",
        )
        pool_beta = capture_or_verify_code_image(args_beta, {}, log)

        # Each org has exactly one pool dir under its own root.
        acme_pools = pool_dirs(rd / "Acme")
        beta_pools = pool_dirs(rd / "Beta")
        assert len(acme_pools) == 1, acme_pools
        assert len(beta_pools) == 1, beta_pools

        # Pool paths live under different org roots.
        assert Path(pool_acme).parent == rd / "Acme"
        assert Path(pool_beta).parent == rd / "Beta"
        assert Path(pool_acme) != Path(pool_beta)

        # Identical source ⇒ identical hash suffix in the dir names.
        assert acme_pools[0].name == beta_pools[0].name, (
            f"expected same hash suffix (identical source), got "
            f"{acme_pools[0].name} vs {beta_pools[0].name}"
        )

        # And the sidecar hash values match.
        h_acme = json.loads((acme_pools[0] / ".code-hash.json").read_text())["hash"]
        h_beta = json.loads((beta_pools[0] / ".code-hash.json").read_text())["hash"]
        assert h_acme == h_beta

    def test_two_orgs_different_source_hashes_maintain_separate_pool_sets(
        self, tmp_path, fake_source_root, capture_args_factory, log
    ):
        """SC-5 companion: mutate the source tree BETWEEN the two org captures
        — each org ends up with its own single pool dir carrying a distinct
        hash (per-org isolation survives source drift)."""
        rd = tmp_path / "results"
        rd.mkdir()

        args_acme = capture_args_factory(
            results_dir=rd, mode="closed", orgname="Acme",
            benchmark="training", command="run", model="unet3d",
        )
        capture_or_verify_code_image(args_acme, {}, log)

        # Mutate the source tree so Beta's capture hashes differently.
        (fake_source_root / "mlpstorage_py" / "beta_marker.py").write_text(
            "# marker present only during Beta's capture\n"
        )

        args_beta = capture_args_factory(
            results_dir=rd, mode="closed", orgname="Beta",
            benchmark="training", command="run", model="unet3d",
        )
        capture_or_verify_code_image(args_beta, {}, log)

        acme_pools = pool_dirs(rd / "Acme")
        beta_pools = pool_dirs(rd / "Beta")
        assert len(acme_pools) == 1, acme_pools
        assert len(beta_pools) == 1, beta_pools

        # Distinct hashes — the mutation propagated to Beta's pool only.
        h_acme = json.loads((acme_pools[0] / ".code-hash.json").read_text())["hash"]
        h_beta = json.loads((beta_pools[0] / ".code-hash.json").read_text())["hash"]
        assert h_acme != h_beta, (
            f"SC-5 companion violated: expected distinct hashes across orgs "
            f"after source drift, both were {h_acme!r}"
        )
