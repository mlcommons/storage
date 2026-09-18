"""Integration coverage for the tree-wide pool shared across orgs.

Historically (Phase 6 SC-5) two orgs sharing one ``--results-dir`` kept
SEPARATE pools at ``<rd>/Acme/code-*/`` and ``<rd>/Beta/code-*/``. The
results-dir hygiene effort (PR4) replaced that with ONE pool at
``<rd>/code-images/``: identical source captured under two orgs yields a
single image, and source drift between the captures yields two images side
by side in the same pool. No per-org root is created.

Scope note (per SC#7): tests call `capture_or_verify_code_image` DIRECTLY
(integration-scope: real files, real hashing).
"""

from __future__ import annotations

import json
from pathlib import Path

from mlpstorage_py.submission_checker.tools.code_image import (
    capture_or_verify_code_image,
)
from tests.integration.conftest import pool_dirs


class TestCrossOrgPoolSharing:
    """Two orgs sharing a results-dir share one code-image pool."""

    def test_two_orgs_identical_source_share_one_image(
        self, tmp_path, fake_source_root, capture_args_factory, log
    ):
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

        assert Path(pool_acme) == Path(pool_beta)
        assert Path(pool_acme).parent == rd / "code-images"
        assert pool_dirs(rd / "code-images") == [Path(pool_acme)]
        assert not (rd / "Acme").exists()
        assert not (rd / "Beta").exists()

    def test_two_orgs_different_source_hashes_land_side_by_side(
        self, tmp_path, fake_source_root, capture_args_factory, log
    ):
        rd = tmp_path / "results"
        rd.mkdir()

        args_acme = capture_args_factory(
            results_dir=rd, mode="closed", orgname="Acme",
            benchmark="training", command="run", model="unet3d",
        )
        pool_acme = capture_or_verify_code_image(args_acme, {}, log)

        # Mutate the source tree so Beta's capture hashes differently.
        (fake_source_root / "mlpstorage_py" / "beta_marker.py").write_text(
            "# marker present only during Beta's capture\n"
        )

        args_beta = capture_args_factory(
            results_dir=rd, mode="closed", orgname="Beta",
            benchmark="training", command="run", model="unet3d",
        )
        pool_beta = capture_or_verify_code_image(args_beta, {}, log)

        pools = pool_dirs(rd / "code-images")
        assert sorted(pools) == sorted([Path(pool_acme), Path(pool_beta)])
        h_acme = json.loads((Path(pool_acme) / ".code-hash.json").read_text())["hash"]
        h_beta = json.loads((Path(pool_beta) / ".code-hash.json").read_text())["hash"]
        assert h_acme != h_beta
