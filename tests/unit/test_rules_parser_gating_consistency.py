"""
Regression test: the CLI parser must honor the gating that the rules-checker tables
and Rules.md specify.

Background — issue #433 surfaced a recurring class of bug where the parser
disagrees with the rules-checker on what is allowed in CLOSED submissions
(e.g. closed training mode rejecting --params even though
TrainingRunRulesChecker.CLOSED_ALLOWED_PARAMS lists dataset.num_files_train).

The audit done alongside #433 lives in:
    .planning/issue-433-gating-audit.md

This test mechanizes the cross-check so future drift fails in CI. It is
deliberately narrow: we test only what the rules-checker tables and Rules.md
explicitly assert, not what the parser "ought to" do in absence of guidance.
"""

from unittest.mock import patch

import pytest

from mlpstorage_py.cli_parser import parse_arguments
from mlpstorage_py.rules.run_checkers.training import TrainingRunRulesChecker


# ---------------------------------------------------------------------------
# Training: rules-checker has explicit allow-lists. Cross-check directly.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dotted_key", TrainingRunRulesChecker.CLOSED_ALLOWED_PARAMS)
def test_closed_training_accepts_every_closed_allowed_param(dotted_key):
    """Every key in TrainingRunRulesChecker.CLOSED_ALLOWED_PARAMS must parse in closed mode.

    The parser surface is --params (which accepts any KEY=VALUE); the per-key
    filtering is the rules-checker's job. So all we assert here is that --params
    is *registered* in closed mode and survives an actual dotted-key payload.
    """
    argv = ['mlpstorage', 'closed', 'training', 'unet3d', 'datagen', 'file',
            '--num-processes', '8', '--results-dir', '/tmp',
            '--params', f'{dotted_key}=1']
    with patch('sys.argv', argv):
        ns = parse_arguments()
    flattened = [kv for batch in (ns.params or []) for kv in batch]
    assert f'{dotted_key}=1' in flattened, (
        f"closed training failed to round-trip --params {dotted_key}=1 — "
        f"likely #433-class gating drift between parser and rules-checker"
    )


def test_open_training_accepts_open_only_params():
    """OPEN_ALLOWED_PARAMS must reach the parser in open mode."""
    payload = [f'{k}=x' for k in TrainingRunRulesChecker.OPEN_ALLOWED_PARAMS]
    argv = ['mlpstorage', 'open', 'training', 'unet3d', 'datagen', 'file',
            '--num-processes', '8', '--results-dir', '/tmp',
            '--params', *payload]
    with patch('sys.argv', argv):
        ns = parse_arguments()
    flattened = [kv for batch in (ns.params or []) for kv in batch]
    for key in TrainingRunRulesChecker.OPEN_ALLOWED_PARAMS:
        assert f'{key}=x' in flattened


# ---------------------------------------------------------------------------
# Checkpointing: Rules.md §4.6 Table 3 is the source of truth.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["closed", "open"])
def test_checkpointing_checkpoint_folder_is_changeable(mode):
    """Rules.md §4.6 Table 3: --checkpoint-folder is Changeable in CLOSED and OPEN."""
    argv = ['mlpstorage', mode, 'checkpointing', 'run',
            '-cm', '64', '-m', 'llama3-8b', '-np', '8',
            '-cf', '/tmp/some/custom/path', '-rd', '/tmp', 'file']
    with patch('sys.argv', argv):
        ns = parse_arguments()
    assert ns.checkpoint_folder == '/tmp/some/custom/path'


@pytest.mark.parametrize("mode", ["closed", "open"])
def test_checkpointing_dlio_bin_path_is_accessible(mode):
    """--dlio-bin-path is a deployment knob, not a submission tunable.

    Training exposes it in core args; checkpointing was inconsistent until this
    audit (see .planning/issue-433-gating-audit.md). It should be accessible
    regardless of mode.
    """
    argv = ['mlpstorage', mode, 'checkpointing', 'run',
            '-cm', '64', '-m', 'llama3-8b', '-np', '8',
            '-cf', '/tmp/ckpt', '-rd', '/tmp', 'file',
            '--dlio-bin-path', '/opt/custom/dlio']
    with patch('sys.argv', argv):
        ns = parse_arguments()
    assert ns.dlio_bin_path == '/opt/custom/dlio'


# ---------------------------------------------------------------------------
# Training: --dlio-bin-path is a deployment knob; both modes should accept it.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["closed", "open"])
def test_training_dlio_bin_path_is_accessible(mode):
    argv = ['mlpstorage', mode, 'training', 'unet3d', 'run', 'file',
            '-cm', '64', '-at', 'b200', '-na', '4', '-rd', '/tmp',
            '--dlio-bin-path', '/opt/custom/dlio']
    with patch('sys.argv', argv):
        ns = parse_arguments()
    assert ns.dlio_bin_path == '/opt/custom/dlio'
