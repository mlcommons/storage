"""
Tests for mlpstorage help behavior: HELP-01, HELP-02, HELP-03.

HELP-01: --help_all prints full command reference and exits 0
HELP-02: context-sensitive --help at every mid-tree level shows "next: X | Y"
HELP-03: leaf-level --help falls through to argparse (shows real flags, not "next:")

Includes R-03-01 fix tests: bare mid-tree invocations without --help also show
discovery help (not an argparse error).
"""

import pytest
from unittest.mock import patch
from mlpstorage_py.cli_parser import parse_arguments


# =====================================================================
# 1. TestHelpAll — HELP-01
# =====================================================================

class TestHelpAll:
    """--help_all prints the complete command reference and exits 0."""

    def test_help_all_exits_0(self, capsys):
        with patch('sys.argv', ['mlpstorage', '--help_all']):
            with pytest.raises(SystemExit) as exc:
                parse_arguments()
        assert exc.value.code == 0

    def test_help_all_prints_banner(self, capsys):
        with patch('sys.argv', ['mlpstorage', '--help_all']):
            with pytest.raises(SystemExit):
                parse_arguments()
        out = capsys.readouterr().out
        assert 'MLPSTORAGE' in out
        assert 'COMPLETE COMMAND REFERENCE' in out

    def test_help_all_prints_placeholder_tr(self, capsys):
        with patch('sys.argv', ['mlpstorage', '--help_all']):
            with pytest.raises(SystemExit):
                parse_arguments()
        out = capsys.readouterr().out
        assert 'TR_DATASIZE_CLOSED' in out

    def test_help_all_prints_synopsis(self, capsys):
        with patch('sys.argv', ['mlpstorage', '--help_all']):
            with pytest.raises(SystemExit):
                parse_arguments()
        out = capsys.readouterr().out
        assert 'SYNOPSIS' in out

    def test_help_all_prints_kv_section(self, capsys):
        with patch('sys.argv', ['mlpstorage', '--help_all']):
            with pytest.raises(SystemExit):
                parse_arguments()
        out = capsys.readouterr().out
        assert 'KV_RUN_OPEN' in out

    def test_help_all_contains_key_placeholders(self):
        """R-03-02 fix: guard against HELP_ALL_TEXT drifting from the real parser."""
        from mlpstorage_py.cli.help_formatter import HELP_ALL_TEXT
        assert 'TR_DATASIZE_CLOSED' in HELP_ALL_TEXT, \
            'drift: TR_DATASIZE_CLOSED missing from HELP_ALL_TEXT'
        assert 'KV_RUN_OPEN' in HELP_ALL_TEXT, \
            'drift: KV_RUN_OPEN missing from HELP_ALL_TEXT'


# =====================================================================
# 2. TestContextHelp — HELP-02 and R-03-01 (bare mid-tree invocations)
# =====================================================================

class TestContextHelp:
    """Context-sensitive help: --help and bare mid-tree paths show "next: ..." and exit 0."""

    @pytest.mark.parametrize('argv, expected_fragment', [
        # ── Bare and top-level --help ──────────────────────────────────────────
        (['mlpstorage'],           'next: closed | open | whatif | reports | history | lockfile | version'),
        (['mlpstorage', '--help'], 'next: closed | open | whatif | reports | history | lockfile | version'),
        (['mlpstorage', '-h'],     'next: closed | open | whatif | reports | history | lockfile | version'),

        # ── Mode-level --help ──────────────────────────────────────────────────
        (['mlpstorage', 'closed', '--help'],  'next: training | checkpointing | vectordb | kvcache'),
        (['mlpstorage', 'open', '--help'],    'next: training | checkpointing | vectordb | kvcache'),
        (['mlpstorage', 'whatif', '--help'],  'next: training | checkpointing | vectordb | kvcache'),

        # ── R-03-01 fix: bare mid-tree invocations (no --help flag) ──────────
        (['mlpstorage', 'closed', 'training'],                    'next: unet3d | retinanet'),
        (['mlpstorage', 'closed', 'training', 'unet3d'],          'next: datasize | datagen | run | configview'),
        (['mlpstorage', 'closed', 'training', 'unet3d', 'run'],   'next: file | object'),
        (['mlpstorage', 'closed', 'checkpointing'],               'next: datasize | run | configview'),
        (['mlpstorage', 'closed', 'kvcache'],                     'next: datasize | run'),

        # ── Benchmark-level --help (training: model choices vary by mode) ─────
        (['mlpstorage', 'closed', 'training', '--help'],  'next: unet3d | retinanet'),
        (['mlpstorage', 'open', 'training', '--help'],    'next: unet3d | retinanet'),
        (['mlpstorage', 'whatif', 'training', '--help'],  'next: cosmoflow | resnet50 | unet3d | dlrm | retinanet | flux'),

        # ── Model-level --help (training only) ────────────────────────────────
        (['mlpstorage', 'closed', 'training', 'unet3d', '--help'],    'next: datasize | datagen | run | configview'),
        (['mlpstorage', 'closed', 'training', 'retinanet', '--help'], 'next: datasize | datagen | run | configview'),

        # ── Command-level --help (storage selector) ───────────────────────────
        (['mlpstorage', 'closed', 'training', 'unet3d', 'datagen', '--help'],    'next: file | object'),
        (['mlpstorage', 'closed', 'training', 'unet3d', 'run', '--help'],        'next: file | object'),
        (['mlpstorage', 'closed', 'training', 'unet3d', 'configview', '--help'], 'next: file | object'),

        # ── Checkpointing (no model positional) ──────────────────────────────
        (['mlpstorage', 'closed', 'checkpointing', '--help'],          'next: datasize | run | configview'),
        (['mlpstorage', 'closed', 'checkpointing', 'run', '--help'],        'next: file | object'),
        (['mlpstorage', 'closed', 'checkpointing', 'configview', '--help'], 'next: file | object'),

        # ── VectorDB ──────────────────────────────────────────────────────────
        (['mlpstorage', 'closed', 'vectordb', '--help'],            'next: datasize | datagen | run'),
        (['mlpstorage', 'closed', 'vectordb', 'datagen', '--help'], 'next: file | object'),
        (['mlpstorage', 'closed', 'vectordb', 'run', '--help'],     'next: file | object'),

        # ── Kvcache (no file|object at any level) ────────────────────────────
        (['mlpstorage', 'closed', 'kvcache', '--help'], 'next: datasize | run'),

        # ── Utility commands ──────────────────────────────────────────────────
        (['mlpstorage', 'reports', '--help'],  'next: reportgen'),
        (['mlpstorage', 'history', '--help'],  'next: show | rerun'),
        (['mlpstorage', 'lockfile', '--help'], 'next: generate | verify'),
    ])
    def test_context_help(self, argv, expected_fragment, capsys):
        with patch('sys.argv', argv):
            with pytest.raises(SystemExit) as exc:
                parse_arguments()
        assert exc.value.code == 0, \
            f'Expected exit 0 for {argv}, got {exc.value.code}'
        out = capsys.readouterr().out
        assert expected_fragment in out, \
            f'Expected {expected_fragment!r} in stdout for {argv}\nGot: {out!r}'


# =====================================================================
# 3. TestBareInvocation — bare mlpstorage (no args)
# =====================================================================

class TestBareInvocation:
    """Bare mlpstorage invocation must print discovery help and exit 0."""

    def test_bare_invocation_exits_0(self):
        with patch('sys.argv', ['mlpstorage']):
            with pytest.raises(SystemExit) as exc:
                parse_arguments()
        assert exc.value.code == 0

    def test_bare_invocation_prints_modes(self, capsys):
        with patch('sys.argv', ['mlpstorage']):
            with pytest.raises(SystemExit):
                parse_arguments()
        out = capsys.readouterr().out
        assert 'next: closed | open | whatif' in out


# =====================================================================
# 4. TestLeafHelp — HELP-03 leaf fallthrough to argparse
# =====================================================================

class TestLeafHelp:
    """At leaf level, get_context_help_tokens returns None, argparse handles --help.

    Characteristics:
    - stdout does NOT start with "next:"
    - stdout DOES contain "--" (argparse flag output)
    """

    @pytest.mark.parametrize('argv', [
        # training datasize (leaf — no storage positional)
        ['mlpstorage', 'closed', 'training', 'unet3d', 'datasize', '--help'],
        # training datagen file (leaf — storage positional supplied)
        ['mlpstorage', 'closed', 'training', 'unet3d', 'datagen', 'file', '--help'],
        # checkpointing datasize (leaf)
        ['mlpstorage', 'closed', 'checkpointing', 'datasize', '--help'],
        # kvcache run (leaf — no file|object for kvcache)
        ['mlpstorage', 'closed', 'kvcache', 'run', '--help'],
        # R-03-04 fix: version --help falls through to argparse (not "next:")
        ['mlpstorage', 'version', '--help'],
    ])
    def test_leaf_shows_flags(self, argv, capsys):
        with patch('sys.argv', argv):
            with pytest.raises(SystemExit) as exc:
                parse_arguments()
        assert exc.value.code == 0, \
            f'Expected exit 0 for {argv}, got {exc.value.code}'
        out = capsys.readouterr().out
        assert not out.strip().startswith('next:'), \
            f'Leaf path {argv} must NOT output "next:..." — got: {out!r}'
        assert '--' in out, \
            f'Leaf path {argv} must output argparse flags (containing "--") — got: {out!r}'


# =====================================================================
# 5. TestRegression — spot-check existing parse paths still work
# =====================================================================

class TestRegression:
    """Positive-path parse_arguments() calls must return args (not SystemExit)."""

    @pytest.mark.parametrize('argv, expected_attrs', [
        # training datasize with flags interspersed — pre-scan strips flags, leaving
        # ['closed', 'training', 'unet3d', 'datasize'] → None → falls through to argparse
        (
            ['mlpstorage', 'closed', 'training', 'unet3d', 'datasize',
             '-cm', '64', '-at', 'b200', '-ma', '4'],
            {'mode': 'closed', 'benchmark': 'training', 'command': 'datasize'},
        ),
        # checkpointing run with file storage positional
        (
            ['mlpstorage', 'closed', 'checkpointing', 'run',
             '-cm', '1024', '-m', 'llama3-8b', '-np', '2',
             '-cf', '/tmp/ckpt', '-rd', '/tmp', 'file'],
            {'benchmark': 'checkpointing', 'command': 'run'},
        ),
        # version subcommand
        (
            ['mlpstorage', 'version'],
            {'mode': 'version'},
        ),
    ])
    def test_existing_parse_paths(self, argv, expected_attrs):
        with patch('sys.argv', argv):
            args = parse_arguments()
        for attr, expected in expected_attrs.items():
            actual = getattr(args, attr, None)
            assert actual == expected, \
                f'For {argv}: expected args.{attr}={expected!r}, got {actual!r}'
