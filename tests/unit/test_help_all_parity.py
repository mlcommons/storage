"""
HELP-04: the hand-curated --help_all reference must stay in lockstep with argparse.

The COMPLETE COMMAND REFERENCE in mlpstorage_py/cli/help_formatter.py is
hand-maintained, not generated from the parser, so a parser-only change
silently drifts the reference (that is how --checkpoint-subset went missing,
storage#844). These tests walk the real tree from build_parser() and diff it
against the reference blocks, resolving the block notation:

    CK_RUN_OPEN
      = CK_RUN_CLOSED plus:      <- inherits every flag of the parent block
      + MPI_ARGS                 <- pulls in a common argument group

Four invariants per leaf command:
  1. every long option the parser accepts appears in the leaf's resolved block
  2. every flag the resolved block declares exists on that leaf's parser
  3. every "--long/-short" pair the block writes matches the parser's aliases
  4. choice sets written as {a,b,c} in a block's own text match the parser

Plus one global invariant: every parser action carries a help string.
"""

import argparse
import re

import pytest

from mlpstorage_py.cli.help_formatter import HELP_ALL_TEXT
from mlpstorage_py.cli_parser import build_parser


# =====================================================================
# Reference-side: extract and resolve the documentation blocks
# =====================================================================

# Block headers sit at column 0: "TR_RUN_CLOSED", "CORE_STD — Standard ..."
_BLOCK_HEADER_RE = re.compile(r'^([A-Z][A-Z0-9_]{2,})(?:\s+—.*)?$')
_SECTION_HEADER_PREFIXES = ('Placeholder definitions', 'Common argument groups')

# A flag *declaration* line: indented, starting with a --flag token
# (optionally --long/-short or --long/--alias). Prose mentions of flags
# ("defaults to the --hosts count") deliberately do not match.
_DECLARATION_RE = re.compile(r'^\s+(--[A-Za-z][\w-]*(?:/-{1,2}[A-Za-z][\w-]*)*)')

_CHOICES_RE = re.compile(r'\{([^{}]+)\}')

# Inheritance ("= PARENT plus:") and group references ("+ MPI_ARGS")
_REF_RE = re.compile(r'^\s*[=+]\s+([A-Z][A-Z0-9_]{2,})', re.M)


def _extract_blocks(text):
    blocks = {}
    current = None
    for line in text.split('\n'):
        m = _BLOCK_HEADER_RE.match(line)
        if m:
            current = m.group(1)
            blocks[current] = []
            continue
        if line.startswith(_SECTION_HEADER_PREFIXES):
            current = None
            continue
        if current is not None:
            blocks[current].append(line)
    return {name: '\n'.join(body) for name, body in blocks.items()}


BLOCKS = _extract_blocks(HELP_ALL_TEXT)


def _resolve(name, _seen=None):
    """Block text plus everything it inherits or includes, transitively."""
    if _seen is None:
        _seen = set()
    if name in _seen or name not in BLOCKS:
        return ''
    _seen.add(name)
    text = BLOCKS[name]
    parts = [text]
    for ref in _REF_RE.findall(text):
        parts.append(_resolve(ref, _seen))
    return '\n'.join(parts)


# =====================================================================
# Parser-side: walk every leaf command of the real tree
# =====================================================================

def _walk(parser, path, out):
    sub = None
    options, positionals = [], []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            sub = action
        elif isinstance(action, (argparse._HelpAction, argparse._VersionAction)):
            continue
        elif action.option_strings:
            options.append(action)
        else:
            positionals.append(action)
    if sub is None:
        out.append((tuple(path), options, positionals))
    else:
        for name, subparser in sub.choices.items():
            _walk(subparser, path + [name], out)


_LEAVES = []
_walk(build_parser(), [], _LEAVES)

_BENCH_PREFIX = {'training': 'TR', 'checkpointing': 'CK', 'vectordb': 'VDB', 'kvcache': 'KV'}
_SPECIAL_BLOCKS = {
    ('reports', 'reportgen'): 'RP_REPORTGEN',
    ('history', 'show'): 'HI_SHOW',
    ('history', 'rerun'): 'HI_RERUN',
    ('lockfile', 'generate'): 'LF_GENERATE',
    ('lockfile', 'verify'): 'LF_VERIFY',
    ('init',): 'INIT',
    ('version',): 'VERSION',
    ('validate',): 'VALIDATE',
    ('rules-coverage',): 'RULES_COVERAGE',
}


def _block_for(path):
    if path in _SPECIAL_BLOCKS:
        return _SPECIAL_BLOCKS[path]
    if len(path) == 4 and path[1] == 'training':
        # ('closed', 'training', '<model>', 'run') — flags are identical across
        # models; the reference documents one block per (command, mode).
        mode, bench, _model, cmd = path
        return f'{_BENCH_PREFIX[bench]}_{cmd.upper()}_{mode.upper()}'
    if len(path) == 3 and path[0] in ('closed', 'open', 'whatif'):
        mode, bench, cmd = path
        return f'{_BENCH_PREFIX[bench]}_{cmd.upper()}_{mode.upper()}'
    raise AssertionError(f'no --help_all block mapping for parser path {path!r}')


# Flags deliberately left out of the --help_all reference. Keep this empty
# unless there is a stated reason a flag must stay undocumented.
UNDOCUMENTED_OK = set()  # e.g. {('closed/training/run', '--some-flag')}

_LEAF_IDS = ['/'.join(p) for p, _, _ in _LEAVES]


# =====================================================================
# 1. Every parser flag is documented in its resolved block
# =====================================================================

@pytest.mark.parametrize('path, options, positionals', _LEAVES, ids=_LEAF_IDS)
def test_every_parser_flag_documented(path, options, positionals):
    block = _block_for(path)
    text = _resolve(block)
    assert text.strip(), \
        f'--help_all has no block {block} for command {"/".join(path)}'
    missing = []
    for action in options:
        longs = [o for o in action.option_strings if o.startswith('--')]
        if not longs:
            continue
        if ('/'.join(path), longs[0]) in UNDOCUMENTED_OK:
            continue
        if not any(l in text for l in longs):
            missing.append(longs[0])
    assert not missing, (
        f'{"/".join(path)}: flags accepted by the parser but absent from '
        f'--help_all block {block} (or its inherited/included blocks): {missing}'
    )


# =====================================================================
# 2. Every flag the block declares exists on the leaf's parser
# =====================================================================

@pytest.mark.parametrize('path, options, positionals', _LEAVES, ids=_LEAF_IDS)
def test_every_documented_flag_exists(path, options, positionals):
    block = _block_for(path)
    text = _resolve(block)
    parser_flags = set()
    for action in options:
        parser_flags.update(action.option_strings)
    stale = []
    for line in text.split('\n'):
        m = _DECLARATION_RE.match(line)
        if not m:
            continue
        declared = m.group(1).split('/')
        if not any(alias in parser_flags for alias in declared):
            stale.append(m.group(1))
    assert not stale, (
        f'{"/".join(path)}: --help_all block {block} declares flags the '
        f'parser does not accept (renamed or removed?): {stale}'
    )


# =====================================================================
# 3. Documented --long/-short alias pairs match the parser
# =====================================================================

@pytest.mark.parametrize('path, options, positionals', _LEAVES, ids=_LEAF_IDS)
def test_documented_alias_pairs_match(path, options, positionals):
    block = _block_for(path)
    text = _resolve(block)
    by_flag = {}
    for action in options:
        for opt in action.option_strings:
            by_flag[opt] = set(action.option_strings)
    mismatched = []
    for line in text.split('\n'):
        m = _DECLARATION_RE.match(line)
        if not m:
            continue
        declared = m.group(1).split('/')
        if len(declared) < 2:
            continue
        anchor = next((a for a in declared if a in by_flag), None)
        if anchor is None:
            continue  # stale flag — test 2 reports it
        wrong = [a for a in declared if a not in by_flag[anchor]]
        if wrong:
            mismatched.append((m.group(1), sorted(by_flag[anchor])))
    assert not mismatched, (
        f'{"/".join(path)}: --help_all block {block} writes alias pairs that '
        f'do not match the parser (documented, actual): {mismatched}'
    )


# =====================================================================
# 4. Choice sets written as {a,b,c} in a block's OWN text match the parser
# =====================================================================

@pytest.mark.parametrize('path, options, positionals', _LEAVES, ids=_LEAF_IDS)
def test_documented_choices_match(path, options, positionals):
    block = _block_for(path)
    text = BLOCKS.get(block, '')  # own text only — inherited blocks may
    # legitimately show a different mode's choice set
    by_flag = {}
    for action in options:
        for opt in action.option_strings:
            by_flag[opt] = action
    wrong = []
    for line in text.split('\n'):
        m = _DECLARATION_RE.match(line)
        if not m:
            continue
        anchor = next((a for a in m.group(1).split('/') if a in by_flag), None)
        if anchor is None:
            continue
        cm = _CHOICES_RE.search(line)
        if not cm:
            continue
        documented = {c.strip() for c in cm.group(1).split(',')}
        actual = by_flag[anchor].choices
        if actual is None:
            wrong.append((anchor, sorted(documented), None))
        elif documented != {str(c) for c in actual}:
            wrong.append((anchor, sorted(documented), sorted(str(c) for c in actual)))
    assert not wrong, (
        f'{"/".join(path)}: --help_all block {block} documents choice sets '
        f'that do not match the parser (flag, documented, actual): {wrong}'
    )


# =====================================================================
# 5. Storage positional and user positionals are documented
# =====================================================================

@pytest.mark.parametrize('path, options, positionals', _LEAVES, ids=_LEAF_IDS)
def test_positionals_documented(path, options, positionals):
    block = _block_for(path)
    text = _resolve(block)
    for action in positionals:
        if action.dest == 'data_access_protocol':
            assert 'file | object' in text, (
                f'{"/".join(path)}: takes the file|object storage positional '
                f'but block {block} never mentions "file | object"'
            )
        else:
            assert action.dest in text, (
                f'{"/".join(path)}: positional {action.dest!r} missing from '
                f'block {block}'
            )


# =====================================================================
# 6. Every parser action carries a help string (argparse --help surface)
# =====================================================================

def test_every_action_has_help_string():
    bare = []
    for path, options, positionals in _LEAVES:
        for action in options + positionals:
            if not (action.help or '').strip():
                bare.append(('/'.join(path), action.option_strings or action.dest))
    # Collapse duplicates (universal args repeat across every leaf)
    unique = sorted({(p.split('/')[-1], str(f)) for p, f in bare})
    assert not bare, (
        f'parser actions with no help= string (shown as (command, flag), '
        f'deduplicated): {unique}'
    )


# =====================================================================
# 7. The command tree in --help_all reflects the real shape
# =====================================================================

def test_tree_shows_training_model_choices():
    """Training is the only benchmark with a model positional; the tree rows
    must carry the real per-mode choice sets."""
    assert 'unet3d | retinanet' in HELP_ALL_TEXT
    assert 'cosmoflow | resnet50 | unet3d | dlrm | retinanet | flux' in HELP_ALL_TEXT


def test_tree_history_subcommands_are_show_and_rerun():
    """The history subcommands are `show` and `rerun` — the reference must not
    keep documenting the old `list`/`replay` names."""
    assert 'HI_SHOW' in HELP_ALL_TEXT
    assert 'HI_RERUN' in HELP_ALL_TEXT
    assert 'HI_LIST' not in HELP_ALL_TEXT
    assert 'HI_REPLAY' not in HELP_ALL_TEXT
