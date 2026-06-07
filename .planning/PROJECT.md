# MLPerf Storage — CLI Redesign Project

## Goal

Refactor the `mlpstorage` CLI from a flag-gated `--open`/`--closed` design to a
fully positional, three-mode hierarchy:

```
mlpstorage [closed|open|whatif] <benchmark> <model|algorithm> <command> [file|object] [options]
mlpstorage (reports|history|lockfile|version) [subcommand] [options]
```

The new design eliminates `is_closed` threading throughout all arg-builder functions,
replaces it with three distinct parser branches built once, and adds context-sensitive
`--help` and `--help_all` output.

## Constraints

- Python ≥ 3.12 (stdlib `tomllib` available)
- No new runtime dependencies beyond what is already in pyproject.toml
- Closed submissions: restricted model/accelerator choices, open-gated params locked to
  defaults and hidden from help
- Whatif mode: all models including legacy, all accelerators — for experimental use
- `reports`, `history`, `lockfile`, `version` are top-level siblings, NOT nested under
  closed/open/whatif
- kvcache: no file|object storage positional (architectural, not policy)
- kvcache closed: no model positional (fixed 3-phase sequence)
- datasize commands: no file|object storage positional

## Key design files (all in `plans/`)

| File | Contents |
|------|----------|
| `plans/help_all_spec.md` | Complete command tree, common groups, all placeholder definitions |
| `plans/version_command.md` | `version` subcommand + VERSION bug fix plan + regression tests |
| `plans/architecture_analysis.md` | Existing architecture notes |

## Key design files (in memory/)

These are loaded into the main conversation context automatically:

| File | Contents |
|------|----------|
| `memory/cli_tree_design.md` | Full positional hierarchy, model/accelerator choices per mode, design principles |
| `memory/cli_parameter_tiers.md` | Core / open-gated / whatif-only classification for all 4 benchmarks |
| `memory/cli_architecture.md` | Current arg builder structure and function signatures |
| `memory/merge_conflicts.md` | 8 unresolved merge conflict blocks across 5 files |
| `memory/version_command.md` | version command plan summary + VERSION bug description |
| `memory/training_model_accelerator_matrix.md` | Dynamic file-presence check for model+accelerator matrix |
| `memory/user_preferences.md` | Collaboration style: design before code, proposal-first |

## Current branch state

Branch: `main` (ahead of `origin/main`)

Five files have unresolved merge conflicts (HEAD = new positional design;
origin/main = old flag-based design):

- `mlpstorage_py/cli/training_args.py`
- `mlpstorage_py/cli/checkpointing_args.py`
- `mlpstorage_py/cli/kvcache_args.py`
- `mlpstorage_py/cli/vectordb_args.py`
- `mlpstorage_py/cli/common_args.py`

The HEAD side of each conflict is the one to keep (it already implements
`is_closed` threading as the first step toward the new design).

## Config.py changes needed

Current constants that need updating:
- `MODELS_CLOSED` currently `[DLRM, RETINANET, FLUX]` → change to `[UNET, RETINANET]`
- Add `MODELS_OPEN = [UNET, RETINANET]` (same as closed for model choices; more flags)
- `ACCELERATORS_CLOSED` currently `[B200, MI355]` → correct (keep)
- `KVCACHE_MODELS_CLOSED` → not needed; model positional is absent from closed kvcache

## VERSION bug (separate but related)

`mlpstorage_py/__init__.py` calls `_pkg_version("mlpstorage_py")` — wrong distribution
name. `pyproject.toml` declares `name = "mlpstorage"`. Fix: change to `"mlpstorage"`,
add `tomllib`-based pyproject.toml fallback. See `plans/version_command.md`.
