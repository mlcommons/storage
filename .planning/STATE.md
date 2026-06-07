---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: unknown
last_updated: "2026-06-07T22:17:29.673Z"
progress:
  total_phases: 4
  completed_phases: 0
  total_plans: 5
  completed_plans: 0
  percent: 0
---

# Project State — as of 2026-06-07

## Status: Design complete, implementation not yet started

All design decisions are resolved and documented.  The next step is bootstrapping
the GSD phase structure and beginning implementation.

## What is done

- [x] Full CLI positional hierarchy agreed (closed/open/whatif × benchmark × model × command × file|object)
- [x] Three-mode parameter tier audit complete (core / open-gated / whatif-only for all 4 benchmarks)
- [x] `--help_all` tree and all placeholder group definitions written (`plans/help_all_spec.md`)
- [x] `version` command plan written (`plans/version_command.md`)
- [x] VERSION bug identified (`mlpstorage_py` → `mlpstorage` distribution name)
- [x] Merge conflict analysis complete (8 blocks, HEAD side is correct)
- [x] `.planning/` directory bootstrapped for GSD workflow

## What is NOT done (implementation backlog)

### P0 — Merge conflicts (blocks everything else)

- Resolve 8 conflict blocks in 5 files; keep HEAD side throughout

### P1 — Core refactor

- Remove `is_closed` parameter from all arg-builder function signatures
- Build three parser branches (closed/open/whatif) in `cli_parser.py`
- Make `model`/`algorithm` a positional (before command) in each benchmark branch
- Make `file`/`object` a positional (after command) for applicable commands
- Update `config.py`: MODELS_CLOSED, add MODELS_OPEN

### P2 — Version command

- Fix `__init__.py` distribution name + add pyproject.toml fallback
- Add `add_version_arguments()` to `utility_args.py`
- Wire `version` subparser into `cli_parser.py`
- Add dispatch in `main.py`
- Write 3 regression tests (`tests/unit/test_version.py`)

### P3 — Help behavior

- Implement `--help_all` pre-parse intercept that emits `plans/help_all_spec.md` content
- Implement context-sensitive `--help` (one-line next-token summary at each positional level)

### P4 — Test coverage

- Unit tests for all three parser modes (closed/open/whatif)
- Verify open-gated args are absent in closed, present in open/whatif
- Verify model/accelerator choice restrictions per mode

## Unresolved questions

None.  All design questions were resolved in the prior conversation session.
See `memory/open_questions.md` for the full list and resolutions.

## Next GSD action

Run `/gsd:plan-phase 1` to generate an execution plan for the P0+P1 work
(merge conflict resolution + core parser refactor).
