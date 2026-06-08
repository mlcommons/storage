---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: phase-4-plan-1-complete
last_updated: "2026-06-08T15:23:00Z"
progress:
  total_phases: 4
  completed_phases: 4
  total_plans: 9
  completed_plans: 9
  percent: 100
---

# Project State — as of 2026-06-08

## Status: Phase 4 Plan 01 complete — test_parser_modes.py with 47 tests covering TEST-01 through TEST-04 (parser mode structure, open-gated arg exclusion, model/accelerator restrictions, version dispatch).

## Phase 1 Complete — Core CLI Refactor (verified 2026-06-08)

All 5 plans executed and all 6 ROADMAP success criteria verified:

- [x] 01-01: config.py — MODELS_CLOSED, MODELS_OPEN constants; common_args.py conflict resolution + signature simplification
- [x] 01-02: training_args.py + checkpointing_args.py three-tier builder rewrite; benchmarks/dlio.py updated
- [x] 01-03: vectordb_args.py + kvcache_args.py three-tier builder rewrite; kvcache.py + vectordbbench.py updated
- [x] 01-04: utility_args.py + lockfile_args.py simplification; cli/__init__.py exports updated
- [x] 01-05: cli_parser.py three-branch positional rewrite; main.py + base.py dispatch updated; unit tests fixed

Verification evidence:

- `parse_arguments()` imports cleanly
- `closed training unet3d run ... file` → `mode=closed, benchmark=training, model=unet3d, data_access_protocol=file`
- `whatif training cosmoflow run ... file` → `model=cosmoflow, mode=whatif`
- `closed ... --loops 2 ...` → SystemExit(2), rejected as unrecognized arg
- Open mode help shows `--loops`, `--params`, `--timeseries-*`; closed mode does not
- `grep -r "is_closed" mlpstorage_py/cli/` → no results
- `grep -rn "<<<<<<<" mlpstorage_py/` → no results
- `pytest tests/unit/test_cli_parser.py tests/unit/test_cli.py -q` → **110 passed**

See `.planning/phases/01-core-cli-refactor/01-VERIFICATION.md` for full report.

## What is done

- [x] Full CLI positional hierarchy agreed and implemented (closed/open/whatif × benchmark × model × command × file|object)
- [x] Three-mode parameter tier audit complete and implemented (core / open-gated / whatif-only for all 4 benchmarks)
- [x] `--help_all` tree and all placeholder group definitions written (`plans/help_all_spec.md`)
- [x] `version` command plan written (`plans/version_command.md`)
- [x] VERSION bug identified (`mlpstorage_py` → `mlpstorage` distribution name)
- [x] Merge conflict analysis complete and resolved (8 blocks, HEAD side kept throughout)
- [x] `.planning/` directory bootstrapped for GSD workflow
- [x] Phase 1 Core CLI Refactor — all 5 plans executed and verified
- [x] Phase 2 Plan 01 — VERSION bug fixed; _resolve_version() with correct dist name + tomllib fallback; 3 regression tests
- [x] Phase 2 Plan 02 — version subcommand wired; add_version_arguments() + version_parser + early-exit dispatch before HistoryTracker
- [x] Phase 3 Plan 01 — help behavior: help_formatter.py (HELP_ALL_TEXT + get_context_help_tokens), pre-scan guard in parse_arguments(), 45 tests in test_help_behavior.py
- [x] Phase 4 Plan 01 — test coverage: test_parser_modes.py (47 tests), TEST-01 through TEST-04 all complete

## Decisions accumulated

- Distribution name is 'mlpstorage' (not 'mlpstorage_py') for importlib.metadata lookup
- Version early-exit placed immediately after parse_arguments() in _main_impl() — before args.debug access — to avoid AttributeError (version subparser has no --debug flag) while also satisfying HistoryTracker bypass constraint
- Pre-scan guard fires UNCONDITIONALLY (R-03-01): get_context_help_tokens() returns None for leaf/unrecognised tokens enabling argparse fallthrough
- HELP_ALL_TEXT is a static string constant; no runtime spec file reading required
- history subcommands are 'show'/'rerun' (not 'list'/'replay'); code is authoritative over research doc

## What is NOT done (implementation backlog)

### P2 — Version command (COMPLETE)

- [x] Fix `__init__.py` distribution name + add pyproject.toml fallback (02-01 complete)
- [x] Write 3 regression tests (`tests/unit/test_version.py`) (02-01 complete)
- [x] Add `add_version_arguments()` to `utility_args.py` (02-02 complete)
- [x] Wire `version` subparser into `cli_parser.py` (02-02 complete)
- [x] Add dispatch in `main.py` (02-02 complete)

### P3 — Help behavior (COMPLETE)

- [x] Implement `--help_all` pre-parse intercept (03-01 complete)
- [x] Implement context-sensitive `--help` and bare mid-tree discovery help (03-01 complete)

### P4 — Test coverage

- Unit tests for all three parser modes (closed/open/whatif)
- Verify open-gated args are absent in closed, present in open/whatif
- Verify model/accelerator choice restrictions per mode

## Unresolved questions

None.

## Next GSD action

Run `/gsd:plan-phase 4` to plan Phase 4 (test coverage: closed/open/whatif mode parity, open-gated arg verification, model/accelerator choice restrictions).
