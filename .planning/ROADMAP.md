# Roadmap: MLPerf Storage — CLI Redesign

## Overview

Refactor the `mlpstorage` CLI from a flag-gated `--open`/`--closed` design to a fully positional, three-mode hierarchy (`closed|open|whatif`). The refactor proceeds in five phases: first rebuilding the core parser from the ground up (resolving merge conflicts in the same pass), then adding the `version` command, then implementing context-sensitive help, then hardening with a full test suite, and finally adding a run configuration summary that prints all effective parameter values after argument parsing.

## Phases

- [ ] **Phase 1: Core CLI Refactor** — Resolve merge conflicts + build three-branch positional parser, eliminating `is_closed`
- [ ] **Phase 2: Version Command** — Fix VERSION bug + implement `mlpstorage version` subcommand
- [ ] **Phase 3: Help Behavior** — `--help_all` full tree output + context-sensitive `--help` at each positional level
- [ ] **Phase 4: Test Coverage** — Three-mode parser tests, open-gated arg validation, model/accelerator restriction tests
- [ ] **Phase 5: Run Configuration Summary** — Print effective parameter values after argument parsing; centralized S3 config resolver; `--quiet` suppression

## Phase Details

### Phase 1: Core CLI Refactor
**Goal**: Rebuild the argument parser as three distinct branches (`closed`, `open`, `whatif`) with positional `model`/`algorithm` before command and `file|object` after command, eliminating `is_closed` throughout. Resolve all 8 merge conflict blocks as part of this rewrite (the conflicts are all in files being rewritten anyway).
**Depends on**: Nothing (first phase)
**Requirements**: CLI-01, CLI-02, CLI-03, CLI-04, CLI-05, CLI-06, CLI-07, CLI-08, CLI-09, CLI-10
**Success Criteria** (what must be TRUE):
  1. `mlpstorage closed training unet3d run file --help` shows closed-mode training flags without open-gated args
  2. `mlpstorage open training unet3d run file --help` shows open-gated args (--loops, --params, --timeseries-*)
  3. `mlpstorage whatif training cosmoflow run file --help` accepts whatif-only models
  4. No `is_closed` parameter in any arg-builder function signature
  5. All 8 merge conflict markers (`<<<<<<<`) are gone from all files
  6. `pytest tests/unit -v` passes (existing tests not broken)
**Plans**: 5 plans in 3 waves

Plans:
- [ ] 01-01-PLAN.md — Fix config.py constants (MODELS_CLOSED, MODELS_OPEN) + resolve common_args.py conflict markers and simplify all function signatures
- [ ] 01-02-PLAN.md — Rewrite training_args.py and checkpointing_args.py with three-tier builder pattern; update benchmarks/dlio.py
- [ ] 01-03-PLAN.md — Rewrite vectordb_args.py and kvcache_args.py with three-tier builder pattern; update benchmarks/kvcache.py and vectordbbench.py
- [ ] 01-04-PLAN.md — Simplify utility_args.py and lockfile_args.py signatures; update cli/__init__.py exports
- [ ] 01-05-PLAN.md — Rewrite cli_parser.py with three-branch positional structure; update main.py and base.py dispatch; fix unit tests

### Phase 2: Version Command
**Goal**: Fix the VERSION bug (wrong distribution name `"mlpstorage_py"` → `"mlpstorage"`) and add `mlpstorage version` as a top-level subcommand that prints the installed package version.
**Depends on**: Phase 1
**Requirements**: VER-01, VER-02, VER-03
**Success Criteria** (what must be TRUE):
  1. `mlpstorage version` exits 0 and prints a non-empty version string
  2. Version string matches what `pip show mlpstorage` reports
  3. When package is not installed (editable dev mode), version falls back to pyproject.toml parse via `tomllib`
  4. `pytest tests/unit/test_version.py -v` passes all 3 regression tests
**Plans**: TBD

Plans:
- [ ] 02-01: Fix VERSION bug in `__init__.py` and wire `version` into `cli_parser.py` and `main.py`

### Phase 3: Help Behavior
**Goal**: Implement `mlpstorage --help_all` (full tree with placeholder definitions) and context-sensitive `--help` that lists valid next-token options at each positional level.
**Depends on**: Phase 1
**Requirements**: HELP-01, HELP-02, HELP-03
**Success Criteria** (what must be TRUE):
  1. `mlpstorage --help_all` prints the complete tree from `plans/help_all_spec.md` and all placeholder group definitions
  2. `mlpstorage` (bare) prints `next: closed | open | whatif | reports | history | lockfile | version`
  3. `mlpstorage closed training` prints `next: unet3d | retinanet`
  4. `mlpstorage closed training unet3d` prints `next: datasize | datagen | run | configview`
  5. `mlpstorage closed training unet3d datasize` prints flags for `TR_DATASIZE_CLOSED`
**Plans**: TBD

Plans:
- [ ] 03-01: Implement `--help_all` pre-parse intercept and context-sensitive `--help` formatter

### Phase 4: Test Coverage
**Goal**: Add comprehensive unit tests validating all three parser modes, open-gated argument exclusion in closed mode, and model/accelerator choice restrictions.
**Depends on**: Phase 1
**Requirements**: TEST-01, TEST-02, TEST-03, TEST-04
**Success Criteria** (what must be TRUE):
  1. `pytest tests/unit -v` passes with 0 failures
  2. Closed-mode parser raises error when open-gated arg (e.g., `--loops 2`) is supplied
  3. `closed training` accepts only [unet3d, retinanet]; `whatif training` accepts [cosmoflow, resnet50, unet3d, dlrm, retinanet, flux]
  4. `closed training unet3d run file --accelerator-type h100` raises an error (h100 not in closed accelerators)
**Plans**: TBD

Plans:
- [ ] 04-01: Three-mode parser unit tests (closed/open/whatif structure and argument availability)

### Phase 5: Run Configuration Summary
**Goal**: After successful argument parsing, print a structured summary of all parameters in effect for the run. User sees exactly what values the benchmark will use — not just what was typed.
**Depends on**: Phase 1
**Requirements**: RUNSUM-01, RUNSUM-02, RUNSUM-03, RUNSUM-04, RUNSUM-05, RUNSUM-06, RUNSUM-07
**Success Criteria** (what must be TRUE):
  1. Every `run`, `datagen`, `datasize`, and `configview` invocation prints a labeled table of Tier 1 CLI args before benchmark execution
  2. `--quiet` suppresses the summary table (no output other than the benchmark itself)
  3. When `data_access_protocol == 'object'`, the table includes a second section for S3/object-storage env vars; absent when protocol is `file`
  4. `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are never printed in plaintext — shown as `[SET — N chars]` or `[not set]`
  5. Resolved S3 endpoint row shows both the resolved value and the source env var label (e.g., `endpoint: s3.example.com  [from S3_ENDPOINT_URIS]`)
  6. A new `resolve_object_storage_config()` function is the single place that reads S3 env vars — all existing readers/writers import from it (no scattered `os.environ.get` calls for S3 config)
  7. `pytest tests/unit -v` passes with 0 regressions
**Plans**: 2 plans in 2 waves

Plans:
- [ ] 05-01-PLAN.md — Create mlpstorage_py/storage_config.py with resolve_object_storage_config(); replace scattered S3 env reads in all 6 storage files; write test_storage_config.py
- [ ] 05-02-PLAN.md — Create mlpstorage_py/run_summary.py with print_run_summary(); wire into main.py post-parse; add --quiet to common_args.py Output Control group; write test_run_summary.py

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Core CLI Refactor | 0/5 | Planned | - |
| 2. Version Command | 0/1 | Not started | - |
| 3. Help Behavior | 0/1 | Not started | - |
| 4. Test Coverage | 0/1 | Not started | - |
| 5. Run Configuration Summary | 0/2 | Planned | - |
