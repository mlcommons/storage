# Requirements: MLPerf Storage — CLI Redesign

**Defined:** 2026-06-07
**Core Value:** Replace flag-gated `--open`/`--closed` design with a fully positional three-mode CLI hierarchy (closed/open/whatif), eliminating `is_closed` threading and enabling context-sensitive help.

## v1 Requirements

### CLI Architecture

- [ ] **CLI-01**: `mlpstorage closed|open|whatif <benchmark> <model|algorithm> <command> [file|object] [OPTIONS]` parses correctly for all four benchmarks
- [ ] **CLI-02**: `model`/`algorithm` positional appears BEFORE command in all benchmark branches
- [ ] **CLI-03**: `file|object` storage positional appears AFTER command for datagen/run/configview; absent for datasize and all kvcache commands
- [x] **CLI-04**: `is_closed` parameter is eliminated from all arg-builder function signatures; mode is implicit in which parser branch is being built
- [ ] **CLI-05**: Three builder tiers are implemented: `_add_{cmd}_core_args` (all modes), `_add_{cmd}_open_args` (open+whatif), `_add_{cmd}_whatif_args` (whatif only)
- [ ] **CLI-06**: Open-gated arguments (--loops, --allow-invalid-params, --timeseries-*, --params, etc.) are absent from closed branch help and raise an error if supplied in closed mode
- [x] **CLI-07**: `config.py` constants corrected: `MODELS_CLOSED = [unet3d, retinanet]`, `MODELS_OPEN = [unet3d, retinanet]` (new), `MODELS` (whatif) unchanged
- [x] **CLI-08**: `reports`, `history`, `lockfile`, `version` remain top-level siblings (not nested under closed/open/whatif)
- [ ] **CLI-09**: kvcache closed has no model positional; kvcache open/whatif has model positional from `KVCACHE_MODELS`
- [x] **CLI-10**: Merge conflicts in 5 files resolved as part of the refactor (HEAD side kept; `add_storage_type_arguments` integrated)

### Version Command

- [ ] **VER-01**: `mlpstorage version` prints the installed package version and exits 0
- [x] **VER-02**: Version lookup uses distribution name `"mlpstorage"` (not `"mlpstorage_py"`); falls back to `tomllib`-based pyproject.toml parse if package not installed
- [ ] **VER-03**: `mlpstorage version` is a top-level sibling of closed/open/whatif (not nested under them)

### Help Behavior

- [ ] **HELP-01**: `mlpstorage --help_all` prints the full command tree with named option-group placeholders, then the definition of each placeholder
- [ ] **HELP-02**: `mlpstorage` (bare) or `mlpstorage --help` prints valid next-token options at the current hierarchy level
- [ ] **HELP-03**: At leaf level, `--help` lists flags for that benchmark/command combination

### Test Coverage

- [ ] **TEST-01**: Unit tests for all three parser modes (closed/open/whatif) verify correct subparser structure
- [ ] **TEST-02**: Open-gated args (--loops, --params, --timeseries-interval, etc.) are absent from closed parsers and present in open/whatif parsers
- [ ] **TEST-03**: Model/accelerator choice restrictions verified per mode (closed: [unet3d, retinanet], whatif: full list)
- [ ] **TEST-04**: `mlpstorage version` returns a non-empty string without raising an exception

## Out of Scope

| Feature | Reason |
|---------|--------|
| New benchmark types | Scope is refactoring existing CLI, not adding benchmarks |
| Runtime MPI changes | MPI execution unchanged; only argument parsing refactored |
| Config YAML format changes | YAML configs unchanged; only CLI interface changes |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CLI-01 | Phase 1 | Pending |
| CLI-02 | Phase 1 | Pending |
| CLI-03 | Phase 1 | Pending |
| CLI-04 | Phase 1 | Complete |
| CLI-05 | Phase 1 | Pending |
| CLI-06 | Phase 1 | Pending |
| CLI-07 | Phase 1 | Complete |
| CLI-08 | Phase 1 | Complete |
| CLI-09 | Phase 1 | Pending |
| CLI-10 | Phase 1 | Complete |
| VER-01 | Phase 2 | Pending |
| VER-02 | Phase 2 | Complete |
| VER-03 | Phase 2 | Pending |
| HELP-01 | Phase 3 | Pending |
| HELP-02 | Phase 3 | Pending |
| HELP-03 | Phase 3 | Pending |
| TEST-01 | Phase 4 | Pending |
| TEST-02 | Phase 4 | Pending |
| TEST-03 | Phase 4 | Pending |
| TEST-04 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 20 total
- Mapped to phases: 20
- Unmapped: 0 ✓

---
*Requirements defined: 2026-06-07*
*Last updated: 2026-06-07 after initial definition*
