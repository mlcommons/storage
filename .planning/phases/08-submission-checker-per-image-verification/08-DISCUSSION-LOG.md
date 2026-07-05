# Phase 8: Submission-checker per-image verification - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-05
**Phase:** 8-Submission-checker per-image verification
**Areas discussed:** STRUCT-06 fate + layout compat, Pool dirs at submission root, CHECK-05 multi-version checksum, Edge cases

---

## STRUCT-06 fate + layout compat

### Q1: Does mlpstorage validate need to handle v1.0 (pre-migration) trees after Phase 8 ships?

| Option | Description | Selected |
|--------|-------------|----------|
| v1.1 only — replace STRUCT-06 | STRUCT-06 replaced by pool checks; v1.0 trees fail CHECK-04 with 'migrate first'. | ✓ |
| Dual-mode — detect layout | Auto-detect via sentinel; run old checks for v1.0, new checks for v1.1. | |
| v1.0 only until user migrates | Phase 8 adds pool checks alongside unchanged v1.0 path. | |

**User's choice:** v1.1 only — replace STRUCT-06

---

### Q2: When mlpstorage validate receives a v1.0 tree, what should it say?

| Option | Description | Selected |
|--------|-------------|----------|
| CHECK-04 fail: 'migrate first' | Fail with: "Legacy code/ layout detected at <path>. Run mlpstorage to auto-migrate before revalidating." | ✓ |
| Silent pass on legacy dirs | Pool checks don't look for pool structure; v1.0 tree passes CHECK-01..03, fails CHECK-04. | |
| Neutral structural error | Flag legacy code/ as unexpected directory without prescribing migration. | |

**User's choice:** CHECK-04 fail: 'migrate first'

---

### Q3: Where do the new pool checks (CHECK-01..04) live in the checker flow?

| Option | Description | Selected |
|--------|-------------|----------|
| Pre-loop, alongside SubmissionStructureCheck | New methods (or new class) running before the per-benchmark loop. CHECK-05 in TrainingCheck/VdbCheck. | ✓ |
| All inside SubmissionStructureCheck | All new @rule methods on existing SubmissionStructureCheck. | |
| New standalone PoolCheck class | New checks/pool_checks.py; registered in run(). | |

**User's choice:** Pre-loop, alongside SubmissionStructureCheck

---

## Pool dirs at submission root

### Q1: How should the checker identify <orgname>/ pool directories?

| Option | Description | Selected |
|--------|-------------|----------|
| Sentinel file: dir with .mlps-image-pool = pool root | Any top-level dir containing .mlps-image-pool is a pool root. Others (not closed/open/systems, not dot-prefixed) are structural errors. | ✓ |
| Suffix pattern: dir containing code-*/ = pool root | Any top-level dir with code-<hash8>/ subdirs is a pool root. | |
| Explicit --org-name flag required | Checker requires --org-name to locate pool. | |

**User's choice:** Sentinel file: dir with .mlps-image-pool = pool root

---

### Q2: Does STRUCT-06's replacement verify that every org has a corresponding pool root?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — missing pool root = CHECK-01 fail | closed/<orgname>/ exists but no pool root → one structural error per org. | ✓ |
| No — check per-leaf only | Each run leaf's pointer resolution fails individually. | |
| Warning only — no sentinel = warn, not fail | Missing sentinel is advisory. | |

**User's choice:** Yes — missing pool root = CHECK-01 fail

---

### Q3: What does top_level_subdirectories_check do with an unrecognized top-level directory?

| Option | Description | Selected |
|--------|-------------|----------|
| Flag as unexpected structural error | Not closed/open/systems and no .mlps-image-pool → violation. Dot-prefixed silently skipped. | ✓ |
| Silently skip dotfiles only, flag everything else | Same as above (dotfile-skip already exists at systems check). | |
| Warn but don't fail | Unrecognized dirs trigger warning only. | |

**User's choice:** Flag as unexpected structural error

---

## CHECK-05 multi-version checksum

### Q1: Which mlpstorage_version determines the expected reference checksum?

| Option | Description | Selected |
|--------|-------------|----------|
| Per-image version from .code-hash.json | Look up REFERENCE_CHECKSUMS[image.mlpstorage_version]. Each run verified against its own image's version. | ✓ |
| Global version from --version flag | All runs checked against the same checksum. Doesn't support multi-version campaigns. | |
| Strictest: all images must share one version | >1 version in pool = fail before any per-image check. | |

**User's choice:** Per-image version from .code-hash.json

---

### Q2: What happens when a pool image's mlpstorage_version is NOT in REFERENCE_CHECKSUMS?

| Option | Description | Selected |
|--------|-------------|----------|
| Warn + skip for CLOSED, pass for OPEN | Mirrors D-12 semantics; self-consistency still runs. | ✓ |
| Fail for CLOSED, skip for OPEN | Unknown version is a hard CLOSED failure. | |
| Always warn, never fail | Unpinned version is always advisory. | |

**User's choice:** Warn + skip for CLOSED, pass for OPEN

---

### Q3: Does --reference-checksum CLI flag stay?

| Option | Description | Selected |
|--------|-------------|----------|
| Stays as per-image override | Overrides REFERENCE_CHECKSUMS for all images if supplied. | |
| Deprecated — per-image lookup only | Remove --reference-checksum; per-image dict lookup is the only path. | ✓ |
| Stays but scoped to specific version | --reference-checksum takes version:checksum pair. | |

**User's choice:** Deprecated — per-image lookup only

---

### Q4: How does TrainingCheck.closed_submission_checksum retarget to pool images?

| Option | Description | Selected |
|--------|-------------|----------|
| Walk to run leaf, read pointer, resolve to pool image | From run leaf, read .mlps-code-image via _read_pointer(), resolve to pool path, call verify + version lookup. | ✓ |
| Pre-resolve in SubmissionStructureCheck, pass paths downstream | Pre-loop builds run-leaf→pool-image map; TrainingCheck reads from it. | |
| Shared helper: resolve_run_pool_image() in helpers.py | New helper extracted to helpers.py; TrainingCheck and VdbCheck call it. | |

**User's choice:** Walk to run leaf, read pointer, resolve to pool image
**Notes:** Planner discretion to extract a `resolve_run_pool_image` helper if the inline logic exceeds ~15 lines.

---

## Edge cases

### Q1: Sentinel present, zero pool images?

| Option | Description | Selected |
|--------|-------------|----------|
| Warn: empty pool is suspicious, not a hard fail | Emit one warning per org. Empty pool after migration from an already-empty v1.0 tree is valid. | ✓ |
| Pass silently | No images = nothing to check. | |
| Fail: sentinel without images is corrupt state | Sentinel with zero images = incomplete migration. | |

**User's choice:** Warn: empty pool is suspicious, not a hard fail

---

### Q2: Pool images present, no sentinel?

| Option | Description | Selected |
|--------|-------------|----------|
| Fail: no sentinel = migration incomplete, migrate first | pool dirs present + no sentinel = partial migration crash. Fail with specific message. | ✓ |
| Treat as v1.1 anyway, skip sentinel check | Sentinel is informational; presence of code-<hash8>/ implies v1.1. | |
| Fail via CHECK-04 (legacy code/ check) | Indirect: CHECK-04 finds legacy code/ dirs if step 3 didn't complete. | |

**User's choice:** Fail: no sentinel = migration incomplete, migrate first

---

### Q3: CHECK-03 orphan detection scope?

| Option | Description | Selected |
|--------|-------------|----------|
| Unreferenced across entire submission root | Union of all run leaves under closed/ AND open/ for the org. | ✓ |
| Unreferenced within each division separately | Check orphans per-division; same net result (cross-mode dedup via D-64). | |
| Orphan = warn, not fail | Unreferenced images are advisory, not blocking. | |

**User's choice:** Unreferenced across the entire submission root

---

### Q4: Missing pointer file vs. dangling pointer — same rule or different?

| Option | Description | Selected |
|--------|-------------|----------|
| Same error class, different message | Both CHECK-01 failures; per-case message is the diagnostic. | ✓ |
| Missing pointer = STRUCT violation, dangling = CHECK-01 | Two rule IDs for two failure modes. | |
| Both fail via PointerMalformed exception | No sub-classification; catch + log. | |

**User's choice:** Same error class, different message

---

## Claude's Discretion

- Whether CHECK-01..04 go into `SubmissionStructureCheck` or a new `PoolStructureCheck` class (based on file size)
- Whether `resolve_run_pool_image` is extracted to `helpers.py` or inlined (extract if >~15 lines)
- D-87 warning dedup implementation (set tracking or pre-loop emission)
- `REFERENCE_CHECKSUMS` lookup mechanics (per-leaf or pre-built map)

## Deferred Ideas

- Checkpointing/KVCache §N.6.1 reference-checksum checks (not in Phase 8 scope)
- `mlpstorage code-image list` / `gc` ergonomics (whole-milestone out-of-scope)
- Cross-org orphan detection
- PointerMalformed sub-classification into missing-vs-dangling-vs-malformed rule IDs
