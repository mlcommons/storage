# Phase 8: Submission-checker per-image verification - Context

**Gathered:** 2026-07-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 8 wires `mlpstorage validate` to understand the v1.1 pool layout. After this phase ships, running `mlpstorage validate --input <results_dir>` against a v1.1 submission tree:

- Verifies every run leaf has a `.mlps-code-image` pointer that resolves to a real pool image (CHECK-01)
- Verifies each pool image's directory name matches its `.code-hash.json.hash` and contents re-hash to that value (CHECK-02)
- Verifies every pool image is referenced by ≥1 run leaf — no orphans (CHECK-03)
- Verifies no legacy unhashed `code/` directory exists anywhere — migration is assumed complete (CHECK-04)
- Runs §3.6.1/§5.6.1 reference-checksum verification against the SPECIFIC pool image each run leaf points at, using that image's recorded `mlpstorage_version` (CHECK-05)

The checker is v1.1-only. A v1.0 tree (legacy `code/` dirs, no sentinel) fails CHECK-04 with a "migrate first" message. No dual-mode or backward-compat v1.0 path is added.

**Requirements delivered (5):** CHECK-01, CHECK-02, CHECK-03, CHECK-04, CHECK-05

### In scope

- New pool-check logic (CHECK-01..04) wired as pre-loop checks in `main.py`'s `run()` function, structured as new `@rule`-decorated methods inside `SubmissionStructureCheck` (or a new `PoolStructureCheck` class — planner picks based on file size and cohesion after inspecting what fits).
- CHECK-05: retarget `TrainingCheck.closed_submission_checksum` (§3.6.1) and `VdbCheck.vdb_closed_submission_checksum` (§5.6.1) to walk to the run leaf, read `.mlps-code-image`, resolve to pool image path, and verify against `REFERENCE_CHECKSUMS[mlpstorage_version]` from that image's `.code-hash.json`.
- Replace STRUCT-06 (`code_directory_contents_check`) with pool-aware equivalent. The existing walk to `<root>/closed/<orgname>/code/` is retired.
- Deprecate `--reference-checksum` CLI flag from `mlpstorage_py/submission_checker/main.py`. Per-image version lookup is the only path.
- Update `top_level_subdirectories_check`: top-level dirs containing `.mlps-image-pool` are recognized pool roots; non-sentinel, non-`closed`/`open`/`systems`, non-dot-prefixed top-level dirs are structural errors.
- Define behavior for edge cases: sentinel-present/pool-empty (warn); pool-images-present/no-sentinel (fail: partial migration); CHECK-01 missing pointer vs dangling pointer (same rule, different messages).
- Test coverage matching Phase 8 ROADMAP success criteria (5 observable behaviors).

### Out of scope (Phase 8)

- **v1.0 backward-compat check path** — Phase 8 is v1.1-only. v1.0 trees get CHECK-04 "migrate first" failure; no separate v1.0 validation mode.
- **`mlpstorage code-image list` / `gc` CLI** — out of scope for the whole v1.1 milestone.
- **Changing `.code-hash.json` schema (D-07)** — retained verbatim.
- **Checkpointing / KVCache `code/` checks** — `CheckpointingCheck` and `KVCacheCheck` have no §N.6.1 reference-checksum rule today; Phase 8 does not add one.
- **`--reference-checksum` migration guide or deprecation warning in help text** — that's a UX/docs concern outside Phase 8's scope; simple removal of the flag and `reference_checksum_override` plumbing is sufficient.

</domain>

<decisions>
## Implementation Decisions

Phase 8 carries forward locked decisions D-1..D-74 from Phases 1..7 verbatim. The decisions below (D-80..D-93) are Phase 8 additions.

### STRUCT-06 fate + v1.0/v1.1 coexistence

- **D-80 — v1.1-only checker; STRUCT-06 replaced.** `SubmissionStructureCheck.code_directory_contents_check` (STRUCT-06) is removed/replaced by the new pool-check methods. The existing `code/`-directory walk logic (VALS-01 for missing `code/`, VALS-02 for hash mismatch) is retired. Phase 8 assumes Phase 7 migration already ran; the checker does not maintain a dual-mode path for v1.0 trees.

  **Rationale:** The ROADMAP phase dependency is explicit: "assumes the v1.1 layout is the ONLY layout at check time (guaranteed by Phase 7's migration)." Keeping STRUCT-06 alongside pool checks would create a contradictory state for v1.1 trees (VALS-01 "missing code/" fires for every valid v1.1 run leaf). Removing STRUCT-06 is a clean break that matches the phase design intent.

- **D-81 — v1.0 trees fail CHECK-04 with an actionable 'migrate first' message.** When CHECK-04's walk finds any directory literally named `code` anywhere under the submission root (the D-63 sentinel pattern), it logs a CHECK-04 violation with the message: `"Legacy code/ layout detected at {path}. Run mlpstorage against this results directory to auto-migrate before revalidating."` The first offending path is named; a count of remaining offenders is appended if N > 1.

  **Rationale:** Actionable error > generic structural rejection. The message names the migration trigger (run mlpstorage) so reviewers can forward it to submitters without needing to know the internals.

- **D-82 — CHECK-01..04 live as pre-loop checks in `main.py:run()`.** New pool-check methods run before the `for logs in loader.load()` loop, the same pattern as `SubmissionStructureCheck` and `SystemYamlSchemaCheck`. Failures are accumulated into `errors` but do NOT short-circuit the per-benchmark loop. Planner picks: add methods to `SubmissionStructureCheck` directly (one class, more methods) or a new `PoolStructureCheck` class (new file, cleaner separation). Both patterns exist in the codebase — planner picks based on post-Phase-8 class size.

  CHECK-05 retargeting lives in `TrainingCheck` and `VdbCheck` (per-run, inside the loop) per D-89.

### Pool root detection at submission root

- **D-83 — Pool root identified by `.mlps-image-pool` sentinel in a top-level directory.** Any top-level directory under `--input` that contains `.mlps-image-pool` is a pool root for that org. `top_level_subdirectories_check` permits these directories. Top-level directories that are NOT one of `{closed, open, systems}` AND do NOT contain `.mlps-image-pool` AND are NOT dot-prefixed → structural error (unexpected entry).

  **Rationale:** Self-describing and sentinel-driven. Org names are user-controlled strings; keying on the sentinel avoids any hardcoded name logic. Aligns with D-72's sentinel design (the sentinel is the machine-readable "migration complete" marker).

- **D-84 — Missing pool root for a known org = CHECK-01 structural failure.** If `closed/<orgname>/` or `open/<orgname>/` exists but `<orgname>/.mlps-image-pool` is absent at the top level, the checker fails CHECK-01 with a single structural error: `"No pool root found for org <orgname>: missing <results_dir>/<orgname>/.mlps-image-pool. Run mlpstorage to migrate."` One error per org, not one per run leaf — avoids O(N) noise for an entirely unmigrated org.

- **D-85 — Unrecognized top-level dirs are structural errors; dot-prefixed entries are skipped.** Mirrors the existing dotfile-skip at `systems_directory_files_check` line 552: any top-level entry whose name starts with `.` is silently skipped. All others that are not `closed`, `open`, `systems`, or a recognized pool root (per D-83) → structural violation.

### CHECK-05 per-image reference checksum

- **D-86 — Per-image `mlpstorage_version` lookup replaces single `get_reference_checksum()`.** When CHECK-05 runs for a run leaf, it reads the pool image's `.code-hash.json.mlpstorage_version` and looks up `REFERENCE_CHECKSUMS[mlpstorage_version]`. If the lookup succeeds, the reference-checksum comparison runs against that image's hash. This correctly handles multi-version submissions (submitter did `git pull` mid-campaign) without any per-run override.

- **D-87 — Unknown version (not in `REFERENCE_CHECKSUMS`): warn + skip upstream-identity for CLOSED; pass silently for OPEN.** Mirrors D-12 semantics from the existing STRUCT-06 "not pinned" warning. Exact message for CLOSED: `"mlpstorage_version {v} not in REFERENCE_CHECKSUMS; upstream-identity check skipped (self-consistency still ran)."` Emitted once per pool image (not per run leaf that references it) to avoid N identical warnings for the same image.

  **Rationale:** Consistent with the existing "not configured" warning path. Submitters using custom/pre-release builds are warned, not failed, so the checker is useful during development. CLOSED submissions should use pinned releases — the warning is the signal.

- **D-88 — `--reference-checksum` CLI flag deprecated and removed.** `mlpstorage_py/submission_checker/main.py:get_args()` removes the `--reference-checksum` argument. `Config.__init__`'s `reference_checksum_override` parameter and `get_reference_checksum()`'s override logic are removed with it. Per-image `REFERENCE_CHECKSUMS` lookup (D-86) is the only path.

  **Rationale:** The flag was designed for v1.0's single-`code/`-per-submitter model. In v1.1, different pool images may carry different versions; a single override checksum is semantically ambiguous. Removing the flag keeps the v1.1 API clean. Any CI script using `--reference-checksum` needs to be updated (a reviewer decision, not an mlpstorage decision).

  **Breaking change:** document as a breaking change in the Phase 8 plan or commit message.

- **D-89 — `TrainingCheck.closed_submission_checksum` and `VdbCheck.vdb_closed_submission_checksum` walk to the run leaf, read `.mlps-code-image`, resolve to pool image, then run self-consistency + version-keyed checksum lookup.** Current walk-up (4 levels from `self.path` to `<root>/closed/<orgname>/code/`) is replaced. New flow:
    1. From `self.path` (the run's leaf directory: `<root>/<mode>/<orgname>/results/<sys>/<type>/<model>/<cmd>/<datetime>/`), read `.mlps-code-image` via `_read_pointer(run_leaf, log)`.
    2. Resolve pool image path: `<results_dir>/<orgname>/` + `_pool_dir_name(full_hash)` (D-62: first 8 hex chars).
    3. Call `verify_image_self_consistent(pool_image_path, log)` — self-consistency.
    4. Read pool image's `.code-hash.json.mlpstorage_version` via `_read_hash_file`.
    5. Look up `REFERENCE_CHECKSUMS[mlpstorage_version]` per D-86/D-87.
    6. If expected is not None: compute `compute_code_tree_md5(pool_image_path)` and compare.

  Missing `code/` is NOT re-logged here (D-84 owns the structural sentinel check; the per-run walk only fires if a run leaf is found, which implies pool root was detected). If `_read_pointer` raises `FileNotFoundError` (no `.mlps-code-image`), that is caught and logged as a CHECK-01 violation (D-93).

  Planner should consider extracting steps 1-4 into a `resolve_run_pool_image(run_leaf, results_dir, orgname, log) -> tuple[Path, dict]` helper in `helpers.py` so both `TrainingCheck` and `VdbCheck` share the lookup without duplication. This is planner discretion.

### Edge cases

- **D-90 — Sentinel-present but pool-empty: warn, don't fail.** If `<orgname>/.mlps-image-pool` exists but no `code-<hash8>/` subdirectory is found under `<results_dir>/<orgname>/`, emit one warning per org: `"Pool sentinel present for <orgname> but no pool images found — nothing to verify."` Not a failure: an org that migrated from an already-empty v1.0 tree (no prior runs) is a valid state.

- **D-91 — Pool images present but no sentinel: fail as partial migration.** If `code-<hash8>/` dirs exist under a top-level dir but `.mlps-image-pool` is absent, that is a partial-migration state (crash between D-71 step 3 and step 4). Fail with: `"Partial migration detected for org <orgname> (pool images found but .mlps-image-pool sentinel absent). Run mlpstorage to complete migration."` This fires as a CHECK-04 violation — the sentinel's absence means migration isn't declared complete.

  **Rationale:** D-71's step order (materialize → pointers → delete legacy → sentinel) means this state has pool images but may still have legacy `code/` dirs (deletion is step 3). CHECK-04 would find the legacy dirs anyway; D-91 adds a MORE SPECIFIC error when only pool images are present without the sentinel — a rarer but possible state (step 3 deleted legacy dirs but process crashed before step 4).

- **D-92 — CHECK-03 orphan detection scope: union of all run leaves across closed/ AND open/ for the org.** Walk all `<datetime>/` leaves under `<root>/closed/<orgname>/results/.../` and `<root>/open/<orgname>/results/.../`. Collect every unique full hash from `.mlps-code-image` files. Any `code-<hash8>/` in the pool whose hash (first 8 chars) is not in that set = orphan → CHECK-03 violation naming the pool image path. Cross-division dedup (D-64) means a pool image referenced by EITHER closed OR open is not an orphan.

- **D-93 — Missing pointer file and dangling pointer are both CHECK-01 failures; same rule ID, different messages.** Missing `.mlps-code-image` in a run leaf: `"run leaf {path} has no .mlps-code-image pointer."` Dangling pointer (file exists, hash references a non-existent pool image): `"run leaf {path} .mlps-code-image references hash {hash8} but code-{hash8}/ not found in pool."` Both under CHECK-01's rule ID and rule name. The per-case message is the diagnostic; no sub-classification into separate rule IDs.

### Claude's Discretion

- Whether CHECK-01..04 go into `SubmissionStructureCheck` as new `@rule` methods or into a new `PoolStructureCheck` class. Planner picks based on class size after inspecting `submission_structure_checks.py` line count post-D-80 (STRUCT-06 removal).
- Whether the "one warning per pool image" dedup in D-87 is tracked with a set or by emitting the warning in the pool-check pre-loop step (not inside the per-run CHECK-05 loop). Planner picks.
- Whether `resolve_run_pool_image` is extracted to `helpers.py` (shared by TrainingCheck + VdbCheck) or inlined in each check. Planner picks — extract if the inline logic exceeds ~15 lines.
- Exact `REFERENCE_CHECKSUMS` key lookup mechanics — whether `mlpstorage_version` is read once per pool image (pre-loop check building a version→checksum map) or per run leaf invocation of CHECK-05. Planner picks based on readability vs. performance tradeoff.
- How to handle a pool image whose `.code-hash.json` is missing or malformed at CHECK-02 — `MissingHashFile` / `MalformedHashFile` exceptions are already typed; catch + log as CHECK-02 violation, consistent with STRUCT-06's existing exception handling pattern.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Design / spec

- `.planning/PROJECT.md` — project overview; Current Milestone v1.1 section; constraints
- `.planning/REQUIREMENTS.md` — CHECK-01..05 requirement text; traceability table
- `.planning/ROADMAP.md` — Phase 8 goal, success criteria (5 observable behaviors), Phase 8 dependencies on Phases 6 and 7
- **Issue #651 design comment** — https://github.com/mlcommons/storage/issues/651#issuecomment-4871997634 (2026-07-03) — reference design; "Submission-checker per-image verification" section describes PR 2 scope
- `.planning/phases/06-content-addressed-pool-capture-or-verify-rewrite/06-CONTEXT.md` — locked decisions D-60..D-67 (pool layout, pointer format, directory suffix, cross-mode dedup, concurrency, atomic writes)
- `.planning/phases/07-one-shot-legacy-migration-hand-edit-detection/07-CONTEXT.md` — locked decisions D-70..D-74 (migration trigger, crash-safety, sentinel format, hand-edit detection, user-facing output)

### Submission checker structure (primary edit surface)

- `mlpstorage_py/submission_checker/main.py` — `run()` entry; `MODE_TO_CHECKERS` dict; `get_args()` where `--reference-checksum` is removed (D-88); pre-loop check registration pattern
- `mlpstorage_py/submission_checker/checks/submission_structure_checks.py` — `SubmissionStructureCheck`; `code_directory_contents_check` (STRUCT-06, line 425) — to be replaced; `top_level_subdirectories_check` (line 248) — to be updated for pool root recognition (D-83/D-85); `_iter_submitter_dirs` — reused by pool-check walk
- `mlpstorage_py/submission_checker/checks/helpers.py` — `_check_code_image_layered` (line 238) — retargeted to accept pool image path instead of legacy `code/` path; reused by D-89 CHECK-05 flow
- `mlpstorage_py/submission_checker/checks/training_checks.py` — `closed_submission_checksum` (§3.6.1, line 660) — retargeted per D-89
- `mlpstorage_py/submission_checker/checks/vdb_checks.py` — `vdb_closed_submission_checksum` (§5.6.1, line 749) — retargeted per D-89
- `mlpstorage_py/submission_checker/checks/base.py` — `BaseCheck`, `@rule` decorator, `log_violation`, `warn_violation` — patterns for new check methods

### Pool / pointer tools (to consume in Phase 8)

- `mlpstorage_py/submission_checker/tools/code_image.py`:
    - `_read_pointer(run_leaf, log) -> tuple[str, str]` — reads and validates `.mlps-code-image` (line 629); Phase 8's CHECK-01 and D-89 consume this
    - `verify_image_self_consistent(image_dir, log) -> bool` — per-image self-consistency (line 403); CHECK-02 and D-89 consume this
    - `_read_hash_file(image_dir, log) -> dict` — reads `.code-hash.json`; provides `mlpstorage_version` for D-86/D-87
    - `_pool_dir_name(full_hash) -> str` — `code-<hash8>/` directory name from full hash (line 567); D-89 uses to resolve pool image path
    - `_find_matching_pool_image(org_root, live_hash, log) -> Path | None` — pool scan helper (line 714); CHECK-01 dangling-pointer check may use this
    - `MissingHashFile`, `MalformedHashFile`, `PointerMalformed`, `CodeImageError` — exception types for CHECK-01/02 error handling
    - `LegacyLayoutDetected` (line 141) — still the exception raised by `capture_or_verify_code_image`; CHECK-04's legacy scan uses a direct filesystem check (walk for dirs named `code`), not this exception — but the exception type and message pattern inform CHECK-04's error wording
- `mlpstorage_py/submission_checker/tools/code_checksum.py:compute_code_tree_md5` — D-89 CHECK-05 upstream-identity check invokes this on the pool image path (same as existing STRUCT-06 usage)

### Configuration

- `mlpstorage_py/submission_checker/configuration/configuration.py` — `Config.get_reference_checksum()` and `reference_checksum_override` — removed by D-88; `REFERENCE_CHECKSUMS` dict is retained (D-86 reads it per-image)
- `mlpstorage_py/submission_checker/constants.py` — `REFERENCE_CHECKSUMS` dict; Phase 8 does NOT add new entries here; it only changes HOW the dict is keyed at lookup time (per D-86: key = pool image's `mlpstorage_version`, not the submission `--version` flag)

### Test context

- `tests/integration/test_pool_*.py` — Phase 6 integration tests; Phase 8 adds parallel `test_submission_checker_pool_*.py` following same shape
- `tests/unit/test_code_image.py` — Phase 6 unit tests; Phase 8 adds unit tests for new CHECK methods
- Phase 8 success criteria from ROADMAP.md (5 observable behaviors) drive the test shape:
    1. Valid v1.1 tree → passes without code-image-related errors
    2. Missing/edited `.mlps-code-image` → CHECK-01 failure naming run + hash
    3. Renamed pool dir or modified pool image → CHECK-02 failure naming image
    4. Orphan pool image OR legacy `code/` dir → CHECK-03 / CHECK-04 failure naming the path
    5. Two runs at two mlpstorage versions → CHECK-05 correctly passes both (uses per-image version lookup)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `_read_pointer(run_leaf, log) -> tuple[str, str]` — `code_image.py:629`. Returns `(algorithm, full_hash)`. Phase 8 CHECK-01 and D-89 CHECK-05 consume this directly; raises `PointerMalformed` on bad format, `FileNotFoundError` if absent.
- `verify_image_self_consistent(image_dir, log) -> bool` — `code_image.py:403`. Existing self-consistency check (re-hashes image, compares to `.code-hash.json.hash`). CHECK-02 and D-89 CHECK-05 self-consistency branch use this verbatim.
- `_read_hash_file(image_dir, log) -> dict` — `code_image.py:511`. Returns the parsed `.code-hash.json` dict (keys: `hash`, `algorithm`, `captured_at`, `mlpstorage_version`, `git_sha`). D-86/D-87 read `mlpstorage_version` from it; D-89 also reads `hash` for cross-verification.
- `_pool_dir_name(full_hash) -> str` — `code_image.py:567`. Computes `code-<hash8>/` directory name. D-89 uses to reconstruct pool image path from pointer full hash.
- `_check_code_image_layered(code_path, division, expected, log, log_violation_cb, rule_id, rule_name)` — `helpers.py:238`. Phase 8 passes pool image path (not legacy `code/`) as `code_path`. The helper is reusable as-is — it just calls `verify_image_self_consistent` + `compute_code_tree_md5`; it doesn't care what kind of directory it receives, as long as it has a `.code-hash.json`.
- `_iter_submitter_dirs(self)` — `submission_structure_checks.py:148`. Yields `(division, submitter, sub_path)` for every org under `closed/` and `open/`. Reused by CHECK-01 run-leaf walk and CHECK-03 orphan detection.
- `REFERENCE_CHECKSUMS` dict — `constants.py`. Already keyed by version string. D-86 reads it with `mlpstorage_version` from `.code-hash.json`; if the version is not in the dict, D-87's warn-and-skip path fires. The dict itself is unchanged.

### Established Patterns

- `@rule(rule_id, rule_name)` decorator from `base.py` — all `@rule`-decorated methods on a check class automatically emit per-check start/pass/fail status at DEBUG. Phase 8's new CHECK-01..04 methods follow the same pattern.
- Accumulate-don't-abort: check methods set `valid = False` on failure but continue walking. Short-circuit only when a missing anchor (e.g., no `.code-hash.json`) would make a subsequent check semantically contradictory (see STRUCT-06 `hashfile_present` gate). Phase 8 applies the same: missing pool root (D-84) short-circuits per-leaf checks for that org but doesn't abort the run.
- Pre-loop / single-shot checks pattern (`SubmissionStructureCheck`, `SystemYamlSchemaCheck` in `main.py:157-167`) — one instantiation, one call, errors accumulated into `errors[]`. Phase 8 CHECK-01..04 fit this pattern.
- Log-level convention: `log_violation` for hard failures, `warn_violation` for advisory warnings (D-87/D-90 warning cases use `warn_violation`).

### Integration Points

- `mlpstorage_py/submission_checker/main.py:157` — where `SubmissionStructureCheck` is called pre-loop. Phase 8's pool checks register in the same block.
- `mlpstorage_py/submission_checker/main.py:51-64` `MODE_TO_CHECKERS` — Phase 8 does NOT change this dict. CHECK-05 is wired into existing `TrainingCheck` and `VdbCheck`, which are already in `MODE_TO_CHECKERS`.
- `mlpstorage_py/submission_checker/checks/helpers.py:_check_code_image_layered` — Phase 8 passes a pool image path to it from D-89's CHECK-05 flow. If the planner extracts `resolve_run_pool_image` to helpers.py (Claude's Discretion), this is the file to add it to.
- `mlpstorage_py/submission_checker/configuration/configuration.py` — `Config.get_reference_checksum()` removed by D-88. Any remaining callers (STRUCT-06 was the main one, now removed) should be audited; planner must grep for `get_reference_checksum` usages.

</code_context>

<specifics>
## Specific Ideas

- D-81's "migrate first" message should reference `mlpstorage` (not `mlpstorage migrate`) since migration is automatic and there is no `migrate` subcommand (Phase 7 out-of-scope, deferred). The message "Run mlpstorage against this results directory to auto-migrate" is correct.
- D-87's per-pool-image "not pinned" warning should fire ONCE per image (not once per run leaf that references it). Dedup by tracking warned pool image paths in a set within the check loop.
- D-92 orphan detection: the "collect all referenced hashes" step should build a set of FULL 32-hex hashes (from `_read_pointer`), then compare against pool image dirs by looking up `_read_hash_file(pool_dir).hash`. This avoids the 8-char truncation from `_pool_dir_name` and keeps the check collision-resistant.
- CHECK-04's legacy walk should reuse D-63's detection pattern: walk `<root>/{closed,open}/<orgname>/` for any subdirectory LITERALLY named `code` (not `code-<hash8>/`). The `_scan_legacy_layout` function in `code_image.py:679` does exactly this — reuse it rather than re-implementing the walk.

</specifics>

<deferred>
## Deferred Ideas

- **Checkpointing/KVCache §N.6.1 reference-checksum checks** — `CheckpointingCheck` and `KVCacheCheck` have no reference-checksum rule today. Adding pool-image checksum verification to those modes would mirror CHECK-05 for Training/VDB, but it's not in the current Phase 8 requirements (CHECK-05 is scoped to §3.6.1 + §5.6.1). A future phase could add it.
- **`mlpstorage code-image list` / `gc` ergonomics** — out of scope for the whole v1.1 milestone. Phase 8 does not add any CLI surface for pool management.
- **Cross-org orphan check** — CHECK-03 is scoped per org. Two orgs sharing a `--results-dir` each have independent orphan analysis. Cross-org pool sharing is explicitly out of scope (REQUIREMENTS.md "Out of Scope").
- **Partial-pointer file (corrupted `.mlps-code-image`)** — `_read_pointer` raises `PointerMalformed` if the file exists but doesn't parse as `md5-tree-v2:<32-hex>`. Phase 8 catches `PointerMalformed` and logs a CHECK-01 violation (falls under D-93's "same error class, different message" rule). A specific sub-classification for malformed-vs-missing-vs-dangling was considered but deferred as unnecessary complexity.

</deferred>

---

*Phase: 8-Submission-checker per-image verification*
*Context gathered: 2026-07-05*
