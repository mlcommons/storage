# ROADMAP — Milestone v1.1

**Milestone:** Content-addressed code-image pool (#651)
**Goal:** Replace the current one-image-per-submission-tree code capture with a content-addressed per-org image pool so submitters can upgrade mlpstorage mid-campaign (via `git pull` against an existing `--results-dir`) without invalidating prior runs.
**Reference design:** #651 comment 4871997634 (2026-07-03)
**Previous milestone:** v1.0 ended at Phase 05.2. This milestone continues numbering — v1.1 starts at Phase 6.
**Granularity:** standard (target 4–6 phases). Actual: 3 phases (the requirement set is tightly clustered around two natural PR boundaries; the runtime side splits cleanly into "capture path rewrite" and "one-shot migration"; the submission-checker side is one coherent phase). Adding a 4th "polish" phase would be padding.

## Phases

- [ ] **Phase 6: Content-addressed pool + capture-or-verify rewrite** — Fresh `--results-dir` runs write under `<results_dir>/<orgname>/code-<hash8>/` with a `.mlps-code-image` pointer in each run leaf; hash mismatch means "new image needed," not "reject."
- [ ] **Phase 7: One-shot legacy migration + hand-edit detection** — First run of new mlpstorage against an existing (v1.0-layout) `--results-dir` auto-migrates every legacy `code/` into the pool, gated by a per-org `.mlps-image-pool` sentinel, aborting cleanly on any hand-edited image.
- [ ] **Phase 8: Submission-checker per-image verification** — `mlpstorage validate` (and the submission checker) verifies every pointer resolves, every pool image is self-consistent, no orphan images, no leftover legacy `code/`, and reference-checksum comparisons run against the correct image per run.

## Phase Details

### Phase 6: Content-addressed pool + capture-or-verify rewrite

**Goal:** A submitter running mlpstorage (v1.1) against a fresh `--results-dir` writes results under the new pool layout, and re-running with a modified source captures a second image alongside the first instead of failing.
**Depends on:** Nothing (first phase of milestone; builds on the existing tree-hash and `.code-hash.json` infrastructure from v1.0).
**Requirements:** POOL-01, POOL-02, POOL-03, POOL-04, PTR-01, PTR-02, CAPVER-01, CAPVER-02, CAPVER-03, UX-01
**Success Criteria** (what must be TRUE):

  1. Running `mlpstorage closed training … run …` against an empty `--results-dir` writes the code image to `<results_dir>/<orgname>/code-<hash8>/` (not the legacy `<results_dir>/closed/<orgname>/code/`) and writes `.mlps-code-image` inside the run-leaf timestamp directory before any run output is produced.
  2. Running the same command a second time with the source unchanged produces zero new pool images — the existing `code-<hash8>/` is detected by hash and reused; a new pointer file is written in the new run's leaf.
  3. Running the same command a third time after making a change under `mlpstorage_py/` (e.g. `git pull` upgrading the fork) succeeds, writes a second `code-<newhash8>/` alongside the first, and writes a pointer to the new hash in the new run's leaf. The literal string "changes to the codebase are not allowed in a CLOSED run" does NOT appear in stdout/stderr.
  4. Running an OPEN mode benchmark reuses a pool image already captured under the same org by a prior CLOSED run when the source hash matches (cross-mode dedup): no second `code-<hash8>/` is materialized.
  5. Two different orgs sharing a `--results-dir` (via distinct `MLPSTORAGE_ORGNAME`) each maintain their own `code-<hash8>/` set under `<results_dir>/<org1>/` and `<results_dir>/<org2>/`; images do not mix.

**Plans:** 3/4 plans executed
**Wave 1**

- [x] 06-01-PLAN.md — Pointer file writer/reader + pool dir-name helpers (D-61, D-62, PTR-01/02, POOL-01/02)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 06-02-PLAN.md — Rewrite `capture_or_verify_code_image` for pool + pointer; refuse legacy; retire reject strings (CAPVER-01/02/03, POOL-01/02/03/04, PTR-01, UX-01, D-63/D-64/D-65/D-66/D-67)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 06-03-PLAN.md — Retire legacy `results_dir/code_image.py` module + call site + tests (D-60)
- [x] 06-04-PLAN.md — Integration coverage for 5 ROADMAP success criteria + D-66 concurrency (POOL-04)

### Phase 7: One-shot legacy migration + hand-edit detection

**Goal:** A submitter with a v1.0-layout `--results-dir` (containing one or more legacy `.../{closed|open}/<orgname>/.../code/` trees) runs v1.1 and observes an automatic, idempotent migration that leaves prior runs valid, or a clean abort if any legacy image was hand-edited.
**Depends on:** Phase 6 (pool layout + pointer write semantics must exist before migration can populate them).
**Requirements:** MIG-01, MIG-02, MIG-03
**Success Criteria** (what must be TRUE):

  1. Running any `mlpstorage <mode> … run|datagen|datasize …` command against an existing v1.0-layout `--results-dir` (no `.mlps-image-pool` sentinel present) triggers migration: every legacy `code/` directory under `.../{closed|open}/<orgname>/...` is discovered, hashed, materialized as `<results_dir>/<orgname>/code-<hash8>/` (identical hashes dedup to one pool image), a `.mlps-code-image` pointer is written in every existing run leaf, all legacy `code/` directories are deleted, and `<results_dir>/<orgname>/.mlps-image-pool` is written.
  2. Running a second command after migration completes does NOT re-scan or re-migrate — the sentinel short-circuits within milliseconds; observable via absence of migration log lines on the second run.
  3. Simulating a crash mid-migration (e.g. by SIGKILL after some pool images are materialized but before the sentinel is written) leaves the tree in a state where a subsequent invocation resumes cleanly: any already-materialized pool image is re-used (dedup), remaining legacy `code/` trees are discovered, and the sentinel is written on completion. No run leaf ends up without a pointer.
  4. If any legacy `code/` on disk does not re-hash to its own `.code-hash.json.hash` (i.e. was hand-edited after capture), migration aborts before modifying any files, emits an error naming both the offending path and the phrase "hand-edited code image detected," and leaves the `.mlps-image-pool` sentinel absent so the submitter can fix and re-run.

**Plans:** TBD

### Phase 8: Submission-checker per-image verification

**Goal:** A reviewer running `mlpstorage validate` against a v1.1-layout submission tree receives a clear pass/fail result grounded in per-image checks: pointer chains resolve, each pool image is self-consistent, no orphan images exist, no legacy `code/` remains, and reference-checksum verification runs against the specific image each run used.
**Depends on:** Phase 6 (defines the pool layout and pointer format the checks read); Phase 7 (defines the sentinel and post-migration invariants the checks assume). Also assumes the v1.1 layout is the ONLY layout in the submission tree at check time (guaranteed by Phase 7's migration).
**Requirements:** CHECK-01, CHECK-02, CHECK-03, CHECK-04, CHECK-05
**Success Criteria** (what must be TRUE):

  1. Running `mlpstorage validate` on a submission tree in which every run leaf has a `.mlps-code-image` pointer resolving to an existing pool image, and every pool image is self-consistent, passes without emitting a code-image-related error.
  2. Deleting the `.mlps-code-image` file from any single run leaf, or editing it to reference a non-existent hash, causes `mlpstorage validate` to fail with an error that names both the offending run path and the referenced (or missing) hash.
  3. Renaming a pool directory so its suffix no longer matches its `.code-hash.json.hash`, or modifying a file inside a pool image so its contents no longer re-hash to the recorded value, causes `mlpstorage validate` to fail with an image-specific error naming that image.
  4. Placing a `code-<hash8>/` directory in the pool that is not referenced by any run leaf's pointer file causes `mlpstorage validate` to fail with an orphan-image error naming that image; symmetrically, leaving any legacy unhashed `code/` directory anywhere in the submission tree causes `mlpstorage validate` to fail with a specific "legacy layout detected" error.
  5. §3.6.1 / §5.6.1 reference-checksum verification, when it runs for a given run leaf, is scoped to the specific pool image that leaf's pointer resolves to (and to that image's recorded `mlpstorage_version`) — verifiable by a submission tree with two runs at two mlpstorage versions where checksum comparison correctly succeeds for both.

**Plans:** TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 6. Content-addressed pool + capture-or-verify rewrite | 4/4 | Complete |  |
| 7. One-shot legacy migration + hand-edit detection | 0/TBD | Not started | — |
| 8. Submission-checker per-image verification | 0/TBD | Not started | — |

## Coverage Matrix

| REQ-ID | Phase | Category |
|--------|-------|----------|
| POOL-01 | 6 | Pool layout |
| POOL-02 | 6 | Pool layout |
| POOL-03 | 6 | Pool layout |
| POOL-04 | 6 | Pool layout |
| PTR-01 | 6 | Pointer files |
| PTR-02 | 6 | Pointer files |
| CAPVER-01 | 6 | Capture-or-verify |
| CAPVER-02 | 6 | Capture-or-verify |
| CAPVER-03 | 6 | Capture-or-verify |
| UX-01 | 6 | User-facing messages |
| MIG-01 | 7 | Migration |
| MIG-02 | 7 | Migration |
| MIG-03 | 7 | Migration |
| CHECK-01 | 8 | Submission checker |
| CHECK-02 | 8 | Submission checker |
| CHECK-03 | 8 | Submission checker |
| CHECK-04 | 8 | Submission checker |
| CHECK-05 | 8 | Submission checker |

**Coverage:** 18/18 v1 requirements mapped, each to exactly one phase. No orphans. No duplicates.

## PR / Phase Alignment

The reference design comment (#651, 2026-07-03) sequences work as two PRs. This roadmap aligns as follows:

| PR (design comment) | Roadmap phases | Rationale for split |
|---------------------|----------------|---------------------|
| PR 1 — pool layout + one-shot migration + capture-or-verify rewrite | Phase 6 + Phase 7 | The runtime rewrite (capture path) and the one-shot migration have distinct observable outcomes (fresh-tree flow vs. legacy-tree upgrade flow) and distinct failure modes (new-image vs. hand-edit-detected abort). A submitter can validate Phase 6 end-to-end without touching a legacy tree, and Phase 7 has its own crash-safety and abort semantics worth verifying independently. The two phases can still ship as a single PR (PR 1) or as two — the phase boundary is a planning boundary, not a merge boundary. |
| PR 2 — submission-checker per-image verification | Phase 8 | Submission-time checks are a separate change surface (`mlpstorage_py/submission_checker/checks/`), a separate command (`mlpstorage validate`), and land in a separate PR per the reference design. |

## Open Architectural Questions

These are not blockers — the roadmap proceeds. Flag for the user to decide before Phase 6 planning begins.

**Q1: Reconcile the OLDER per-mode capture at `mlpstorage_py/results_dir/code_image.py` with the new pool layout.**

Context: v1.0 (LAY-06 / Rules.md §2.1.6) shipped a second, independent code-capture layer at `mlpstorage_py/results_dir/code_image.py`, invoked from `Benchmark.__init__` (`mlpstorage/benchmarks/base.py:193-200`). It writes to:

- `<results_dir>/closed/<orgname>/code/` (closed mode), or
- `<results_dir>/open/<orgname>/code/<benchmark>/<command>/` (open mode)

This is precisely the legacy layout that v1.1 replaces. Two options:

  (a) **Retire it entirely.** `Benchmark.__init__` no longer calls `capture_code_image`; the CLI-level `capture_or_verify_code_image` (`main.py:224`) becomes the sole capture path. Cleanest, but breaks any consumer that reads `self.code_image_path` on the benchmark object.

  (b) **Make it a no-op.** Keep the function signature but have it return the pool image path (computed from `orgname` + live-source hash) without writing anything, since Phase 6's capture already ran at `main.py:224` before `Benchmark.__init__`. Preserves `self.code_image_path` semantics for downstream readers.

Recommendation: option (a) unless a `self.code_image_path` consumer is found. This question should be resolved during Phase 6 discuss/plan; a grep for `self.code_image_path` across `mlpstorage_py/` will settle it in seconds.

**Q2: Does the `.mlps-code-image` pointer file need to include the algorithm identifier (e.g. `md5-tree-v2:<hash8>`) or just the hash string?**

The requirement (PTR-01) says "plain text, one line, exactly the hash string" — but that couples the pointer format to the current algorithm. A future algorithm rev would require pointer rewrites. Worth confirming during Phase 6 planning whether the plain-hash form is deliberate (accepting the coupling) or an oversight.

**Q3: What is the submission-checker's error path when it discovers a `.mlps-image-pool` sentinel but no pool images (or vice versa)?**

CHECK-04 forbids leftover legacy `code/`, but doesn't specify the reverse: a submission tree with the sentinel but zero `code-<hash8>/` images. This should not happen in practice (migration writes both), but the checker should have a defined behavior. Nail down in Phase 8 planning.

---
*Roadmap drafted 2026-07-04. Awaits user approval; phase details subject to refinement during `/gsd-plan-phase 6`.*
