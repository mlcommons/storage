# MLPerf Storage Benchmark v3.0 — Significant Bugs Fixed by Version

**Date:** 2026-07-11
**Covers:** Versions 3.0.3 through 3.0.42 (May 28 – July 11, 2026)

**Source repositories** — this file cites fixes that live across three repos. Follow the links to read the code, PR discussions, and full commit bodies:

- **mlpstorage**: https://github.com/mlcommons/storage — this repo; the benchmark harness, CLI, rules, and reportgen.
- **DLIO_local_changes**: https://github.com/mlcommons/DLIO_local_changes — the MLPerf Storage fork of DLIO; pinned by `pyproject.toml` and pulled in via the DLIO commit-hash bump on each release.
- **s3dlio**: https://github.com/russfellows/s3dlio — the Rust-backed object-storage engine that DLIO calls into; pinned by version floor in both DLIO and this repo.

---

## Table of Contents

Jump to the group of releases you care about. Only versions divisible by 10 are linked — each link
lands on the first entry in that decade of point-releases.

- [All the 3.0.10's releases](#versions-3010--3013-june-15--18-2026)
- [All the 3.0.20's releases](#version-3020-june-25-2026)
- [All the 3.0.30's releases](#version-3031-july-4-2026) *(3.0.29 and 3.0.30 were not released; the decade starts at 3.0.31)*
- [All the 3.0.40's releases](#version-3040-july-9-2026)

---

## Source Code Size For Versions 3.0.3 (May 28, 2026) and 3.0.40 (July 9, 2026)

| Subtree | 3.0.3 (excl. tests) | 3.0.3 (tests only) | 3.0.40 (excl. tests) | 3.0.40 (tests only) |
|---|---:|---:|---:|---:|
| `mlpstorage_py` | 22,222 | 2,995 | 41,605 | 16,515 |
| `training` + `checkpointing` + `DLIO` | 20,746 | 3,389 | 18,521 | 9,762 |
| `vdb_benchmark` | 11,346 | 5,632 | 15,332 | 7,320 |
| `kv_cache_benchmark` | 6,424 | 4,013 | 6,413 | 4,112 |
| **Subtotal** | **60,738** | **16,029** | **81,871** | **37,709** |
| **Total (all code)** | **76,767** | | **119,580** | |

---

Each section below lists significant bugs fixed in that release, organized into three categories:

- **Score-Affecting** — bugs that alter measured throughput, latency, recall, or pass/fail outcome
- **Invocation** — bugs that prevented the benchmark from starting or completing a run
- **Other Significant** — bugs in submission validation, report generation, or metadata that could cause valid results to appear invalid, or vice versa

Trivial fixes (CLI message wording, whitespace, docs-only changes, test-only changes) are excluded.
Issue and PR numbers are provided at the end of each entry for reference.

Each entry is a single line of 120 characters or fewer, written to help a reader who has already
"finished" running a benchmark decide whether prior runs affected by the bug are worth rerunning
under this version or a later one — so entries lead with the user-visible symptom, not the internals.

---

## Versions 3.0.3 – 3.0.9 (May 28 – June 14, 2026)

The initial v3.0 release went through rapid iteration. These version numbers cover a period of broad
feature stabilization and critical correctness fixes; exact per-version boundaries within this range
are not tracked here.

**Score-Affecting**
- VDB: P50/P90/P99 latencies fabricated when `batch_size > 1` — identical values per batch hid true tail latency. (#399)
- VDB: flat_index ground-truth rebuild path was dead code — recall silently used potentially stale precomputed ground truth. (#399)
- VDB: `row_count()` triggered a full Milvus flush on every call during timed runs, perturbing storage state and distorting throughput. (#399)
- VDB: vector primary keys were non-unique across processes — index quality and recall measurements were corrupted. (#387)

**Invocation**
- unet3d training had three silent hang causes at scale — runs stalled indefinitely with no error message. (#394)
- Checkpointing CLOSED rejected `--num-checkpoints-write`/`read` — mandatory submission parameters could not be set. (#434, #435)
- CLOSED mode rejected `--params` flags and produced incorrect `datasize` output — downstream run configs were wrong. (#433, #437)
- `--mpi-params` rejected any flag starting with a dash (e.g., `--bind-to socket`) — MPI topology tuning was impossible. (#422, #426)
- Multi-node VectorDB orchestration was broken — only single-host datagen and run were functional. (#424)

---

## Versions 3.0.10 – 3.0.13 (June 15 – 18, 2026)

**Score-Affecting**
- KVCache `storage_tokens_processed` undercounted by `tensor_parallel` factor — reported throughput was systematically low. (#402)
- KVCache: tokens reaching a cache tier via eviction excluded from byte counts — cache utilization metric was understated. (#420)
- Memory budget OOM check divided by total workers, not per-node workers — multi-host runs got spurious OOM rejections. (#448)

**Invocation**
- Training crashed with `SemLock FileNotFoundError` when systemd-logind `RemoveIPC=yes` removed shared memory mid-run. (#447, #460)
- S3 `storage.storage_root` was double-prefixed (`s3://s3://`) — all object-storage I/O was broken until this was fixed. (#392, #459)
- KVCache `datasize` and `whatif` crashed with `AttributeError: 'Namespace' has no attribute 'loops'`. (#444, #457)
- DLIO triggered a `sudo` password prompt at large scale, blocking unattended multi-host runs. (#391, #462)
- `dlio_sampler` `ceil(N/size)` caused ranks to deadlock between epochs when `num_samples % comm_size != 0`. (#455, #470)

---

## Version 3.0.14 (June 19, 2026)

**Score-Affecting**
- Training AU from epoch 2+ was inflated: DLIO workers reused cached file shards instead of re-reading storage. (#464, #475)

---

## Version 3.0.15 (June 19, 2026)

**Invocation**
- Large S3 datasets required 12+ hour pre-run file listings; flat filesystems OOM'd — `skip_listing` eliminated this. (#449, #450, #451, #466, #472)

---

## Version 3.0.16 (June 22, 2026)

**Score-Affecting**
- CLOSED VDB and KVCache results were auto-tagged as OPEN — every CLOSED result before this fix is mistagged and will fail submission review. (#488)

---

## Version 3.0.17 (June 24, 2026)

**Score-Affecting**
- RetinaNet CLOSED: tool-injected `skip_listing` flag treated as user override — all CLOSED training runs were marked INVALID. (#494, #496)
- VDB: configured build params ignored — index used engine defaults, not configured values; recall results are from a misconfigured index. (#463, #497)

**Invocation**
- KVCache OPEN: fixed CLOSED params forced onto every run, overriding user YAML/CLI — no OPEN-mode parameter customization worked. (#498, #501)

---

## Version 3.0.18 (June 24, 2026)

**Other Significant**
- Code-image hash was non-deterministic — the same source could verify on one run and fail the next. (#505, #512)
- Code-image captures from before 3.0.18 trigger an opaque `MalformedHashFile` error; recapture with 3.0.18. (#505, #514)

---

## Version 3.0.19 (June 25, 2026)

**Score-Affecting**
- VDB: empty ground-truth collection after Milvus fallback was silently accepted — recall scores were 0 or random. (#489, #516)

**Invocation**
- KVCache `--num-processes` had no effect — always ran one process per host regardless of the flag. (#500, #509)

---

## Version 3.0.20 (June 25, 2026)

**Score-Affecting**
- RetinaNet minimum AU threshold was 90%; spec requires 85% — runs at 85–89% AU were incorrectly rejected; they should pass. (#513, #523)

**Invocation**
- KVCache CLOSED: `--mpi-params` silently ignored — custom MPI topology flags had no effect on closed runs. (#520, #522)
- KVCache: results silently lost when `--results-dir` lacked shared filesystem access — no output, no error. (#521, #524)

**Other Significant**
- `reportgen` crashed when `system_info` was absent; TOOL_INJECTED_PARAMS mismatch caused false INVALID verdicts for auto-injected parameters. (#503, #511)

---

## Versions 3.0.21 – 3.0.22 (June 25 – 26, 2026)

No new mlpstorage-level significant bug fixes in this range. DLIO was updated from the PR #27 pin through PR #36, pulling in the DLIO-layer fixes below (PR #37 is test-only and excluded). The s3dlio floor was also bumped 0.9.100 → 0.9.102, pulling in the s3dlio-layer fixes below.

**Score-Affecting**
- DLIO: slow kernel flush misread as sudo refused — page-cache drops silently disabled for the run, inflating throughput. (#487; DLIO PR #28)
- DLIO: unset S3 endpoint caused s3torchconnector to silently route to real AWS — benchmarks measured the wrong storage. (#472; DLIO PR #32)

**Invocation**
- DLIO: `skip_listing` not copied from Hydra config to `args` — mlpstorage auto-injection had no effect; full object-storage listing always ran. (#504; DLIO PR #30)
- DLIO: failed S3 uploads continued queuing BytesIO payloads until OOM — pipeline did not stop on first upload error. (#504; DLIO PR #31)
- DLIO: PyTorch shm-reap `RuntimeError` from `RemoveIPC=yes` surfaced as bare traceback with no actionable guidance. (#528; DLIO PR #34)
- DLIO: flat-directory listing double-allocated all URIs — ~8.8 GB peak RAM on rank 0 before training started at 50 M files. (#466; DLIO PR #35)
- DLIO: `direct://` and `file://` schemes skipped path-existence and writability checks — misconfigured paths not caught before ranks started. (#507; DLIO PR #36)
- s3dlio: documented `S3DLIO_CONNECT_TIMEOUT_SECS` was never read — actual connect ceiling was 5 s (SDK layer shadowed the 10 s transport). Cold-start `dispatch failure` at large-scale S3 warmup traced to this; now wired end-to-end, default raised 10 s → 20 s. (#506; s3dlio PR #144)

**Other Significant**
- DLIO: drop_caches timeout warnings silenced after first occurrence — per-epoch retry status not visible in logs. (#487; DLIO PR #29)
- s3dlio: cold-start `RuntimeError: dispatch failure` was opaque — 22 S3 call sites now surface the full SDK error chain (endpoint, TCP error, connect refused) instead of a bare message. New `S3DLIO_MAX_RETRY_ATTEMPTS` (default 3) enables fast-fail (`=1`) for cold-start diagnosis or extended retries (`5+`) for flaky links. (#506; s3dlio PR #144)

---

## Version 3.0.23 (June 26, 2026)

**Score-Affecting**
- KVCache `metadata['parameters']` always empty — `reportgen` reported "Missing model parameter" for every KVCache run. (#537, #539)
- VDB: CLI flags `--collection`, `--dimension`, `--num-vectors` ignored when YAML config present — CLI had no effect. (#541, #542)

**Invocation**
- `--o-direct` set `storage_type=s3` instead of `direct_fs` — runs used the S3 I/O library on local storage. (#538, #544)
- Checkpointing `--o-direct`: relative `checkpoint_folder` resolved incorrectly — checkpoint writes always failed. (#536, #540)
- `IndentationError` in kvcache module blocked all kvcache imports — mlpstorage kvcache was completely non-functional. (#545, #547)

---

## Versions 3.0.24 – 3.0.25 (June 28, 2026)

No new mlpstorage-level significant bug fixes in this range. DLIO was updated from the PR #36 pin to include PRs #37 and #38; PR #37 is test-only and excluded.

**Invocation**
- DLIO: no `direct_fs` storage type existed — `--o-direct` routed reads and writes to the S3 library on local storage, using the wrong I/O path. (#538; DLIO PR #38)

---

## Version 3.0.26 (June 29, 2026)

**Score-Affecting**
- Object-mode detection broken for `data_access_protocol='object'` — S3 runs failed with E401 or bad checkpoint parent checks. (#584, #585)
- Checkpoint folder double-prefixed (`s3://s3://`) in object-storage mode — checkpoint writes and reads all failed. (#583, #586)

**Invocation**
- CAP-01 called `statvfs()` on an `s3://` URI in S3 mode — every S3 run crashed before benchmark work started. (#568, #579)
- `BenchmarkVerifier` ran in `whatif` mode, producing spurious failures — whatif was supposed to skip all validation. (#571, #580)
- KVCache `_print_summary()` crashed with `NameError` on `cache_stats` — every kvcache run ended with a traceback. (#570, #576)
- CAP-02 rejected DAOS/FUSE via `st_dev` comparison — genuine shared-FS filesystems failed the probe on multi-host runs. (#566, #577)
- Multi-host CAP-02 staging overwrote data from all but the last host — probe results from earlier hosts were lost. (#569, #577)
- Missing workload YAML surfaced as a cryptic `ValueError` — no indication of which model names are supported. (#517, #563)

**Other Significant**
- CAP-02 result exceeded `PIPE_BUF` at >4096 ranks — truncated stdout caused "payload unreadable" at large scale. (#573, #587)

---

## Version 3.0.27 (July 1, 2026)

**Score-Affecting**
- Rule 3.3.1 used a placeholder, not the real datasize — `num_files_train` cross-check with datasize was always a no-op. (#608, #611)

**Invocation**
- MPI collector hardcoded `python3` — in a virtualenv, mpi4py import failed and system-info collection was silently skipped. (#594, #597)
- S3 storage env vars not forwarded to remote MPI ranks — multi-host S3 runs were uncredentialed on worker nodes. (#592, #596)

**Other Significant**
- `reportgen` used stale metadata keys and failed on v3.0 layout — all reports generated before 3.0.27 are incorrect. (#599, #600, #601, #610)
- Rule 2.1.14 required training-run JSON from datagen dirs (never present) — datagen submissions always failed this check. (#600, #604)
- KVCache and vectordb submissions always failed: checker used wrong directory names for these workloads. (#612, #614)

---

## Version 3.0.28 (July 2, 2026)

**Score-Affecting**
- Multipart S3 uploads silently dropped parts at scale — objects stored short, causing corrupt training reads. (#593; DLIO PR #41)
- Checkpointing reportgen: read missing `end_datetime` key — all checkpointing runs showed INVALID invocation structure. (#618, #632)
- CAP-01 demanded free disk during training even when dataset already existed — full-disk runs were blocked. (#627, #633)

**Other Significant**
- Submission checker used v1.0 metadata keys on v3.0 data — all parameter extraction and rule checks were wrong. (#606)
- `reportgen` generated root-level not per-model `results.json` — per-model submission check always failed. (#620)
- Drive type collected as lowercase `sata`/`sas`; schema expects uppercase — disk-system validation always failed. (#637, #639)
- KVCache run summary misnamed — submission checker expected `summary.json`; all KVCache submissions failed. (#638, #640)
- `prefetch_window` rejected as invalid in OPEN mode — OPEN submissions using this tunable were incorrectly flagged. (#629, #634)

---

## Version 3.0.31 (July 4, 2026)

*Note: versions 3.0.29 and 3.0.30 were not released; the version incremented directly from 3.0.28 to 3.0.31.*

**Score-Affecting**
- DiskANN: configured index params ignored (wrong key casing) — all DiskANN VDB results used default index settings. (#590, #670)
- CLOSED KVCache: 20 s inter-option delay used, 90 s required — prior "passing" runs may not meet spec. (#664, #665)
- Checkpoint read dropped `s3://` URI scheme — object-storage checkpoint reads always returned file-not-found errors. (#641, #649)
- Object-storage checkpoint writer deadlocked after `fork()` — Tokio mutex corruption stalled all ranks at 0% CPU. (#642, #650)
- CAP-01 subset disk check used full model size — subset runs required far more free space than actually needed. (#644, #654)

**Invocation**
- Volatile OMPI/NCCL env vars leaked into the system fingerprint — spurious `SystemDriftError` between identical runs. (#643, #647, #648, #652, #672)

---

## Version 3.0.32 (July 4 – 5, 2026)

**Invocation**
- All-ranks checkpoint LIST in DLIO caused throttling — reads of intact checkpoints aborted at scale. (#667; DLIO PR #42)

---

## Version 3.0.33 (July 5, 2026)

**Score-Affecting**
- Rule 3.1.2 double-counted host memory, doubling required dataset size — valid submissions may have been rejected. (#669, #675)
- Checkpointing `num_hosts` = product of all ranks, not actual host count — e.g., 16 hosts reported as 961; capacity check wrong. (#671, #675)

**Other Significant**
- Training `results.json` written inside `run/` subdirectory but submission checker expected it one level up. (#680, #683)
- s3dlio: opt-in HEAD-verify + retry for object writes now available via `S3DLIO_PUT_VERIFY` and `S3DLIO_MPU_PUT_VERIFY` (both default off) — enables per-run detection of silent write truncation on any network backend, complementing the DLIO PR #41 obj_store_lib fix from 3.0.28. s3dlio floor bumped 0.9.102 → 0.9.106; v0.9.104's always-on verification was reverted to opt-in in v0.9.106 to avoid the always-on HEAD-per-PUT throughput cost. (#593; s3dlio PR #145, PR #147)

---

## Version 3.0.34 (July 6, 2026)

**Score-Affecting**
- Rule 3.3.1 threw `TypeError` on int vs str comparison — training data file-count validation was always skipped. (#681, #684)

**Invocation**
- gRPC 256 MB default message limit caused `RESOURCE_EXHAUSTED` for large VDB datasets — large-scale VDB runs failed. (#572, #685)

---

## Version 3.0.35 (July 7, 2026)

**Score-Affecting**
- Checkpoint-save throughput ~41% lower on `file`/`local_fs` backends — unconditional `forkserver` start method lost copy-on-write warm state and CPU/NUMA locality vs `fork`. (#682, #700; DLIO PR #44 unaffected — write path was already correct)
- Checkpoint datagen under-threaded on multi-node runs — dgen-py thread count divided per-node CPU count by global rank count; e.g. 192 vCPUs / 64 global ranks = 3 threads instead of ~48, bottlenecking on client CPU rather than storage. (#689, #702)

**Invocation**
- Object-storage checkpoint reads always found 0 checkpoints and aborted — `checkpoint_folder` path double-prepended bucket name (`s3://bucket/bucket/prefix`); writes succeeded but reads never started. (#690; DLIO PR #44)
- Multi-host shared-filesystem checkpoint runs crashed with `FileExistsError` despite `exist_ok=True` — stale negative dentry cache on networked FS (NFS/GPFS/Lustre) caused `isdir()` recheck to fail within the brief convergence window. (#699; DLIO PR #45)

---

## Version 3.0.36 (July 7, 2026)

**Other Significant**
- `reportgen` output rows contained no computed metric values — the per-workload aggregation function was a stub (`metrics={}`) since reportgen was introduced; every row in every `results.{csv,json}` file was missing all throughput, latency, and recall columns. (#707)
- `reportgen` grouped workloads by `(model, accelerator)` regardless of benchmark type — VDB workloads were keyed on non-existent fields instead of `(engine, index_type)`; KVCache workloads omitted `performance_profile`; multi-org submissions merged all orgs into a single incorrect aggregate row. (#707)
- `reportgen` did not set `INVALID` for invocation count violations — training runs with a count other than 6 (1 warmup + 5 real per §2.1.17) and checkpointing runs with op lists other than 10 (per §2.1.23) were silently aggregated with the wrong category. (#707)
- `reportgen` applied INVALID gates to `whatif` rows — simulation output was incorrectly marked INVALID when invocation counts did not meet submission requirements; `whatif` is not a submission and should bypass all rules-strict gates. (#707)
- Training per-model rollup written to `<model>/run/` but empty-dir scan walked to `<model>/` — path mismatch caused a stray empty `results.{csv,json}` to be written at `training/<model>/` on every `reportgen` invocation. (#707)

---

## Version 3.0.37 (July 7, 2026)

**Invocation**
- VDB datagen: concurrent per-rank flush calls hit Milvus's per-collection rate limit (0.1 qps); pymilvus retry budget exhausted, datagen aborted at scale. (#705, #709)

**Other Significant**
- CLOSED code-image verify raised `CodeImageError` on any hash mismatch — valid submissions were blocked if mlpstorage was upgraded between runs of the same submission. (#710)

---

## Version 3.0.38 (July 8, 2026)

**Score-Affecting**
- §4.7.1 30 s gap check charged the read invocation's own framework startup (~50 s) to the failover-callout budget — legitimate split-mode submissions were unmeetable. Gap origin is now `read.metadata.invocation_start_time`. (#696, #714)
- reportgen D-26 flagged every v3.0 6-invocation training group INVALID — the warmup detector fired only on the retired `--loops` collision; the 5-run mean was skipped and `train_mean_of_au_percentage` was omitted from results.json. Lex-earliest tiebreak now identifies the warmup. (#719, #720, #722, #730)

**Invocation**
- Multi-host checkpoint runs raised `Unsupported URI scheme` on model-parallel shards off the head node — `MLPS_/MLPSTORAGE_` env vars set by mlpstorage weren't forwarded via `mpirun -x`. (#712, #713)
- Submission-mode runs colliding with a capture-prewritten pointer leaf bumped the timestamp +1 s and split outputs across two leaves — both failed submission gates (~40–50 spurious `[ERROR]` lines per validate). (#718, #729)
- `mlpstorage history rerun N` failed with E101 (orgname not resolved) — the historical command line clobbered the pre-swap resolved args; orgname now re-resolves from the historical `--results-dir` sentinel. (#721, #736)

**Other Significant**
- Legacy-layout migration wrote `.mlps-code-image` pointers into leaf subdirs (`dlio_config/`, `collector-staging/`, `.chk_iterations/`) — 2.1.15 / 2.1.20 / 2.1.26 exact-file-set checks then marked previously-valid CLOSED submissions broken as soon as any run triggered migration. (#725, #737)
- Training `results.json` emitted no `train_mean_of_au_percentage` — `metadata.json` omits the `metric` block and reportgen never fell back to `summary.json`; every training run silently lost its headline number. (#733, #735)
- Training `datagen` runs hit the D-27 "expected 6 training invocations" gate and were flagged INVALID — rules-strict count gates now scope to the `run` command only (also fixes latent D-20/D-24 checkpointing case). (#717, #728)
- First-run submission trees tripped CHECK-04 D-91 — pool `code-*` dirs existed but the `.mlps-image-pool` sentinel did not; both capture and retro-heal paths now write it. (#716, #727)
- Network-attached storage targets always failed submission on the `disk_io` check — `disk_io` is now marked N/A for network-attached targets. (#591, #726)
- Training `datagen` could silently overwrite an existing populated `<data-dir>/<model>` tree — datagen now refuses non-empty targets and emits a self-describing `.mlps-datagen-manifest.json`. (#731, #732)

---

## Version 3.0.39 (July 9, 2026)

One mlpstorage-level fix landed in this range. The DLIO pin did not move. The s3dlio floor was bumped 0.9.106 → 0.9.110, pulling in the v0.9.108 performance & concurrency audit (17 fixes) and the v0.9.110 multi-agent bug audit (39 fixes across 6 phases). The most benchmark-relevant s3dlio-layer changes are listed below. Several fixes live in s3dlio's own DLIO-adapter subpackage (`python/s3dlio/integrations/dlio/`, which ships inside the s3dlio wheel) — these are s3dlio-side code changes, distinct from the DLIO storage handlers in `mlcommons/DLIO_local_changes`.

**Score-Affecting**
- s3dlio: per-process GET throughput on many-small-object workloads (unet3d-style) was capped at ~1.1 GB/s regardless of prefetch depth — task-level parallelism at 9 fetch sites, zero-copy range assembly, and a streaming SDK connector deliver 3.2–3.8× on 64 KB workloads at high concurrency, 1.6–2.1× on single-object concurrent range GET ≥64 MiB, and +17–19% on 256 KB – 1 MB GET. (#701; s3dlio PR #149)
- s3dlio: HTTP/2 was ALPN-negotiated by default on `https://` and is measurably slower than HTTP/1.1 for object-storage workloads in this codebase — default reversed to HTTP/1.1 on both schemes. Object-storage throughput measurements from prior versions may not be directly comparable to 3.0.39+. Restore prior behavior with `S3DLIO_HTTPS_H2=1` or `S3DLIO_ENABLE_HTTP2=1`. (s3dlio PR #149)
- s3dlio: `direct://` (O_DIRECT) `list()` returned `file://` URIs — round-tripping the results through `store_for_uri()` silently dropped O_DIRECT semantics, so recursive walks over a `direct://` tree quietly reverted to buffered I/O. (s3dlio PR #159)
- s3dlio: `S3DLIO_PUT_MAX_RETRIES=0` produced a 0-attempt retry loop that failed every write with "no attempts made" — now correctly falls back to the documented default of 3. (s3dlio PR #159)

**Invocation**
- Multi-host SSH preflight aborted valid PALS/Slurm runs — `mlpstorage` probed passwordless SSH between compute nodes for every distributed run, but PALS (`palsd`) and Slurm (`slurmstepd`) launchers do not spawn ranks over SSH; on sites where SSH is disabled by policy, correctly configured runs failed with no launcher-aware bypass. New `--skip-ssh-check` targets only the SSH probe; PALS and Slurm launchers are auto-detected via `PALS_*` / `SLURM_JOB_ID` and skip the probe automatically. (#740)
- s3dlio DLIO-adapter subpackage: the `S3dlioStorage` adapter (shipped in the s3dlio wheel, not in DLIO_local_changes) hardcoded a 32 MiB multipart threshold with no env override, unlike its sibling `ObjStoreLibStorage` in DLIO_local_changes — v0.9.110 wires `S3DLIO_MULTIPART_THRESHOLD_MB`, `_PART_SIZE_MB`, `_MAX_IN_FLIGHT`, and `S3DLIO_DISABLE_MULTIPART` through the adapter, so operators can force single-PUT for large objects. (#715; s3dlio PR #159)
- s3dlio: `RangeEngine::download()` on a zero-byte object returned an error — legitimately empty files on Azure/GCS aborted the read path; now succeeds. (s3dlio PR #159)
- s3dlio DLIO-adapter subpackage: `s3_torch_storage.walk_node()` silently flattened nested object-store keys — subdirectory structure was destroyed on any recursive listing, breaking checkpointing layouts that rely on prefix structure. (s3dlio PR #159)
- s3dlio DLIO-adapter subpackage: `s3_torch_storage.create_node` / `delete_node` / `walk_node` swallowed every exception and returned `True` / `False` / `[]` — auth failures, network errors, and permission denials were invisible to callers and never surfaced as an aborted run. (s3dlio PR #159)
- s3dlio: Azure multipart uploads swallowed mid-upload errors in the `__exit__` path and had no client-side 50,000-block cap check — silent partial uploads or a wasted full upload's worth of network I/O before Azure rejected the commit server-side. (s3dlio PR #159)

**Other Significant**
- s3dlio DLIO-adapter subpackage: `AWS_ENDPOINT_URL` set in the user's environment was clobbered by the adapter's own endpoint selection — user-configured S3-compatible endpoints were silently overridden. (s3dlio PR #159)
- s3dlio: `parse_s3_uri_full` endpoint-detection heuristic misrouted legitimate bucket names with 2+ dots, leading digits, or names containing `minio` / `ceph` / `localhost` (e.g. `mycompany.data.backups`) as custom endpoint hostnames — heuristic narrowed; use `S3DLIO_S3_ENDPOINT_HINT_TLDS` to opt back in. (s3dlio PR #159)
- s3dlio: GCS RAPID-bucket detection had no exponential backoff on retry, and a transient failure permanently poisoned the detection cache with the wrong answer — subsequent GCS reads used the wrong code path for the rest of the process lifetime. (s3dlio PR #159)

---

## Version 3.0.40 (July 9, 2026)

No mlpstorage-level code change landed in this range. The DLIO pin advanced twice in the same day — first to DLIO main HEAD `86945a7a` picking up DLIO PRs #46 and #47 (storage#626 concurrency work, buckets 1 and 2), then to `95c6a9d4` picking up DLIO PR #48 (storage#741 page-cache-vs-memory-guard fix). All three DLIO-side fixes are listed below; each lives entirely inside `mlcommons/DLIO_local_changes` and reached submitters via the version bump. The s3dlio floor did not move in this range.

**Score-Affecting**

_(None. All three DLIO fixes in this range are invocation or metadata correctness — no measured-throughput or latency change.)_

**Invocation**
- DLIO: `read_threads` memory guard sampled `MemAvailable` before the per-epoch page-cache flush, so on Lustre/POSIX the reclaimable cache from run 1 was counted as "used" and runs 2–5 crashed with a bogus per-node memory-budget error. Valid 5-run sets landed as `INVALID: 0 runs`. Rerun any 5-run set where only run 1 completed. (#741; DLIO PR #48)
- DLIO: `_s3_iterable_mixin` created its prefetch `ThreadPoolExecutor` at module import in the parent; the worker thread didn't survive `os.fork()`, so `minio` and `s3torchconnector` DataLoader workers blocked forever on `future.result()`. `s3dlio` path was unaffected. Rerun any minio/s3torch training run that hung at 0% CPU. (#626; DLIO PR #46)

**Other Significant**
- DLIO: per-worker prefetch concurrency in `_S3IterableMixin` depended on `storage_library`: `s3dlio` 64, `minio` 16 (4× lower), `s3torchconnector` fully sequential (up to 64× lower). CLOSED comparisons across libraries were unfair; unified to a shared 64-way ceiling. Rerun any cross-library S3 comparison from earlier versions. (#626; DLIO PR #47)

---

## Version 3.0.41 (July 10, 2026)

Six mlpstorage-level fixes plus one DLIO pin advance (to `a7d56e73`, picking up DLIO PR #49) landed in this range. The s3dlio floor did not move.

**Score-Affecting**

_(None. All 3.0.41 fixes are invocation or metadata correctness — no measured-throughput or latency change.)_

**Invocation**
- DLIO: Hydra's `run_job`/`_save_config` called `Path.mkdir(exist_ok=True)` on the shared `hydra.run.dir` from every rank on every node; on networked FS the post-`FileExistsError` `is_dir()` recheck could see stale peer metadata and re-raise, aborting the run before benchmark work started. Rerun any multi-node run that died at Hydra bootstrap. (#754; DLIO PR #49)
- `training run` and `checkpointing run` returned SUCCESS regardless of DLIO's exit code and never stat-checked required leaf artifacts; a crashed or killed DLIO looked healthy until `mlpstorage validate` later complained about missing `dlio.log`/`summary.json`. Rerun any training/checkpointing "success" you can't corroborate with an intact leaf. (#761, #764)
- KVCache discarded mpirun's rc, downgraded missing rank output files to per-rank WARNINGs, and averaged zeros into `summary.json` — runs whose `kv-cache.py` crashed (OOM on llama3.1-70b-instruct, etc.) reported SUCCESS with meaningless aggregates. Rerun any KVCache success whose numbers look implausibly small. (#758, #759)
- Training `datagen` wrote `.mlps-datagen-manifest.json` and returned SUCCESS even when DLIO exited non-zero or left required leaf artifacts (`dlio.log`, `dlio_config/`, `training_*_metadata.json`) missing; downstream `run`/`reportgen` consumed the partial dataset silently. Rerun any datagen whose SUCCESS you can't corroborate with an intact leaf. (#744, #750)

**Other Significant**
- Rule 2.1.12 (STRUCT-12) required training workload dirs to contain exactly `{datagen, run}`, but `mlpstorage training datasize` emits a `datasize/` directory that PR #611 taught rule 3.3.1 to require. Submitters were stuck between STRUCT-12 error and 3.3.1 warning. `datasize/` is now allowed. Re-validate any datasize submission previously flagged INVALID. (#752)
- `DLIOBenchmark.datasize()` writes `dataset.total_disk_bytes` into the datasize sentinel per §3.3.1, but the run-rules checker didn't list it in `TOOL_INJECTED_PARAMS`; reportgen saw a user override, emitted `[INVALID] Disallowed parameter override`, and the cascaded "requires 5 runs" gate failed valid submissions. Re-validate submissions that failed that gate. (#760, #762)

---

## Version 3.0.42 (July 11, 2026)

No mlpstorage-level code change landed in this range. The DLIO pin advanced once, to `edaf4bb` (DLIO v3.0.4 / PR #50), picking up three DLIO-side fixes; the s3dlio floor moved from `>=0.9.110` to `>=0.9.112`, picking up the counterpart runtime fix plus FFI error-chain preservation.

**Score-Affecting**
- DLIO: `ObjStoreLibStorage` derived `S3DLIO_RT_THREADS` from the pre-auto-size `write_threads=1` sentinel, pinning s3dlio's Tokio runtime to one worker; NP=1 S3 writes dropped ~9x (~214 vs ~1928 MB/s on UNET3D datagen). Rerun any S3 datagen/checkpointing with auto-sized `write_threads`. (#780; DLIO PR #50)
- s3dlio: MPI-aware auto-init — `configure_thread_pools(0)` at import sizes the Tokio runtime to `cpus/MPI-world-size` when no explicit env-var override is present, avoiding oversubscription at NP>1 and undersizing at NP=1. (s3dlio v0.9.112 / PR #163)
- s3dlio: defense-in-depth clamp — `S3DLIO_RT_THREADS` values below `RT_THREADS_LIMIT/4` are raised to `RT_THREADS_LIMIT`, so any downstream miscomputation of the env var (like DLIO's pre-fix #780) no longer starves the runtime. (s3dlio v0.9.112 / PR #163)

**Invocation**
- DLIO: every MinIO run died at storage construction with `AttributeError: 'MinIOAdapter' object has no attribute 'bucket_exists'`, re-raised by preflight as `ConnectionError: cannot reach bucket ... via minio`. Delegator added. Rerun any MinIO preflight failure. (#756; DLIO PR #50)

**Other Significant**
- DLIO: opaque `RuntimeError: concurrent range chunk failed` on undersized hosts now emits a proactive workload-shape warning at read start (FD/RAM/CPU projections) and chains a live resource snapshot (fd, RSS, load-avg) onto any RuntimeError. Diagnostic-only. (#755; DLIO PR #50)
- s3dlio: FFI-boundary hardening (Tiers 1-5) preserves anyhow error chains across the Rust/Python boundary, so a Python-side `RuntimeError` now shows the real underlying cause instead of a bare "concurrent range chunk failed". (s3dlio v0.9.112 / PR #163)

---

## Version 3.0.43 (July 12 – 15, 2026)

Nine mlpstorage-level fixes landed in this range. The DLIO pin and s3dlio floor did not move.

**Score-Affecting**
- Rule 3.1.2's memory-derived minimum is a float; datagen and runtime accepted floor(N.x) but `mlpstorage validate` re-derived N.x and rejected by exactly one file — runs looked valid then failed submission with a truncated "N < N" message. All three sites now use `math.ceil`. (#796)
- Split-mode checkpoint submissions failed §4.7.1's 30s failover-callout gap: the write phase's post-benchmark teardown (30–60s of multi-node collection + JSON serialization + rank-0 exit at scale) was charged against the budget. New `INVOCATION_END` bookend excludes teardown, symmetric to #714's read-side fix. (#782, #783, #787)
- `datasize` sized against `--client-host-memory-in-gb` only, but runtime `check_num_files_train` and rule 3.1.2 use MPI-measured host RAM — when measured > declared, datasize under-recommended `num_files_train` and the run failed the check. Now takes `max(declared, measured)` with a WARNING when measured wins. (#785, #786)

**Invocation**
- Multi-node llama3-70b checkpoint runs livelocked cluster-wide after checkpoint 1 — the #768 child-side immediate-unregister pattern collided with mlpstorage's per-`save()` `_release_buffer_pool` unlinks, producing a 700+ `sem_unlink → FileNotFoundError` cascade. Now skips the child's REGISTER via a scoped monkey-patch. (#777, #780)
- `--results-dir` inside the mlpstorage source tree caused `capture_or_verify_code_image` to `shutil.copytree` its own destination into itself, recursing until `[Errno 36] File name too long`. Now refuses up front with a `ConfigurationError` naming `--results-dir`. (#778, #781)
- CLOSED checkpointing accepted `--num-processes` values that were neither the subset (8) nor the full-model count, ran the full write+read pair, and only `mlpstorage validate` later rejected with a misleading "subset requires exactly 8". New pre-run gate rejects with both valid forms named. (#792, #794)

**Other Significant**
- `reports reportgen` on a CLOSED checkpointing submission bundled with a `checkpointing datasize` preflight reported "got 3 invocations, expected 1 or 2" and "expected 10 ops, found 20" — datasize inherits the default 10/10 for `--num-checkpoints-*` and the workload-key lumped it with `run`. Now excluded. (#791, #793)
- Rules 3.4.2 / 4.4.2 / 5.4.2 flagged shared-mount `<data-dir>` / `<results-dir>` as ERROR — many valid layouts share the two paths. Findings now emit at WARN and no longer fail the check; the D-B8 evidence-gap path (no CAP-03 sidecar and no `df` block) still fires as ERROR. Re-validate submissions previously flagged INVALID for this. (#779, #788)
- VDB uniform-random query recall was non-discriminative at 1536-d (query-corpus similarity concentrates at ~1.1, top-k boundary within float32 noise) — results didn't transfer across index configs. New `query_mode=planted` (contrast >100) and `recall_epsilon` (tie-aware credit); defaults unchanged. (#625, #789)

---

## Version 3.0.44 (July 17, 2026)

Three fixes landed in this range, all closing the #805 VDB ground-truth-integrity failure (Solidigm v3.0.42 AISAQ reported Recall@10 = 0.00 across all five runs, yet every run validated) — one in the vdb_benchmark engine and two in `mlpstorage validate`. The DLIO pin and s3dlio floor did not move.

**Score-Affecting**
- AISAQ Recall@10 = 0.00 on every query: `create_flat_collection` reused an existing FLAT ground-truth collection whenever its entity *count* matched the source, so a stale ground truth left over from a regenerated collection (same size, different vectors) was silently paired with the new index — every ANN result missed the ground truth and recall was exactly 0.00, while QPS/latency and `coverage: 1.0` still looked healthy. Content is now verified (sampled-vector comparison) before reuse, and any all-zero-recall run aborts at run time instead of writing `valid: true`. (#805; #806)
- `mlpstorage validate` marked 0.0-recall runs valid: rule §5.3.2 (`vdbRecallReported`) checked only that a recall value was *present*, never its magnitude, so the broken AISAQ artifacts passed submission. It now reads the recall value and invalidates any run whose recall is 0.0, independent of the deferred per-scale minimum-recall table. (#807)
- `mlpstorage validate` was blind to ground-truth completeness: it never read `result_verdict.json`, so a run whose FLAT ground truth was incomplete (`coverage < 1.0`, the benchmark's "degraded" state) or failed to build passed validation. New rule §5.3.5 (`vdbGroundTruthIntegrity`) reads the raw ground-truth-setup fields and fails such runs; a missing or older record warns rather than false-failing. (#808; #809)

---

## Version 3.0.45 (July 18, 2026)

VDB runs failed submission validation (§5.4.1) because `storage_root` was not recorded in run metadata. Measured scores — QPS, latency, recall — are unaffected, so no rerun is needed; existing v3.0.42 VDB submissions now validate. (#802, #815)

---

## Version 3.0.46 (July 20, 2026) — Release Candidate `v3.0-rc1`

This is the release-candidate cut of the v3.0 line: the tree is promoted and tagged `v3.0-rc1` to signal
that v3.0 is feature-complete and under submission-window freeze. Two HPE/Cray PALS launcher fixes landed
over 3.0.45. Both are strictly PALS-specific: if you launch with `mpirun` (OpenMPI) or run single-node,
these changes are a complete no-op — nothing about your runs changes. They matter only on HPE/Cray PALS
`mpiexec` systems (ALCF Crux/Polaris/Aurora), where they are invocation fixes: on those systems the run
never started, so no completed run produced wrong numbers. The DLIO pin and s3dlio floor did not move.

**Invocation**
- kvcache multi-node runs on HPE/Cray PALS (ALCF Crux/Polaris/Aurora) failed to launch: the benchmark appended OpenMPI-only `--mca orte_abort_on_non_zero_status 0` to PALS `mpiexec`, which rejected it with "unrecognized option '--mca'". The `--mca` params are now emitted only on the mpirun/OpenMPI path. (#819)
- Multi-node runs on HPE/Cray PALS failed the CAP-02b `--results-dir` shared-FS probe: its launcher passed OpenMPI-only `--map-by`/`--bind-to` flags to PALS `mpiexec` ("unrecognized option --map-by"). It now takes the PALS-native `--ppn` + bare `--hosts` branch its sibling probe already used. (#818)

Scores from any 3.0.45 run are unaffected, so no rerun is needed relative to 3.0.45. If you are on an
older release, review the significant issues fixed between your version and 3.0.46 in the sections above
to determine whether your prior runs need a rerun or one is merely recommended.
