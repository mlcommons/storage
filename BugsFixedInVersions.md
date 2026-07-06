# MLPerf Storage Benchmark v3.0 — Significant Bugs Fixed by Version

**Date:** 2026-07-06
**Covers:** Versions 3.0.3 through 3.0.34 (May 28 – July 6, 2026)

---

## Source Size Code For Versions 3.0.3 – 3.0.9 (May 28 – June 14, 2026)

| Subtree | May 29 (excl. tests) | May 29 (tests only) | HEAD (excl. tests) | HEAD (tests only) |
|---|---:|---:|---:|---:|
| `mlpstorage_py` | 22,222 | 2,995 | 39,309 | 16,395 |
| `training` + `checkpointing` (DLIO) | 20,746 | 3,389 | 18,226 | 8,552 |
| `vdb_benchmark` | 11,346 | 5,632 | 14,424 | 6,964 |
| `kv_cache_benchmark` | 6,424 | 4,013 | 6,413 | 4,112 |
| **Subtotal** | **60,738** | **16,029** | **78,372** | **36,023** |
| **Total (all code)** | **76,767** | | **114,395** | |

---

Each section below lists significant bugs fixed in that release, organized into three categories:

- **Score-Affecting** — bugs that alter measured throughput, latency, recall, or pass/fail outcome
- **Invocation** — bugs that prevented the benchmark from starting or completing a run
- **Other Significant** — bugs in submission validation, report generation, or metadata that could cause valid results to appear invalid, or vice versa

Trivial fixes (CLI message wording, whitespace, docs-only changes, test-only changes) are excluded.
Issue and PR numbers are provided at the end of each entry for reference.

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

DLIO dependency updates (DLIO PRs #29 and #36). No new mlpstorage-level significant bug fixes above 3.0.20.

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

DLIO dependency updates (DLIO PRs #37 and #38). No new mlpstorage-level significant bug fixes above 3.0.23.

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

---

## Version 3.0.34 (July 6, 2026)

**Score-Affecting**
- Rule 3.3.1 threw `TypeError` on int vs str comparison — training data file-count validation was always skipped. (#681, #684)

**Invocation**
- gRPC 256 MB default message limit caused `RESOURCE_EXHAUSTED` for large VDB datasets — large-scale VDB runs failed. (#572, #685)
