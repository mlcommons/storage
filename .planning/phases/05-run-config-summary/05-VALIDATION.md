---
phase: 5
slug: run-config-summary
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-09
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pyproject.toml` (pytest section) |
| **Quick run command** | `pytest tests/unit -v -k "storage_config or run_summary or quiet"` |
| **Full suite command** | `pytest tests/unit -v` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/unit -v -k "storage_config or run_summary or quiet"`
- **After every plan wave:** Run `pytest tests/unit -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 1 | RUNSUM-06 | — | Credentials never in resolver return value | unit | `pytest tests/unit/test_storage_config.py -v` | ❌ W0 | ⬜ pending |
| 05-01-02 | 01 | 1 | RUNSUM-06 | — | resolve_object_storage_config returns (value, source) tuples | unit | `pytest tests/unit/test_storage_config.py -v` | ❌ W0 | ⬜ pending |
| 05-01-03 | 01 | 1 | RUNSUM-06 | — | Endpoint fallback chain resolves in priority order | unit | `pytest tests/unit/test_storage_config.py -v` | ❌ W0 | ⬜ pending |
| 05-02-01 | 02 | 2 | RUNSUM-01 | — | Summary prints before benchmark execution | unit | `pytest tests/unit/test_run_summary.py -v` | ❌ W0 | ⬜ pending |
| 05-02-02 | 02 | 2 | RUNSUM-02 | — | --quiet suppresses all summary output | unit | `pytest tests/unit/test_run_summary.py -v -k "quiet"` | ❌ W0 | ⬜ pending |
| 05-02-03 | 02 | 2 | RUNSUM-03 | — | S3 section absent when protocol != object | unit | `pytest tests/unit/test_run_summary.py -v -k "protocol"` | ❌ W0 | ⬜ pending |
| 05-02-04 | 02 | 2 | RUNSUM-04 | — | Credentials redacted in all output paths | unit | `pytest tests/unit/test_run_summary.py -v -k "redact"` | ❌ W0 | ⬜ pending |
| 05-02-05 | 02 | 2 | RUNSUM-05 | — | Endpoint row shows source label | unit | `pytest tests/unit/test_run_summary.py -v -k "endpoint"` | ❌ W0 | ⬜ pending |
| 05-02-06 | 02 | 2 | RUNSUM-07 | — | No regressions in full suite | regression | `pytest tests/unit -v` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_storage_config.py` — stubs for RUNSUM-06 (resolver unit tests)
- [ ] `tests/unit/test_run_summary.py` — stubs for RUNSUM-01 through RUNSUM-05

*Existing infrastructure (pytest, conftest.py, fixtures) covers all other phase requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Summary visually readable in terminal output | RUNSUM-01 | Format/readability is subjective | Run `mlpstorage closed training unet3d run file --num-accelerators 2 --accelerator-type a100 --data-dir /tmp --results-dir /tmp`; verify table is legible |
| `--quiet` silences only summary, not benchmark output | RUNSUM-02 | Requires live benchmark invocation | Run same command with `--quiet`; verify no summary table, normal benchmark output |
| S3 section appears for object protocol | RUNSUM-03 | Requires object-mode run | Run with `--data-access-protocol object`; verify S3 section present with env values |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
