---
phase: 5
reviewers: [gemini]
reviewed_at: 2026-06-09T00:00:00Z
plans_reviewed: [05-01-PLAN.md, 05-02-PLAN.md]
---

# Cross-AI Plan Review — Phase 5: Run Configuration Summary

> **Reviewers**: Gemini (external)
> **Claude skipped** — running as current executor (independence rule)

---

## Gemini Review

This review evaluates the implementation plans for **Phase 5: Run Configuration Summary**.

### Summary
The plans are well-structured, prioritizing architectural cleanliness through centralization and ensuring correctness via a TDD (Test-Driven Development) approach. **Plan 05-01** successfully addresses the technical debt of scattered environment variable reads by consolidating them into a single resolver, while **Plan 05-02** provides the requested visibility for users. The use of "Wave" ordering and grep-verified acceptance criteria demonstrates a mature understanding of refactoring safety.

---

### Strengths
- **Centralized Source of Truth**: Consolidating S3 configuration resolution into `storage_config.py` significantly improves maintainability and prevents "logic drift" between different storage backends.
- **Source Transparency**: Returning a tuple `(value, var_name)` for the endpoint resolution is a clever way to satisfy the requirement of showing *why* a specific value was chosen.
- **Safety-First Refactoring**: The explicit mention of preserving the `os.environ` write in `s3dlio_writer.py` shows deep research into the MPI-related side effects of the codebase.
- **Robust Redaction**: The redaction strategy `[SET — N chars]` is excellent as it confirms the presence and approximate validity (length) of a key without exposing it.

---

### Concerns

- **Missing Raw Credentials in Resolver (HIGH Severity)**:
  Plan 05-01's implementation spec for `resolve_object_storage_config` lists `aws_access_key_id_redacted` in the return dict but does **not** explicitly list the raw, unredacted keys. Since the plan aims to replace all `os.environ.get` calls in storage drivers (like `minio_reader.py`), these drivers will need the **raw** values to authenticate. If the resolver only returns redacted values, the storage layer will break.

- **Summary Accuracy vs. `.env` Loading (MEDIUM Severity)**:
  The Research Findings note that `.env` loading happens *after* the summary is printed. This creates a "What You See Is NOT What You Get" scenario. If a user defines their S3 bucket in a `.env` file, the summary will show it as `[not set]`, leading to confusion or false bug reports, even if the benchmark later loads it correctly.

- **None-Value Semantics in S3 (LOW Severity)**:
  In `s3torch_reader.py`, the plan mentions `endpoint = endpoint or None`. It is critical that the centralized resolver distinguishes between "Empty String" (user error/unset) and `None` (SDK default to AWS S3) to avoid breaking standard AWS connectivity.

- **Hardcoded Table Width (LOW Severity)**:
  `_WIDTH = 28` might be too narrow for long S3 endpoint templates or deeply nested results directories, potentially causing messy wrapping in the CLI.

---

### Suggestions

1. **Update Resolver Signature**: Ensure `resolve_object_storage_config()` returns both raw and redacted keys.
   - *Example return keys:* `aws_access_key_id` (raw) and `aws_access_key_id_display` (redacted).
2. **Align `.env` Loading**: If possible, move the call to load the `.env` file (currently inside `_apply_object_storage_params`) to the very beginning of `_main_impl` in `main.py`. This ensures the summary reflects the *actual* parameters that will be used.
3. **Endpoint Source Detail**: For the `S3_ENDPOINT_URIS` case (which often contains a comma-separated list), ensure the resolver or the summary doesn't truncate the list, as seeing the full load-balancing pool is vital for debugging.
4. **Flexible Width**: Consider calculating the width based on the longest label or using a slightly wider default (e.g., `32`) to accommodate modern storage URIs.

---

### Risk Assessment: MEDIUM

**Justification**:
While the logic is sound, the risk stems from the wide-reaching refactor of six core storage files. If the centralized resolver has a bug or omits raw credentials, it will disable object-storage functionality across the entire suite. However, this is mitigated by the comprehensive TDD plan and the sequential Wave-based deployment.

**Verdict**: The plans are **Approved with Corrections**. Proceed once the resolver is updated to provide raw credentials to the storage backends.

---

## Consensus Summary

Only one external reviewer (Gemini) was available. Consensus derived from a single source.

### Agreed Strengths

- The `(value, source_label)` tuple for endpoint resolution elegantly covers RUNSUM-05 without any additional API surface.
- The `[SET — N chars]` redaction format is recognized as the right balance: confirms presence, reveals approximate validity, never leaks the value.
- Preserving the `os.environ` write at `s3dlio_writer.py:173` shows that the plan authors understood the MPI subprocess env-propagation side effect.
- Wave-ordered plans with grep-verified acceptance criteria indicate mature refactoring discipline.

### Key Concerns

**HIGH — Credential visibility in resolver (Gemini)**

> Gemini flags that the resolver returns only `aws_access_key_id_redacted` but the storage drivers need raw values for SDK auth.

**Planner response**: This concern is already handled in Plan 05-01 Task 2 — the instructions for `minio_reader.py` and `minio_writer.py` explicitly say:
> *"Keep: `access_key = os.environ.get('AWS_ACCESS_KEY_ID')` (raw — SDK auth, not display)"*

The resolver is for the **display/summary path only**. The SDK auth path stays as direct `os.environ.get()` calls. The resolver never needs to return raw credentials. This is a misread of the plan rather than a gap — but worth adding an explicit clarifying comment to the `storage_config.py` module docstring at implementation time.

**MEDIUM — `.env` loading post-summary (Gemini)**

> The summary prints before `.env` is loaded, so object-storage vars from `.env` will show `[not set]`.

**Planner response**: This is intentional and documented. The module docstring in `run_summary.py` is required to contain:
> *"NOTE: .env file loading happens in `_apply_object_storage_params()`, which runs after `run_benchmark()`. This summary shows pre-.env-load env state — by design."*

If the user defines S3 config in `.env`, the summary will be honest about what env state existed at invocation time. Moving `.env` loading earlier would require non-trivial surgery to `main.py` and is out of scope. A future enhancement could add a `[.env pending]` annotation to affected rows.

**LOW — `None` vs empty string semantics for endpoint (Gemini)**

> `s3torch_reader.py` uses `endpoint = endpoint or None` — care needed to not break None-means-AWS-S3.

**Planner response**: Plan 05-01 Task 2 explicitly addresses this:
> *"`endpoint = endpoint or None` (preserves None-means-AWS-S3 semantics in this file)"*

The `_resolve_endpoint()` helper returns `(None, '')` when all vars are unset — after unpacking, `endpoint_val or None` is still `None`. Valid.

### Divergent Views

No second reviewer available to produce a divergent view. If Gemini's HIGH concern was the only red flag, and it is resolved by re-reading the existing plan text, overall phase risk is:

**LOW–MEDIUM**: The refactor is broad (6 files) but well-bounded. TDD coverage and grep-verified acceptance criteria reduce regression risk significantly.
