"""Registries that augment ``discover_rules()`` introspection for the
``rules_coverage`` CLI tool.

Two top-level constants:

* ``OUT_OF_SCOPE_RULES`` — Rules.md IDs deliberately skipped, with a
  free-text reason string. Populated on 2026-07-02 with the two §6
  KVCache rules whose content is descriptive/narrative rather than a
  submission-validator contract (§6.4.2 I/O model, §6.6.3 OPEN example
  invocation). See mlcommons/storage#658 and PR #602.
* ``STUB_COVERAGE`` — maps stub-class name → list of Rules.md IDs the
  stub *advertises* as covered. ``VdbCheck`` has been retired here
  (Phase 4 gave it real ``@rule`` bindings for §5). ``KVCacheCheck``
  now advertises the seventeen enforceable §6 KVCache rules added by
  PR #602; real ``@rule`` bindings are a follow-up phase (~800–1000 LOC
  checker + ~800–1200 LOC tests + emitter changes for
  ``aggregated_device_{read,write}_p95_ms`` under §6.3.4.3). Until then
  PR #602's own enforcement-status note applies: the §6 rules are
  enforced by the run-time CLI locks (6.3.2.1) and manual review.

This module has **no imports** and exposes **no functions** — it is a
pure data module consumed by ``rules_coverage`` (Plan 03-04). Stubs in
``checks/vdb_checks.py`` and ``checks/kvcache_checks.py`` MUST NOT
depend on this module (D-S3).
"""

# §6.4.2 and §6.6.3 are descriptive/narrative sections of Rules.md §6
# (KVCache) that carry no submission-validator contract. §6.4.2
# documents the POSIX I/O model (.npy files, np.save+fsync, np.load
# after POSIX_FADV_DONTNEED); §6.6.3 shows an example direct
# ``kv-cache.py`` invocation for OPEN submissions with latency tracing
# / RAG / BurstGPT. The "must record the exact invocation" clause of
# §6.6.3 is satisfied by the OPEN run's normal metadata capture, not a
# distinct check. See mlcommons/storage#658.
OUT_OF_SCOPE_RULES: dict[str, str] = {
    "6.4.2": (
        "descriptive — Rules.md §6.4.2 documents the KVCache POSIX I/O "
        "model (.npy files, np.save+fsync, np.load after "
        "POSIX_FADV_DONTNEED). No submission-level enforcement contract."
    ),
    "6.6.3": (
        "OPEN-narrative — Rules.md §6.6.3 shows an example direct "
        "kv-cache.py invocation for OPEN submissions with "
        "--enable-latency-tracing / --enable-rag / --use-burst-trace. "
        "The 'record exact invocation' clause is satisfied by the OPEN "
        "run's normal metadata capture, not a distinct validator check."
    ),
}


# Stub-class coverage advertisement: maps stub class name -> list of Rules.md
# rule IDs the stub stands in for. VdbCheck used to live here when Rules.md
# §5 was empty; after Phase 4 Plan 04-02 (D-01) it carries real
# ``@rule``-decorated methods for every §5 ID (5.1.1-5.6.5) and
# ``discover_rules`` picks them up directly, so the VdbCheck entry has been
# removed. ``KVCacheCheck`` advertises the seventeen enforceable §6 IDs
# added by PR #602; the descriptive §6.4.2 and §6.6.3 sit in
# ``OUT_OF_SCOPE_RULES`` above. See mlcommons/storage#658.
STUB_COVERAGE: dict[str, list[str]] = {
    "KVCacheCheck": [
        # §6.3.1 Sanctioned workload Options
        "6.3.1.1",
        # §6.3.2 CLOSED sequence locks
        "6.3.2.1",
        "6.3.2.2",
        # §6.3.3 Client scaling
        "6.3.3.1",
        "6.3.3.2",
        "6.3.3.3",
        "6.3.3.4",
        # §6.3.4 Result aggregation
        "6.3.4.1",
        "6.3.4.2",
        "6.3.4.3",
        "6.3.4.4",
        "6.3.4.5",
        # §6.4 POSIX Access (§6.4.2 → OUT_OF_SCOPE)
        "6.4.1",
        # §6.5 Object Access
        "6.5.1",
        "6.5.2",
        # §6.6 OPEN versus CLOSED (§6.6.3 → OUT_OF_SCOPE)
        "6.6.1",
        "6.6.2",
    ],
}
