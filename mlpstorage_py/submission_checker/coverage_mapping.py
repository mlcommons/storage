"""Registries that augment ``discover_rules()`` introspection for the
``rules_coverage`` CLI tool.

Two top-level constants:

* ``OUT_OF_SCOPE_RULES`` — Rules.md IDs deliberately skipped, with a
  free-text reason string. Empty at Phase 3 land time because the
  aggressive retrofit (D-R1, D-R2) covers every current Rules.md §2/§3/§4
  ID via ``@rule``-decorated check methods. The dict exists so future
  contributors can record deliberate skips (e.g., "v2 milestone", "object
  API deferred") in one place instead of scattering ``# TODO`` comments.
* ``STUB_COVERAGE`` — maps stub-class name → list of Rules.md IDs the
  stub *advertises* as covered. Both lists are empty at Phase 3 land
  time because Rules.md §5 (VDB) and §6 (KVCache) are empty. The
  structure exists so future contributors can fill in the IDs (e.g.,
  ``"5.1.1"``, ``"6.2.3"``) when Rules.md gains those sections — the
  ``rules_coverage`` tool consumes ``STUB_COVERAGE`` to report the IDs
  as "covered by stub" without the stubs themselves needing to know
  about the coverage tool (D-S3 decoupling).

This module has **no imports** and exposes **no functions** — it is a
pure data module consumed by ``rules_coverage`` (Plan 03-04). Stubs in
``checks/vdb_checks.py`` and ``checks/kvcache_checks.py`` MUST NOT
depend on this module (D-S3).
"""

# D-A1: empty at land time because the aggressive retrofit (D-R1/D-R2)
# covers every current Rules.md §2/§3/§4 ID. Populate with entries of the
# shape:
#   "rule_id": "free-text reason"
# (e.g., ``"6.5.1": "object API deferred to v2 milestone"``).
OUT_OF_SCOPE_RULES: dict[str, str] = {}


# D-A1/D-S3: lists are empty because Rules.md §5 (VDB) and §6 (KVCache)
# are empty at Phase 3 land time. Populate the lists when those sections
# gain IDs (e.g., ``"VdbCheck": ["5.1.1", "5.1.2"]``).
STUB_COVERAGE: dict[str, list[str]] = {
    "VdbCheck": [],       # populated when Rules.md §5 (VDB) gains IDs
    "KVCacheCheck": [],   # populated when Rules.md §6 (KVCache) gains IDs
}
