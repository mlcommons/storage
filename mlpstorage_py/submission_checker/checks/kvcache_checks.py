"""KVCacheCheck — Rules.md §6 (KVCache) extension stub.

Rules.md §6 is empty at Phase 3 land time. STUB-02 establishes the
extension point so ``main.py`` can instantiate the class identically to
``DirectoryCheck`` / ``TrainingCheck`` / ``CheckpointingCheck`` (D-S4),
and so a future phase can fill in real checks without touching the
``main.py`` wiring shape.

Design constraints (Phase 3 CONTEXT.md):

* **D-S2:** registers exactly one placeholder method
  (``_section_unimplemented``) which is a no-op returning ``True``. No
  ``@rule`` decorator is applied — stubs MUST contribute zero rule-ID
  bindings via ``discover_rules`` (success criterion #2: stubs emit zero
  violations).
* **D-S3:** this module does NOT import ``coverage_mapping``. Stubs stay
  decoupled from the coverage tool; coverage advertisement lives only in
  ``STUB_COVERAGE`` inside ``coverage_mapping.py``.
"""

from .base import BaseCheck
from ..configuration.configuration import Config
from ..loader import SubmissionLogs


class KVCacheCheck(BaseCheck):
    """Stub check class for Rules.md §6 (KVCache) rules.

    Mirrors the ``CheckpointingCheck`` / ``TrainingCheck`` constructor shape
    (``__init__(self, log, config, submissions_logs)``) so the existing
    ``for checker in checkers:`` loop in ``main.py`` (Plan 03-04) can
    instantiate ``KVCacheCheck`` without any special-casing.

    Emits zero violations. Future phase populates
    ``STUB_COVERAGE['KVCacheCheck']`` in ``coverage_mapping.py`` when
    Rules.md §6 gains IDs.
    """

    def __init__(self, log, config: Config, submissions_logs: SubmissionLogs):
        """Initialize KVCacheCheck.

        Args:
            log: Logger instance (passed through to ``BaseCheck``).
            config: A ``Config`` instance containing submission configuration.
            submissions_logs: A ``SubmissionLogs`` instance for accessing
                submission logs. The stub stores it but does not introspect
                its contents.
        """
        super().__init__(log=log, path=submissions_logs.loader_metadata.folder)
        self.config = config
        self.submissions_logs = submissions_logs
        self.name = "kvcache checks"
        self.checks = []
        self.init_checks()

    def init_checks(self):
        """Register the placeholder no-op (D-S2).

        Rules.md §6 (KVCache) is empty at Phase 3 land time. When that
        section gains IDs, a future phase fills in real ``@rule``-decorated
        check methods here and populates ``STUB_COVERAGE['KVCacheCheck']``
        in ``coverage_mapping.py``.
        """
        self.checks = [self._section_unimplemented]

    def _section_unimplemented(self) -> bool:
        """No-op placeholder. Emits zero violations (Phase 3 success criterion #2)."""
        return True
