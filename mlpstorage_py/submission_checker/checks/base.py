from abc import ABC, abstractmethod


class BaseCheck(ABC):
    """
    A generic check class meant to be inherited by concrete check implementations.
    Subclasses must register their check methods into `self.checks`.
    """

    def __init__(self, log, path):
        self.checks = []
        self.log = log
        self.path = path
        self.name = "base checks"
        pass

    def log_violation(self, rule_id, rule_name, path, msg, *args):
        """Log a Rules.md violation in the canonical locked format.

        Emits ``[<rule_id> <rule_name>] <path>: <msg>`` through
        ``self.log.error``, passing ``*args`` through for ``logging``'s lazy
        ``%``-style formatting (consistent with the ``self.log.error("%s ...",
        x)`` pattern used throughout the existing checks).

        Locked format (D-07): one space between rule_id and rule_name;
        colon and single space between path and msg.

        Args:
            rule_id: Dotted rule ID from Rules.md (e.g. ``"2.1.2"``).
            rule_name: camelCase rule name from Rules.md
                (e.g. ``"topLevelSubdirectories"``).
            path: The filesystem path where the violation was detected.
            msg: A ``%``-style format string describing the violation.
            *args: Format arguments for ``msg``.
        """
        prefix = "[%s %s] %s: " % (rule_id, rule_name, path)
        self.log.error(prefix + msg, *args)

    def warn_violation(self, rule_id, rule_name, path, msg, *args):
        """Warning-level counterpart to ``log_violation``.

        Emits ``[<rule_id> <rule_name>] <path>: <msg>`` through
        ``self.log.warning``.  Used when a condition deviates from the spec
        but is not definitively an error — for example, when STRUCT-06 runs
        against a CLOSED submission but no reference checksum is configured
        (D-12), or when the MD5 predicate encounters a symlink (D-13).

        Args:
            rule_id: Dotted rule ID from Rules.md (e.g. ``"2.1.6"``).
            rule_name: camelCase rule name from Rules.md
                (e.g. ``"codeDirectoryContents"``).
            path: The filesystem path where the condition was detected.
            msg: A ``%``-style format string describing the condition.
            *args: Format arguments for ``msg``.
        """
        prefix = "[%s %s] %s: " % (rule_id, rule_name, path)
        self.log.warning(prefix + msg, *args)

    def info_violation(self, rule_id, rule_name, path, msg, *args):
        """Info-level counterpart to ``log_violation`` / ``warn_violation``.

        Emits ``[<rule_id> <rule_name>] <path>: <msg>`` through
        ``self.log.info``. Used for notes that describe validator
        incompleteness rather than anything about the submission — e.g.
        the deferred vdb scale/recall/query target tables (worklist A6):
        those must stay grep-visible per rule ID but should not appear
        as per-submission WARNINGs in a review report.

        Args:
            rule_id: Dotted rule ID from Rules.md (e.g. ``"5.1.1"``).
            rule_name: camelCase rule name from Rules.md
                (e.g. ``"vdbDatasetScale"``).
            path: The filesystem path the note applies to.
            msg: A ``%``-style format string describing the note.
            *args: Format arguments for ``msg``.
        """
        prefix = "[%s %s] %s: " % (rule_id, rule_name, path)
        self.log.info(prefix + msg, *args)

    def run_checks(self):
        """
        Execute all registered checks. Returns True if all checks pass, False otherwise.
        """
        valid = True
        errors = []
        for check in self.checks:
            try:
                v = self.execute(check)
                valid &= v
            except BaseException:
                valid &= False
                # exc_info=True attaches the current exception's type +
                # message + traceback to the log record so the underlying
                # bug is debuggable instead of being silently described as
                # "Exception occurred". Required after the 2026-06-11
                # checkpoint_files typo fix, which unmasked latent
                # TypeError / AttributeError bugs in five DirectoryCheck
                # rule methods (2.1.22 through 2.1.26) that the typo had
                # been silently hiding.
                self.log.error(
                    "Exception occurred in %s while running %s in %s",
                    self.path,
                    check.__name__,
                    self.__class__.__name__,
                    exc_info=True,
                )
        return valid

    def execute(self, check):
        """Custom execution of a single check method."""
        return check()

    def __call__(self):
        """Allows the check instance to be called like a function.

        Per-check start/passing status lines are emitted at DEBUG. The
        per-rule violations are already self-describing (each carries
        ``[<rule_id> <rule_name>]`` plus the offending path), and
        ``main.run`` emits a single ``SUMMARY: ...`` line at the end of
        the validation. Wrapping every passing check with extra
        "Starting ..." / "All ... passed" lines just clutters the
        default output. Use ``--debug`` / ``-v`` to surface them again
        when tracing.

        The failure-path "Some X failed for: ..." stays at ERROR because
        it is a useful transition marker when the user is already
        scanning failure output. (A8: subclass names already end in
        "checks" — e.g. "directory checks" — so the format no longer
        appends a literal "Checks" that doubled the word.)
        """
        self.log.debug("Starting %s for: %s", self.name, self.path)
        valid = self.run_checks()
        if valid:
            self.log.debug("All %s passed for: %s", self.name, self.path)
        else:
            self.log.error(
                "Some %s failed for: %s",
                self.name,
                self.path)
        return valid
