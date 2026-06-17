"""
Registry helpers for Rules.md-traceable check methods.

Exposes a ``@rule(rule_id, rule_name)`` decorator that attaches Rules.md
metadata to a check method without altering its behaviour, and a
``discover_rules`` introspection helper that enumerates every decorated
method on a check class.  Phase 3's coverage tool and QUAL-02's enforcement
test both consume ``discover_rules``.
"""

import inspect


def rule(rule_id: str, rule_name: str):
    """Decorator factory that attaches Rules.md metadata to a check method.

    The decorated function is returned *unchanged* — no wrapper is created,
    so ``inspect.signature``, ``self`` binding, and
    ``inspect.getmembers`` lookups all behave as if the decorator were absent.
    The only effect is the addition of two attributes on the function object:
    ``__rule_id__`` and ``__rule_name__``.

    Args:
        rule_id: The dotted rule ID from Rules.md (e.g. ``"2.1.2"``).
        rule_name: The camelCase rule name from Rules.md
            (e.g. ``"topLevelSubdirectories"``).

    Returns:
        A decorator that attaches ``__rule_id__`` and ``__rule_name__`` to the
        decorated callable and returns it unchanged.

    Example::

        @rule("2.1.2", "topLevelSubdirectories")
        def top_level_subdirectories_check(self):
            ...
        # top_level_subdirectories_check.__rule_id__   == "2.1.2"
        # top_level_subdirectories_check.__rule_name__ == "topLevelSubdirectories"
    """
    def decorator(func):
        func.__rule_id__ = rule_id
        func.__rule_name__ = rule_name
        return func
    return decorator


def discover_rules(check_class) -> dict:
    """Enumerate every ``@rule``-decorated method on a check class.

    Uses ``inspect.getmembers(check_class, predicate=callable)`` and filters
    to members that carry a ``__rule_id__`` attribute (attached by the
    ``@rule`` decorator).  Non-callable attributes, ``property`` objects, and
    plain methods without ``@rule`` decoration are silently skipped.

    Args:
        check_class: A class (typically a ``BaseCheck`` subclass) whose
            methods may be decorated with ``@rule``.

    Returns:
        A ``dict`` mapping ``rule_id`` (str) to a ``(rule_name, method_name)``
        tuple, where ``rule_name`` is the camelCase Rules.md name and
        ``method_name`` is the Python attribute name on the class.  Returns an
        empty dict if no decorated methods are found.

    Example::

        rules = discover_rules(SubmissionStructureCheck)
        # {"2.1.2": ("topLevelSubdirectories", "top_level_subdirectories_check"), ...}
    """
    result: dict = {}
    for method_name, method in inspect.getmembers(check_class, predicate=callable):
        rule_id = getattr(method, "__rule_id__", None)
        if rule_id is not None:
            rule_name = getattr(method, "__rule_name__", "")
            result[rule_id] = (rule_name, method_name)
    return result
