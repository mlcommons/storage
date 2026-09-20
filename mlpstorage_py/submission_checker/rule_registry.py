"""
Registry helpers for Rules.md-traceable check methods.

Exposes a ``@rule(rule_id, rule_name, *, since=None, until=None)`` decorator
that attaches Rules.md metadata to a check method without altering its
behaviour, and introspection helpers that enumerate every decorated method
on a check class.  Phase 3's coverage tool and QUAL-02's enforcement test
both consume ``discover_rules``.

Edition gates
-------------
A rules edition (``"3.0"``; see ``mlpstorage_py/editions.py``) is the
version of Rules.md a submission declares.  A check whose *logic* changed
between editions is bound per edition instead of growing ``if edition``
branches: the old method keeps its ``@rule`` with ``until="4.0"`` and the
new method carries the same rule id with ``since="4.0"``.  The range is
half-open, ``[since, until)``, and either bound may be omitted; no bounds
means every edition, which is what every pre-existing binding says.
``BaseCheck.run_checks`` skips a registered check whose gate excludes the
edition of its ``config`` (built per submission from ``submission.yaml``),
so neighbours are untouched when one rule is replaced.

Several methods may still implement facets of one rule id in the same
edition (4.7.1 has two); ``discover_rules`` keeps one entry per id, the last
by method name, exactly as before gates existed.  ``gated_overlaps`` reports
the one shape that is almost certainly a mistake: two *gated* bindings of
one rule id whose ranges meet.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Any, List, NamedTuple, Optional, Tuple

from mlpstorage_py.config import RULES_EDITION

_EDITION_ID_RE = re.compile(r"[0-9]+\.[0-9]+")


def edition_key(edition_id: Any) -> Tuple[int, int]:
    """Sort key of a rules edition id: ``"3.10"`` sorts after ``"3.9"``.

    Raises ``ValueError`` for anything but a ``<major>.<minor>`` string.
    """
    if not isinstance(edition_id, str) or not _EDITION_ID_RE.fullmatch(edition_id):
        raise ValueError(f"not a rules edition id (expected '<major>.<minor>'): {edition_id!r}")
    major, minor = edition_id.split(".")
    return int(major), int(minor)


@dataclass(frozen=True)
class EditionGate:
    """The half-open edition range ``[since, until)`` a binding applies to.

    ``None`` on either side leaves it unbounded; both ``None`` is the open
    gate (:data:`ALL_EDITIONS`).
    """
    since: Optional[str] = None
    until: Optional[str] = None

    def __post_init__(self):
        if self.since is not None:
            edition_key(self.since)
        if self.until is not None:
            edition_key(self.until)
        if self.since is not None and self.until is not None \
                and edition_key(self.since) >= edition_key(self.until):
            raise ValueError(f"empty edition range: since={self.since!r} must be below until={self.until!r}")

    @property
    def is_open(self) -> bool:
        return self.since is None and self.until is None

    def applies_to(self, edition_id: str) -> bool:
        key = edition_key(edition_id)
        if self.since is not None and key < edition_key(self.since):
            return False
        if self.until is not None and key >= edition_key(self.until):
            return False
        return True

    def overlaps(self, other: "EditionGate") -> bool:
        """True when some edition satisfies both ranges."""
        lo = max((g.since for g in (self, other) if g.since is not None), key=edition_key, default=None)
        hi = min((g.until for g in (self, other) if g.until is not None), key=edition_key, default=None)
        return lo is None or hi is None or edition_key(lo) < edition_key(hi)

    def __str__(self) -> str:
        if self.is_open:
            return "all editions"
        parts = []
        if self.since is not None:
            parts.append(f">= {self.since}")
        if self.until is not None:
            parts.append(f"< {self.until}")
        return "editions " + " and ".join(parts)


ALL_EDITIONS = EditionGate()


def rule(rule_id: str, rule_name: str, *, since: Optional[str] = None, until: Optional[str] = None):
    """Decorator factory that attaches Rules.md metadata to a check method.

    The decorated function is returned *unchanged* — no wrapper is created,
    so ``inspect.signature``, ``self`` binding, and
    ``inspect.getmembers`` lookups all behave as if the decorator were absent.
    The only effect is the addition of three attributes on the function
    object: ``__rule_id__``, ``__rule_name__`` and ``__rule_gate__``.

    Args:
        rule_id: The dotted rule ID from Rules.md (e.g. ``"2.1.2"``).
        rule_name: The camelCase rule name from Rules.md
            (e.g. ``"topLevelSubdirectories"``).
        since: First rules edition the binding applies to (inclusive),
            e.g. ``"4.0"``. Omitted = unbounded below.
        until: First rules edition the binding no longer applies to
            (exclusive), e.g. ``"4.0"``. Omitted = unbounded above.

    Together ``since`` / ``until`` form the half-open edition gate
    ``[since, until)`` stored as an :class:`EditionGate` in
    ``__rule_gate__``. Neither given (the two-argument form every existing
    binding uses) means every edition. A malformed or empty range raises
    ``ValueError`` at decoration time.

    Returns:
        A decorator that attaches the three attributes to the decorated
        callable and returns it unchanged.

    Example::

        @rule("2.1.2", "topLevelSubdirectories")
        def top_level_subdirectories_check(self):
            ...
        # top_level_subdirectories_check.__rule_id__   == "2.1.2"
        # top_level_subdirectories_check.__rule_name__ == "topLevelSubdirectories"
        # top_level_subdirectories_check.__rule_gate__ is ALL_EDITIONS

        @rule("4.2.1", "auThreshold", until="4.0")     # logic of editions < 4.0
        def au_threshold_check(self): ...
        @rule("4.2.1", "auThreshold", since="4.0")     # replacement from 4.0 on
        def au_threshold_v4_check(self): ...
    """
    gate = ALL_EDITIONS if since is None and until is None else EditionGate(since, until)

    def decorator(func):
        func.__rule_id__ = rule_id
        func.__rule_name__ = rule_name
        func.__rule_gate__ = gate
        return func
    return decorator


class RuleBinding(NamedTuple):
    """One ``@rule``-decorated method of a check class."""
    rule_id: str
    rule_name: str
    method_name: str
    gate: EditionGate


def gate_of(check: Any) -> EditionGate:
    """The edition gate of a decorated callable (bound or not); the open
    gate for anything undecorated or decorated before gates existed."""
    gate = getattr(check, "__rule_gate__", None)
    return gate if isinstance(gate, EditionGate) else ALL_EDITIONS


def discover_rule_bindings(check_class) -> List[RuleBinding]:
    """Every ``@rule``-decorated method on a check class, across all
    editions, ordered by (rule_id, method_name)."""
    bindings: List[RuleBinding] = []
    for method_name, method in inspect.getmembers(check_class, predicate=callable):
        rule_id = getattr(method, "__rule_id__", None)
        if rule_id is not None:
            bindings.append(RuleBinding(rule_id, getattr(method, "__rule_name__", ""),
                                        method_name, gate_of(method)))
    bindings.sort(key=lambda b: (b.rule_id, b.method_name))
    return bindings


def discover_rules(check_class, edition: Optional[str] = None) -> dict:
    """Enumerate the ``@rule``-decorated methods of a check class that apply
    to one rules edition (the current edition, ``config.RULES_EDITION``, by
    default).

    Uses ``inspect.getmembers(check_class, predicate=callable)`` and filters
    to members that carry a ``__rule_id__`` attribute (attached by the
    ``@rule``decorator) whose edition gate admits ``edition``.  Non-callable
    attributes, ``property`` objects, plain methods without ``@rule``
    decoration and bindings gated out of the edition are silently skipped.

    Args:
        check_class: A class (typically a ``BaseCheck`` subclass) whose
            methods may be decorated with ``@rule``.
        edition: The rules edition to project onto (``"3.0"``); ``None``
            is the current edition.

    Returns:
        A ``dict`` mapping ``rule_id`` (str) to a ``(rule_name, method_name)``
        tuple, where ``rule_name`` is the camelCase Rules.md name and
        ``method_name`` is the Python attribute name on the class.  When
        several methods implement one rule id in that edition the last by
        method name is kept (``inspect.getmembers`` order).  Returns an
        empty dict if no decorated methods apply.

    Example::

        rules = discover_rules(SubmissionStructureCheck)
        # {"2.1.2": ("topLevelSubdirectories", "top_level_subdirectories_check"), ...}
    """
    edition = RULES_EDITION if edition is None else edition
    edition_key(edition)  # reject garbage before it silently matches nothing
    result: dict = {}
    for method_name, method in inspect.getmembers(check_class, predicate=callable):
        rule_id = getattr(method, "__rule_id__", None)
        if rule_id is not None and gate_of(method).applies_to(edition):
            result[rule_id] = (getattr(method, "__rule_name__", ""), method_name)
    return result


def gated_overlaps(check_class) -> List[Tuple[str, RuleBinding, RuleBinding]]:
    """Pairs of *gated* bindings of one rule id whose ranges meet — the
    replace-a-rule mistake (old logic not closed before the new one opens).
    Ungated facets of a rule id never count.  Empty when the class is sound."""
    by_id: dict = {}
    for b in discover_rule_bindings(check_class):
        if not b.gate.is_open:
            by_id.setdefault(b.rule_id, []).append(b)
    found = []
    for rule_id, group in by_id.items():
        for i, first in enumerate(group):
            for second in group[i + 1:]:
                if first.gate.overlaps(second.gate):
                    found.append((rule_id, first, second))
    return found
