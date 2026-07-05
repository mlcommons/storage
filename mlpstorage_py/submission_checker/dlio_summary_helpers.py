"""Helpers for interpreting DLIO ``summary.json`` fields that the raw
array shape makes error-prone.

Currently limited to ``host_memory_GB``. DLIO populates that field via
an MPI ``SUM``-Reduce keyed on ``self.MPI.node()`` — one slot per
"node index", filled once per host by that host's local-rank-0. On
multi-rank-per-host clusters where ``self.MPI.node()`` does not
monotonically map to distinct ``0..nnodes-1`` slots, the resulting
array has some slots doubled (two hosts' local-rank-0 wrote to the
same slot; the SUM combined them) and some slots at 0 (no host's
local-rank-0 wrote there). The *sum* is always correct
(SUM-reduce is total-preserving); *positional* access is not.

See mlcommons/storage#669 for the reporter's evidence and the DLIO
follow-up that will emit a clean one-value-per-host array upstream.
Until that lands, every checker that needs to reason about cluster
memory must aggregate the whole array. These helpers are the single
place that aggregation lives so the anti-pattern
(``num_hosts * host_memory_GB[0]``, or a positional per-host loop
that trusts index alignment) cannot come back at a single call site.

The helpers accept the raw ``summary`` dict (not a pre-extracted
list) so callers uniformly go through this module rather than
sometimes reading the field themselves.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


def cluster_total_host_memory_gb(summary: Mapping[str, Any]) -> float:
    """Return the aggregate cluster host memory in GiB.

    Uses ``sum(host_memory_GB)``, which is authoritative regardless of
    DLIO's per-slot layout (see module docstring). Missing / non-list
    / non-numeric entries collapse to 0 rather than raising, since
    this helper is called on user-supplied summary.json content that
    may be malformed for reasons unrelated to storage#669.

    Args:
        summary: The ``summary.json`` dict as loaded from disk.

    Returns:
        Cluster total host memory in GiB. Returns 0.0 if the field is
        missing / empty / entirely non-numeric.
    """
    raw = summary.get("host_memory_GB")
    if not isinstance(raw, (list, tuple)):
        return 0.0
    total = 0.0
    for entry in raw:
        try:
            total += float(entry)
        except (TypeError, ValueError):
            continue
    return total


def per_host_memory_gb(summary: Mapping[str, Any]) -> Optional[float]:
    """Return the per-host memory in GiB, computed as cluster total /
    ``num_hosts``.

    Assumes homogeneous hosts, which is required for CLOSED
    submissions by rule 2.1.9 (``identicalSystemConfig``) and is the
    only defensible reading of a positionally-broken DLIO array. On
    heterogeneous OPEN submissions the returned value is the mean
    per host, which is still the best available answer given the
    input's information loss.

    Args:
        summary: The ``summary.json`` dict as loaded from disk.

    Returns:
        Per-host memory in GiB, or ``None`` if ``num_hosts`` is
        missing / zero / non-numeric.
    """
    total = cluster_total_host_memory_gb(summary)
    try:
        num_hosts = int(summary.get("num_hosts", 0))
    except (TypeError, ValueError):
        return None
    if num_hosts <= 0:
        return None
    return total / num_hosts
