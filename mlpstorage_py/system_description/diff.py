"""Pure-function diff core for systemname.yaml drift detection — Phase 05 / Plan 05-01.

This module is the comparison-subject layer between the on-disk systemname.yaml
(parsed into Python dicts by Slice 2's `parse_on_disk_systemname_yaml`) and the
freshly recomputed in-memory client stanzas (produced by Phase 02-04's
`write_systemname_yaml` → `_resolve_host_info_list` → `node_dict_from_host`).
It contains NO filesystem I/O, NO MPI involvement, and NO exception raises —
the `SystemDriftError` raise lives at the Slice 2 call site, not here. This
module is a pure transformation: list of dicts in, structured diff out.

Phase 05 / Plan 05-01 deliverables (Slice 1 of the lifecycle vertical):

- `DiffEntry` — frozen dataclass (path, old, new) representing one row of
  difference between the on-disk and in-memory views (D-37 / D-40).

- `DiffResult` — dataclass wrapping `entries: list[DiffEntry]` with a
  convenience `.empty` property. Returned by `diff_node_dict_lists`.

- `_flatten_to_paths(value, prefix='')` — generator yielding
  `(jsonpath, leaf_value)` for every leaf in the nested dict/list. Dict
  children use `prefix.key`, list children use `prefix[index]`. Empty
  containers yield nothing.

- `_compute_fingerprint(stanza)` — reuses `_FINGERPRINT_KEYS` and
  `_resolve_fingerprint_key` from `auto_generator.py` so the diff layer uses
  the exact same 11-tuple identity rule Phase 4 settled on for
  quantity-grouping. The D-38 round-trip-recompute contract depends on this:
  the same client must hash to the same fingerprint whether emitted into the
  on-disk YAML or recomputed in memory.

- `diff_node_dict_lists(on_disk, in_memory)` — the load-bearing public
  function. Indexes both sides by fingerprint, then for each fingerprint:
  - present only on-disk → emit a `<present>` / `<absent>` orphan entry (D-47);
  - present only in-memory → emit a `<absent>` / `<present>` orphan entry (D-46);
  - present on both → flatten both sides, walk the union of paths, emit one
    DiffEntry per differing leaf with the SER-02 Pitfall 3 direction (a)
    blank-preservation skip applied: if `in_memory_value == ''` and the
    on-disk value is filled, that path is skipped (submitter-filled values
    are sacred when the collector returned blank).

- `format_unified_diff(result, on_disk_path)` — converts a `DiffResult` to a
  human-readable unified-diff-style string (D-40 / D-41). Shape:
  `--- on-disk: <path>` / `+++ in-memory: <computed from live MPI fleet>`
  headers; `@@ <JSONPath> @@` hunk markers; `- <old>` / `+ <new>` lines;
  trailing `Remediation:` block listing both the rename and rm hints.
  Values are emitted verbatim (no truncation, no repr-wrapping) so long
  sysctl tuples like `4096\\t87380\\t16777216` round-trip cleanly (D-41).

Architecture notes:

- The `_SENTINEL_ABSENT` module-level object is used to distinguish "field
  absent on this side" from "field present but empty string". This matters
  because the SER-02 blank preservation rule (Pitfall 3 direction (a)) only
  fires when in-memory IS the empty string AND on-disk has a filled value —
  not when in-memory is absent entirely.

- Fingerprint orphan paths use `clients[fingerprint=<repr>]` rather than a
  positional index because the on-disk and in-memory sides may have
  different cardinalities and positional indices would not correspond.
  Sorting by `repr(fp)` defends against `TypeError: '<' not supported
  between instances of 'X' and 'Y'` on mixed-type fingerprint tuples (the
  same defense Phase 3-04 / Plan 04-04 settled on for `_network_signature`).

- DiffEntry is `@dataclass(frozen=True)` so individual entries are immutable
  once constructed; DiffResult.entries is a list to permit Slice 2's call
  site to extend or replace as needed (T-5-01-02 threat register accepted).

Slice 2 (next plan) will import `diff_node_dict_lists` and
`format_unified_diff` from this module — the public API is now LOCKED.

Phase 5.2 / HANDFILL-01 extension (D-60..D-66):

- `diff_node_dict_lists` gains a soft-pair pre-pass (`_soft_pair_orphans`)
  that runs between the existing exact-match indexing (pass-1) and the
  D-46/D-47 orphan emission (pass-3). For each in-memory orphan whose
  fingerprint has at least one '' position in the 7 scalar dotted-key
  positions, the pre-pass looks for the unique on-disk orphan whose
  non-empty scalar positions all align AND whose 4 signature positions
  match exactly (D-61). When the candidate is unique, the two stanzas
  fall through to the leaf-comparison branch and the existing Pitfall
  3(a) SER-02 rule preserves the submitter's hand-filled value. When
  the candidate is ambiguous (D-63), neither side is paired and both
  flow through to orphan emission — never silently conflate distinct
  machines.

- A new module-level `logger = logging.getLogger(__name__)` emits the
  D-60 reverse-direction INFO log: when on-disk is '' at a scalar
  position and the collector finally resolves a non-empty value, the
  diff layer logs `collector resolved <field>=<value>` and emits NO
  DiffEntry. The on-disk file stays unchanged per LIFE-04; the operator
  may hand-edit the YAML to lock the resolved value.

- Public API (`DiffEntry`, `DiffResult`, `diff_node_dict_lists`,
  `format_unified_diff`) is unchanged. `_compute_fingerprint`'s identity
  contract is unchanged. The Pitfall 3(a) SER-02 block is unchanged —
  it now fires on fingerprint-scalar-position cases via the soft-pair
  fall-through, which is the entire point of HANDFILL-01.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterator

from mlpstorage_py.system_description.auto_generator import _FINGERPRINT_KEYS, _resolve_fingerprint_key


# ---------------------------------------------------------------------------
# Module-level logger — added Phase 5.2 / HANDFILL-01 for D-60 reverse-
# direction INFO emission. The logger name resolves to
# `mlpstorage_py.system_description.diff` and matches the existing
# `logging.getLogger(__name__)` convention used elsewhere in the project
# (e.g. mlpstorage_py/mlps_logging.py:130).
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------
__all__ = [
    "DiffEntry",
    "DiffResult",
    "diff_node_dict_lists",
    "format_unified_diff",
]


# ---------------------------------------------------------------------------
# WR-01 fix (Phase 5.2 follow-up): scope D-60 reverse-direction INFO log
# to the 7 scalar fingerprint positions ONLY. Phase 5.2's original
# implementation applied the swallow-and-log rule to every leaf path in
# `_emit_leaf_diffs`, which silently demoted pre-existing drift on
# non-fingerprint leaves (e.g. `friendly_description`) from DiffEntry to
# INFO log. The fingerprint-scalar path set is derived from the 7 string
# entries in `_FINGERPRINT_KEYS` (positions 0-6 — positions 7-10 are
# `(name, callable)` signature tuples).
# ---------------------------------------------------------------------------
_FINGERPRINT_SCALAR_PATHS: frozenset = frozenset(
    k for k in _FINGERPRINT_KEYS if isinstance(k, str)
)


# ---------------------------------------------------------------------------
# Module-level sentinel used to disambiguate "field absent on this side"
# from "field present but empty string". `object()` is a unique identity
# distinct from every value the collector could ever emit.
# ---------------------------------------------------------------------------
_SENTINEL_ABSENT: Any = object()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiffEntry:
    """One row of difference between on-disk and in-memory views.

    `path` is a JSONPath-style dotted/bracketed string (e.g.
    `clients[0].chassis.cpu_model` or `clients[fingerprint=(...)]`).
    `old` and `new` are the on-disk and in-memory leaf values respectively;
    sentinel strings `"<present>"` / `"<absent>"` are used at the
    fingerprint-orphan level when a whole stanza is present on only one side.
    """

    path: str
    old: Any
    new: Any


@dataclass
class DiffResult:
    """Wrapper around `entries: list[DiffEntry]` with a `.empty` property.

    Mutable on purpose: Slice 2's call site may want to extend the entries
    list with synthetic header entries before rendering. T-5-01-02 in the
    plan's threat register accepts this trade-off because the result is
    constructed per-call and not stored in shared state.
    """

    entries: list[DiffEntry] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return len(self.entries) == 0


# ---------------------------------------------------------------------------
# _flatten_to_paths — recursive generator over nested dict/list structures.
# ---------------------------------------------------------------------------


def _flatten_to_paths(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Yield (jsonpath, leaf_value) pairs for every leaf in `value`.

    Dict children: `f"{prefix}.{k}"` if prefix else `k`.
    List children: `f"{prefix}[{i}]"`.
    Empty dict/list: yields nothing.
    Scalars (str/int/float/bool/None) at any level: yields `(prefix, value)`.

    The scalar-at-root case yields `("", value)` so the caller can distinguish
    "empty container" (no entries) from "scalar input" (one entry with empty
    prefix).
    """
    if isinstance(value, dict):
        if not value:
            return
        for k, v in value.items():
            sub_prefix = f"{prefix}.{k}" if prefix else str(k)
            yield from _flatten_to_paths(v, sub_prefix)
    elif isinstance(value, list):
        if not value:
            return
        for i, v in enumerate(value):
            sub_prefix = f"{prefix}[{i}]"
            yield from _flatten_to_paths(v, sub_prefix)
    else:
        # Scalar leaf (str, int, float, bool, None, or any non-container type).
        yield (prefix, value)


# ---------------------------------------------------------------------------
# _compute_fingerprint — reuses Phase-4 11-tuple identity rule.
# ---------------------------------------------------------------------------


def _compute_fingerprint(stanza: dict) -> tuple:
    """Return the 11-tuple fingerprint per D-38 / auto_generator.py.

    This is the IDENTITY function for client stanzas: two stanzas with the
    same fingerprint are considered the same "client class" and their
    field-level differences will be surfaced by leaf comparison; two stanzas
    with different fingerprints surface as orphan entries (D-38 / Pitfall 2).
    """
    return tuple(_resolve_fingerprint_key(stanza, k) for k in _FINGERPRINT_KEYS)


# ---------------------------------------------------------------------------
# _render_fingerprint — verbatim-value fingerprint renderer.
# ---------------------------------------------------------------------------


def _render_fingerprint(fp: tuple) -> str:
    """Render a fingerprint tuple as a string with leaf values shown verbatim.

    The naive `repr(fp)` would escape control characters in string leaves
    (notably tabs in multi-value sysctl leaves like `4096\\t87380\\t16777216`
    per D-41), defeating the round-trip-verbatim contract. This helper walks
    the fingerprint structure and emits each leaf via plain `str()` so the
    bytes appear as-is in the rendered output.

    The rendering is purely cosmetic — fingerprints are still keyed and
    sorted by the tuple itself (which is hashable and ordered by Python's
    native tuple comparison after the `key=repr` defense applied by the
    caller).
    """
    def render(v: Any) -> str:
        if isinstance(v, tuple):
            return "(" + ", ".join(render(x) for x in v) + ")"
        return str(v)

    return render(fp)


# ---------------------------------------------------------------------------
# _soft_pair_orphans — Phase 5.2 / HANDFILL-01 hand-fill affordance.
# ---------------------------------------------------------------------------


def _soft_pair_orphans(
    in_memory_orphans_by_fp: dict[tuple, dict],
    on_disk_orphans_by_fp: dict[tuple, dict],
) -> tuple[list[tuple[tuple, tuple]], list[tuple], list[tuple]]:
    """Soft-pair leftover orphans across hand-filled scalar fingerprint positions.

    Per D-62 pass-2 of `diff_node_dict_lists`:

    For each in-memory orphan whose fingerprint has at least one '' position
    in the 7 SCALAR positions, find the unique on-disk orphan whose
    NON-EMPTY scalar positions all align AND whose 4 SIGNATURE positions
    match exactly (per D-61). If exactly one such on-disk orphan exists AND
    it has not yet been soft-paired, treat the two stanzas as the same
    client.

    Returns:
        (paired_fp_pairs, remaining_in_memory_fps, remaining_on_disk_fps)
        where `paired_fp_pairs` is a list of `(in_memory_fp, on_disk_fp)`
        tuples to feed into the leaf-comparison branch, and the two
        `remaining_*_fps` lists hold the leftover orphans for the existing
        D-46 / D-47 emission path.

    D-63 ambiguity rule (two-phase / CR-01 fix): a pair is committed
    only when (a) the in-memory orphan has exactly ONE candidate AND
    (b) that candidate is uniquely claimed (no other in-memory orphan
    lists it as a candidate). Phase A enumerates all candidate sets
    against the FULL original on-disk pool — no greedy consumption.
    Phase B commits only globally-unique pairings.

    This protects against the original greedy bug: if mem_X uniquely
    matches DiskA, and mem_Y is genuinely ambiguous between DiskA and
    DiskB, a single-pass greedy algorithm would consume DiskA on mem_X
    and then silently pair mem_Y → DiskB. The two-phase algorithm sees
    that DiskA is contested (claim_count=2) and rejects mem_X's pair as
    well — both fall through to orphan emission. Conservative; never
    silently conflate distinct machines.

    D-61 signature-position rule: the 4 callable signature positions
    (positions 7-10 of the fingerprint tuple) require EXACT equality.
    Empty signature `()` does NOT count as wildcard.

    Input contract: `in_memory_orphans_by_fp` and `on_disk_orphans_by_fp`
    MUST be orphan-only (no fingerprints present on both sides) — the
    function does not validate this and a caller violating the contract
    would produce duplicate DiffEntries.
    """
    scalar_positions = [i for i, k in enumerate(_FINGERPRINT_KEYS) if isinstance(k, str)]
    signature_positions = [i for i, k in enumerate(_FINGERPRINT_KEYS) if not isinstance(k, str)]

    # D-22 mixed-type defense: sort by repr so DiffEntry path ordering
    # remains deterministic across runs (matches the all_fps sort in
    # diff_node_dict_lists).
    sorted_mem_fps = sorted(in_memory_orphans_by_fp.keys(), key=repr)
    sorted_disk_fps = sorted(on_disk_orphans_by_fp.keys(), key=repr)

    # --- Phase A: enumerate candidate sets against the ORIGINAL pool ---
    # No consumption; every mem_fp sees the full disk_fp pool.
    candidate_map: dict[tuple, list[tuple]] = {}
    for mem_fp in sorted_mem_fps:
        candidates: list[tuple] = []
        for disk_fp in sorted_disk_fps:
            # D-61 STRICT-SIGNATURE GATE: signatures must match exactly,
            # including the () empty-tuple case (NOT a wildcard).
            if any(mem_fp[i] != disk_fp[i] for i in signature_positions):
                continue

            # SCALAR ALIGNMENT (D-62 forward + D-60 reverse): positions
            # where EITHER side is '' are ignored in the comparison
            # (soft-pair semantics, symmetric). All other scalar positions
            # must match exactly.
            scalar_ok = True
            at_least_one_empty = False
            for i in scalar_positions:
                if mem_fp[i] == "" or disk_fp[i] == "":
                    at_least_one_empty = True
                    continue
                if mem_fp[i] != disk_fp[i]:
                    scalar_ok = False
                    break
            if not scalar_ok:
                continue

            # Eligibility: at least one scalar position empty on some
            # side. Pure exact matches would have hashed identical and
            # been routed through pass-1 already.
            if not at_least_one_empty:
                continue

            candidates.append(disk_fp)

        candidate_map[mem_fp] = candidates

    # --- Phase B: tally claim counts and commit globally-unique pairs --
    disk_claim_count: dict[tuple, int] = {}
    for cands in candidate_map.values():
        for d in cands:
            disk_claim_count[d] = disk_claim_count.get(d, 0) + 1

    paired_fp_pairs: list[tuple[tuple, tuple]] = []
    paired_on_disk_fps: set[tuple] = set()
    paired_in_memory_fps: set[tuple] = set()

    # Iterate in the same sorted order Phase A used so committed pairs
    # are produced in a deterministic order (matches the existing
    # `key=lambda p: repr(p[0])` sort in diff_node_dict_lists).
    for mem_fp in sorted_mem_fps:
        cands = candidate_map[mem_fp]
        if len(cands) != 1:
            continue
        disk_fp = cands[0]
        # D-63: commit only when the candidate is uniquely claimed.
        # claim_count >= 2 means another mem_fp also lists this disk
        # as a candidate (i.e., this disk is contested) — both mems
        # fall through to orphan emission.
        if disk_claim_count[disk_fp] != 1:
            continue
        paired_fp_pairs.append((mem_fp, disk_fp))
        paired_on_disk_fps.add(disk_fp)
        paired_in_memory_fps.add(mem_fp)

    remaining_in_memory_fps = [
        fp for fp in in_memory_orphans_by_fp if fp not in paired_in_memory_fps
    ]
    remaining_on_disk_fps = [
        fp for fp in on_disk_orphans_by_fp if fp not in paired_on_disk_fps
    ]

    return paired_fp_pairs, remaining_in_memory_fps, remaining_on_disk_fps


# ---------------------------------------------------------------------------
# _emit_leaf_diffs — extracted from diff_node_dict_lists for reuse across
# exact-match pairs (pass-1) and soft-paired orphans (pass-2). Phase 5.2.
# ---------------------------------------------------------------------------


def _emit_leaf_diffs(
    on_disk_stanza: dict,
    in_memory_stanza: dict,
    entries: list[DiffEntry],
) -> None:
    """Flatten both stanzas to paths and append a DiffEntry per differing leaf.

    Implements the two empty-side rules:

    - D-60 reverse-direction (Phase 5.2): on-disk == '' and in-memory
      holds a non-empty scalar string AT A FINGERPRINT SCALAR PATH
      (the 7 string entries of `_FINGERPRINT_KEYS`). Collector finally
      learned a value the submitter did NOT hand-fill. Emit an INFO log;
      NO DiffEntry; the on-disk file stays unchanged per LIFE-04. Other
      "" → non-empty changes (non-fingerprint leaves) continue to
      surface as DiffEntries — pre-Phase-5.2 drift semantics preserved.

    - Pitfall 3(a) SER-02 (Phase 5): in-memory == '' and on-disk holds a
      non-empty filled value. Submitter hand-filled the value; collector
      returned blank. Silently keep the submitter's value; NO DiffEntry.

    Otherwise any disk_v != mem_v emits a DiffEntry.
    """
    disk_paths = dict(_flatten_to_paths(on_disk_stanza))
    mem_paths = dict(_flatten_to_paths(in_memory_stanza))

    for path in sorted(set(disk_paths) | set(mem_paths), key=repr):
        disk_v = disk_paths.get(path, _SENTINEL_ABSENT)
        mem_v = mem_paths.get(path, _SENTINEL_ABSENT)

        # D-60 reverse-direction (Phase 5.2 / HANDFILL-01): the collector
        # finally learned a value the user did not hand-fill. On-disk is
        # '', in-memory is non-empty. Emit INFO log; NO DiffEntry; the
        # on-disk file stays unchanged per LIFE-04.
        #
        # WR-01 fix: scope to fingerprint scalar paths only. Phase 5.2
        # broadened this to every leaf, silently swallowing pre-existing
        # drift on non-fingerprint paths (e.g. friendly_description).
        if (
            path in _FINGERPRINT_SCALAR_PATHS
            and disk_v == ""
            and mem_v is not _SENTINEL_ABSENT
            and mem_v != ""
            and isinstance(mem_v, str)
        ):
            # WR-04: use deferred %-style formatting so the formatting
            # cost is paid only when the handler accepts INFO.
            logger.info(
                "collector resolved %s=%r (was \"\" on disk; "
                "on-disk file unchanged per LIFE-04 — manually update the "
                "YAML if you want to lock this value)",
                path, mem_v,
            )
            continue

        # Pitfall 3(a) SER-02 blank preservation: in-memory empty string +
        # on-disk filled non-empty value means the submitter filled it in
        # by hand; the collector returning blank is NOT drift.
        if (
            mem_v == ""
            and disk_v is not _SENTINEL_ABSENT
            and disk_v != ""
        ):
            continue

        if disk_v != mem_v:
            entries.append(DiffEntry(
                path=path,
                old=("<absent>" if disk_v is _SENTINEL_ABSENT else disk_v),
                new=("<absent>" if mem_v is _SENTINEL_ABSENT else mem_v),
            ))


# ---------------------------------------------------------------------------
# diff_node_dict_lists — public diff function.
# ---------------------------------------------------------------------------


def diff_node_dict_lists(on_disk: list[dict], in_memory: list[dict]) -> DiffResult:
    """Compare two lists of client stanzas and return a structured DiffResult.

    Algorithm (D-62 two-pass pairing):
      1. Index each side by fingerprint (`_compute_fingerprint`).
      2. PASS 1 — exact fingerprint match: for each fingerprint present on
         BOTH sides, flatten and compare leaf-by-leaf (existing behavior
         unchanged).
      3. PASS 2 (Phase 5.2 / HANDFILL-01) — soft-pair the leftover orphans
         via `_soft_pair_orphans`. Soft-paired stanzas fall through to the
         same `_emit_leaf_diffs` branch so the existing Pitfall 3(a) SER-02
         rule (in-memory '' + on-disk filled → submitter's value preserved)
         and the new D-60 reverse-direction INFO log both fire.
      4. PASS 3 — surviving orphans (no exact match AND no soft-pair) emit
         the existing D-46 / D-47 `<present>` / `<absent>` entries.

    D-22 mixed-type defense (`key=repr` sort) applied at every layer so
    DiffEntry path ordering remains deterministic across runs.
    """
    on_disk_by_fp: dict[tuple, dict] = {_compute_fingerprint(s): s for s in on_disk}
    in_memory_by_fp: dict[tuple, dict] = {_compute_fingerprint(s): s for s in in_memory}

    common_fps = set(on_disk_by_fp) & set(in_memory_by_fp)
    on_disk_orphan_fps = set(on_disk_by_fp) - common_fps
    in_memory_orphan_fps = set(in_memory_by_fp) - common_fps
    on_disk_orphans_by_fp = {fp: on_disk_by_fp[fp] for fp in on_disk_orphan_fps}
    in_memory_orphans_by_fp = {fp: in_memory_by_fp[fp] for fp in in_memory_orphan_fps}

    paired_fp_pairs, remaining_in_memory_fps, remaining_on_disk_fps = _soft_pair_orphans(
        in_memory_orphans_by_fp, on_disk_orphans_by_fp,
    )

    entries: list[DiffEntry] = []

    # PASS 1: exact-match pairs fall through to leaf comparison.
    for fp in sorted(common_fps, key=repr):
        _emit_leaf_diffs(on_disk_by_fp[fp], in_memory_by_fp[fp], entries)

    # PASS 2: soft-paired orphans fall through to the same branch so
    # Pitfall 3(a) and D-60 fire on hand-fill cases.
    for mem_fp, disk_fp in sorted(paired_fp_pairs, key=lambda p: repr(p[0])):
        _emit_leaf_diffs(on_disk_by_fp[disk_fp], in_memory_by_fp[mem_fp], entries)

    # PASS 3a: D-46 — in-memory orphans (no exact match, no soft-pair).
    # The fingerprint tuple is rendered via `_render_fingerprint` (not
    # `repr(fp)`) so multi-value sysctl leaves like
    # `4096\t87380\t16777216` survive verbatim in the path string —
    # repr() would escape the literal tabs to `\\t` and trip the D-41
    # round-trip lock.
    for fp in sorted(remaining_in_memory_fps, key=repr):
        entries.append(DiffEntry(
            path=f"clients[fingerprint={_render_fingerprint(fp)}]",
            old="<absent>",
            new="<present>",
        ))

    # PASS 3b: D-47 — on-disk orphans. See D-41 note above.
    for fp in sorted(remaining_on_disk_fps, key=repr):
        entries.append(DiffEntry(
            path=f"clients[fingerprint={_render_fingerprint(fp)}]",
            old="<present>",
            new="<absent>",
        ))

    return DiffResult(entries=entries)


# ---------------------------------------------------------------------------
# format_unified_diff — human-readable rendering.
# ---------------------------------------------------------------------------


def format_unified_diff(result: DiffResult, on_disk_path: str) -> str:
    """Render a DiffResult as a unified-diff-style string per D-40 / D-41.

    Output shape:
        --- on-disk: <path>
        +++ in-memory: <computed from live MPI fleet>
        @@ <JSONPath_1> @@
        - <old_1>
        + <new_1>
        @@ <JSONPath_2> @@
        - <old_2>
        + <new_2>
        ...
        <blank line>
        Remediation:
          • Rename the existing yaml and re-run with --systemname <new>
            (a fresh systemname.yaml will be generated)
          • Remove <path> and re-run
            (you will lose hand-filled blanks)

    Values are emitted via plain `str()` (NOT `repr()`) so long sysctl tuples
    like `4096\\t87380\\t16777216` round-trip verbatim (D-41 lock).
    """
    lines: list[str] = [
        f"--- on-disk: {on_disk_path}",
        "+++ in-memory: <computed from live MPI fleet>",
    ]

    for entry in result.entries:
        lines.append(f"@@ {entry.path} @@")
        lines.append(f"- {entry.old}")
        lines.append(f"+ {entry.new}")

    lines.append("")
    lines.append("Remediation:")
    lines.append("  • Rename the existing yaml and re-run with --systemname <new>")
    lines.append("    (a fresh systemname.yaml will be generated)")
    lines.append(f"  • Remove {on_disk_path} and re-run")
    lines.append("    (you will lose hand-filled blanks)")

    return "\n".join(lines)
