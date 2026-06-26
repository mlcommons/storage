"""Shared pure-function helpers for Phase 2 check methods.

This module is LOG-FREE: helpers return status tuples and never call
``log_violation`` or ``self.log.error`` directly (with the exception of
``_check_code_image_layered``, which invokes a caller-supplied
``log_violation_cb`` so the caller's rule ID/name are carried into the
violation message — see CD-04 below). Callers emit violations using the
standard ``BaseCheck.log_violation`` / ``warn_violation`` pattern
(Pitfall #11, PROJECT.md accumulate-don't-abort principle).

Exports:
  DF_HEADER_RE          — compiled regex matching the ``df`` header line (D-B1)
  _check_filesystem_separation — filesystem-separation helper (D-B1..B5)
  _check_code_image_layered    — benchmark-agnostic layered code-image helper
                                  (Phase 4 CD-04; shared by §3.6.1 and §5.6.1)
  _pair_checkpoint_runs — write/read run pairing helper (D-D2)
  _parse_iso_gap        — ISO-timestamp gap helper (D-D2, CHKPT-03)

References:
  - D-B1..B7 in Phase 2 CONTEXT.md (df parsing, longest-prefix mount match)
  - D-D2 in Phase 2 CONTEXT.md (pairing write/read checkpoint runs)
  - Phase 4 CONTEXT.md D-06 / CD-04 (layered helper extraction)
  - RESEARCH.md §Shared Helpers
"""

import datetime
import logging
import os
import re
from pathlib import Path

_LOG = logging.getLogger(__name__)

from ..tools.code_checksum import compute_code_tree_md5
from ..tools.code_image import (
    verify_image_self_consistent,
    CodeImageError,
    MissingHashFile,
    MalformedHashFile,
)


# ---------------------------------------------------------------------------
# df header regex (D-B1, locked)
# ---------------------------------------------------------------------------

# Anchored header: tolerates both `df` (1K-blocks column / "Available") and
# `df -h` (Size column / "Avail") because the second column is matched by \S+
# (any non-whitespace token) and the fourth column accepts "Avail" or "Available".
DF_HEADER_RE = re.compile(
    r"^Filesystem\s+\S+\s+Used\s+Avail\w*\s+Use%\s+Mounted on",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# _check_filesystem_separation
# ---------------------------------------------------------------------------

def _check_filesystem_separation(
    metadata_args: dict,
    logfile_path: str,
) -> tuple[bool, bool]:
    """Verify that data_dir and results_dir are on different filesystems.

    Reads the logfile for a ``df`` output block (D-B1, D-B2). Uses longest-prefix
    matching of realpath(data_dir) and realpath(results_dir) against the mount
    column of each ``df`` row (D-B2).

    Returns a ``(ok, df_found)`` tuple:
      - ``(True,  True)``  — different mounts found (pass) or silent-skip (D-B3)
      - ``(False, True)``  — same mount (violation; caller emits [3.4.2] or [4.4.2])
      - ``(False, False)`` — df block not found / logfile missing (D-B4; caller
                              emits "df output not found" violation)

    **D-B3 silent-skip:** returns ``(True, True)`` when either ``data_dir`` or
    ``results_dir`` is absent from *metadata_args*. The sibling check
    ``mlpstorage_path_args`` / ``checkpointPathArgs`` owns that diagnostic; this
    helper does not double-count.

    **D-B7 note:** the *caller* is responsible for checking ``benchmark_API``
    and only calling this helper when ``benchmark_API == 'file'``. This helper
    does not read ``benchmark_API`` — it has no access to the system YAML.

    **Known limitation:** single-line ``df`` block parse only. Multi-line
    device-name wrapping (some ``df`` versions write the device name on its own
    line when it is too long) is OUT OF SCOPE for this MVP. TODO-001 defines a
    machine-readable ``df`` output contract that will supersede this parser.
    Until then, real submissions with wrapped device names hard-fail with
    "df output not found" (D-B4), which is the desired gap-surfacing behaviour.

    Args:
        metadata_args: The ``metadata["args"]`` dict from a submission log tuple.
            Must contain ``"data_dir"`` and ``"results_dir"`` keys (or their
            checkpointing analogs ``"checkpoint_folder"`` / ``"results_dir"``).
        logfile_path: Absolute path to the ``*_run.stdout.log`` file to scan.

    Returns:
        ``(ok: bool, df_found: bool)``
    """
    data_dir = metadata_args.get("data_dir") or metadata_args.get("checkpoint_folder")
    results_dir = metadata_args.get("results_dir")

    # D-B3: silent-skip when either path is missing
    if not data_dir or not results_dir:
        return (True, True)

    # D-B4: logfile does not exist → df not found
    if not os.path.exists(logfile_path):
        return (False, False)

    with open(logfile_path, "r", errors="replace") as fh:
        content = fh.read()

    # Find the df header
    match = DF_HEADER_RE.search(content)
    if not match:
        return (False, False)

    # Walk lines after the header; collect mount column per row.
    # The regex match ends at the last char of "Mounted on" (before the newline),
    # so content[match.end():] starts with '\n'. We skip that initial newline by
    # starting after the end of the matched line.
    #
    # TODO(TODO-001): the current "scan df output of the log file" approach is
    # planned to be superseded by capturing `stat -f -c '%i' "$data_dir"` per
    # node at runtime — a single scalar FS identity stored alongside per-node
    # metadata, compared for equality across nodes. That removes both this
    # multi-line-device-name parse limitation and the substring-matching
    # fragility called out in WR-06's silent-pass case. Until that migration
    # lands, real submissions with wrapped device names hard-fail with
    # "df output not found" (D-B4), which is the desired gap-surfacing behaviour.
    mounts = []
    header_end = content.find("\n", match.end())  # find the end of the header line
    if header_end == -1:
        return (False, False)  # header is the last line; no rows follow
    rest = content[header_end + 1:]
    for line in rest.splitlines():
        line = line.rstrip()
        if not line:
            break  # blank line ends the df block
        # rsplit with maxsplit=5 handles multi-word mount points
        # (splits from the right: Filesystem, 1K-blocks/Size, Used, Available, Use%, Mounted_on)
        parts = line.rsplit(None, 5)
        if len(parts) < 6:
            break  # malformed / non-df line ends the block
        mounts.append(parts[-1])  # last field is mount point

    if not mounts:
        return (False, False)

    # Realpath both paths (D-B2: longest-prefix match)
    real_data = os.path.realpath(data_dir)
    real_results = os.path.realpath(results_dir)

    def _best_mount(realpath: str) -> str | None:
        """Return the longest mount column that is a prefix of *realpath*."""
        best = None
        best_len = -1
        for mount in mounts:
            # Ensure the mount is a proper path prefix (add trailing / to avoid
            # matching /data against /data2)
            if realpath == mount or realpath.startswith(mount.rstrip("/") + "/"):
                if len(mount) > best_len:
                    best = mount
                    best_len = len(mount)
        return best

    data_mount = _best_mount(real_data)
    results_mount = _best_mount(real_results)

    # If either path cannot be matched to a mount → indeterminate.
    # Emit a warning so the gap is grep-visible (a typo'd data_dir would
    # otherwise silent-pass this check). The pass return is preserved so we
    # don't false-positive on weird mount tables that nonetheless contain
    # a legitimate data_dir / results_dir pair the regex can't resolve.
    if data_mount is None or results_mount is None:
        _LOG.warning(
            "_check_filesystem_separation: could not match data_dir=%s "
            "(realpath %s) or results_dir=%s (realpath %s) to any df mount "
            "in %s; treating as pass (data_mount=%s, results_mount=%s)",
            data_dir, real_data, results_dir, real_results, logfile_path,
            data_mount, results_mount,
        )
        return (True, True)

    # Same mount → violation
    return (data_mount != results_mount, True)


# ---------------------------------------------------------------------------
# _check_code_image_layered (Phase 4 CD-04)
# ---------------------------------------------------------------------------

def _check_code_image_layered(
    code_path: str,
    division: str,
    expected: str | None,
    log,
    log_violation_cb,
    rule_id: str,
    rule_name: str,
) -> bool:
    """Benchmark-agnostic layered code-image check (self-consistency + upstream-identity).

    Mirrors the two inner branches of STRUCT-06
    (``submission_structure_checks.code_directory_contents_check``) so the
    same layered model is enforced under multiple rule IDs without duplicating
    the implementation across check classes:

      * ``2.1.6 codeDirectoryContents`` — STRUCT-06 itself, calls
        ``self.log_violation`` directly with its own ID/name.
      * ``3.6.1 trainingClosedSubmissionChecksum`` — TrainingCheck, calls this
        helper with the 3.6.1 rule ID/name pair.
      * ``5.6.1 vdbClosedSubmissionChecksum``     — VdbCheck, calls this helper
        with the 5.6.1 rule ID/name pair.

    The duplication of rule IDs is intentional (Phase 4 D-06): downstream
    tooling must be able to tell whether a code-image mismatch fired under
    §2.1.6 (structural), §3.6.1 (Training CLOSED), or §5.6.1 (VDB CLOSED). The
    *implementation* of the check is unified here (CD-04); the *attribution*
    stays per-rule via the caller-supplied ``rule_id`` / ``rule_name``.

    The helper performs the same two-step check defined for STRUCT-06 at
    ``submission_structure_checks.py:442-470``:

      1. Self-consistency: try ``verify_image_self_consistent(code_path, log)``.
         If it returns False, log a violation and set ``valid = False``. Catch
         ``MissingHashFile`` / ``MalformedHashFile`` / ``CodeImageError`` and log
         the exception message as a violation.
      2. Upstream-identity (CLOSED only, D-06 + D-07): if ``division == "closed"``
         AND ``expected is not None``, compute ``compute_code_tree_md5`` and
         compare against ``expected``. Mismatch → log a violation.

    Args:
        code_path: Absolute on-disk path to the ``code/`` directory to validate.
        division: ``"closed"`` or ``"open"``. The upstream-identity branch fires
            only for ``"closed"`` (matches STRUCT-06 L467 + D-06).
        expected: The reference digest returned by
            ``Config.get_reference_checksum()``. ``None`` means upstream-identity
            is skipped (matches STRUCT-06 L417 + D-12 single-warning behavior).
        log: Logger instance, passed through to ``verify_image_self_consistent``
            and ``compute_code_tree_md5``.
        log_violation_cb: A callable with the same signature as
            ``BaseCheck.log_violation`` —
            ``(rule_id, rule_name, path, fmt, *args)``. Decoupling the helper
            from a specific check class is what makes it benchmark-agnostic.
        rule_id: The caller's Rules.md rule ID (e.g., ``"3.6.1"``, ``"5.6.1"``).
            Passed through to every ``log_violation_cb`` call so violations
            carry the CALLER's rule ID, not a generic helper ID.
        rule_name: The caller's camelCase Rules.md rule name (e.g.,
            ``"trainingClosedSubmissionChecksum"``, ``"vdbClosedSubmissionChecksum"``).

    Returns:
        ``True`` if every branch passed; ``False`` if any violation was logged.
    """
    valid = True
    # When .code-hash.json is absent, the per-tree integrity anchor does not
    # exist — the upstream-identity branch would re-walk the entire tree and
    # log a SECOND, contradictory violation per leaf with no diagnostic value
    # over the first ("missing .code-hash.json"). MalformedHashFile and
    # CodeImageError are different: the JSON parses but the hash mismatches
    # or refers to an absent root — keep dual-violation behavior for those
    # so the upstream-identity walk still adds signal.
    hashfile_present = True

    # 1. Self-consistency branch (STRUCT-06 L448-L464 analog).
    try:
        if not verify_image_self_consistent(Path(code_path), log):
            log_violation_cb(
                rule_id, rule_name, code_path,
                "code tree hash does not match .code-hash.json at %s",
                code_path,
            )
            valid = False
    except MissingHashFile as e:
        hashfile_present = False
        log_violation_cb(
            rule_id, rule_name, code_path,
            "%s", str(e),
        )
        valid = False
    except (MalformedHashFile, CodeImageError) as e:
        log_violation_cb(
            rule_id, rule_name, code_path,
            "%s", str(e),
        )
        valid = False

    # 2. Upstream-identity branch (STRUCT-06 L466-L476 analog; CLOSED + expected only).
    # Skip the O(tree) re-walk when no .code-hash.json anchored step 1 — the
    # caller already knows the leaf is broken; a redundant violation here
    # just adds noise without surfacing new information.
    if division == "closed" and expected is not None and hashfile_present:
        digest = compute_code_tree_md5(code_path, log)
        if digest != expected:
            log_violation_cb(
                rule_id, rule_name, code_path,
                "code tree MD5 mismatch: expected %s, got %s",
                expected, digest,
            )
            valid = False

    return valid


# ---------------------------------------------------------------------------
# _pair_checkpoint_runs
# ---------------------------------------------------------------------------

def _pair_checkpoint_runs(summaries: list) -> list[tuple]:
    """Pair write-only and read-only checkpoint runs by timestamp order.

    A "write-only" run has ``num_checkpoints_write > 0`` AND
    ``num_checkpoints_read == 0``. A "read-only" run has the reverse.
    "Combined" runs (both > 0) are silently dropped — they do not participate
    in the write→read pair validation.

    **Known limitation (D-D2, Gray Area 2):** when write_runs and read_runs
    have unequal lengths, ``zip()`` truncates to the shorter list. The caller
    (CHKPT-02, CHKPT-03) surfaces the missing-run diagnostic via the
    timestamp-presence check rather than a count-mismatch error here. This
    is documented and intentional — do not change without updating D-D2.

    Args:
        summaries: A list of ``(summary_dict, metadata_dict, timestamp_str)``
            tuples as yielded by ``Loader.load()``'s checkpoint branch.

    Returns:
        A sorted list of ``(write_entry, read_entry)`` tuples where each entry
        is the original ``(summary, metadata, timestamp)`` triple. Sorted by
        write-entry timestamp (lexicographic, correct for ``YYYYMMDD_HHmmss``
        format). Returns ``[]`` if no split-mode (write-only + read-only) runs
        are found.
    """
    write_runs = []
    read_runs = []

    for entry in summaries:
        _summary, metadata, _ts = entry
        if metadata is None:
            # Defensive: treat None metadata as combined-mode (both == 0)
            continue
        args = metadata.get("args", {}) or {}
        n_write = int(args.get("num_checkpoints_write", 0) or 0)
        n_read = int(args.get("num_checkpoints_read", 0) or 0)

        if n_write > 0 and n_read == 0:
            write_runs.append(entry)
        elif n_read > 0 and n_write == 0:
            read_runs.append(entry)
        # else: combined or degenerate — silently drop

    if not write_runs and not read_runs:
        return []

    # Sort by timestamp string (lexicographic order is chronological for YYYYMMDD_HHmmss)
    write_runs.sort(key=lambda e: e[2])
    read_runs.sort(key=lambda e: e[2])

    return list(zip(write_runs, read_runs))


# ---------------------------------------------------------------------------
# _parse_iso_gap
# ---------------------------------------------------------------------------

def _parse_iso_gap(start_str: str, end_str: str) -> float:
    """Parse two ISO-format timestamps and return (end - start) in seconds.

    Accepts both space-separated (``"YYYY-MM-DD HH:MM:SS"``) and ISO T-form
    (``"YYYY-MM-DDTHH:MM:SS"``). Mirrors the ``datetime.fromisoformat`` usage
    in ``directory_checks.py`` lines 250/377.

    For Python 3.11+ ``datetime.fromisoformat`` natively accepts both forms.
    For Python 3.10 compatibility the space form is normalised to T-form before
    parsing.

    Args:
        start_str: Start timestamp string.
        end_str: End timestamp string.

    Returns:
        float — duration in seconds (may be negative if end < start).

    Raises:
        ValueError: if either string cannot be parsed as an ISO timestamp.
    """
    def _parse(s: str) -> datetime.datetime:
        try:
            return datetime.datetime.fromisoformat(s)
        except ValueError:
            # Python 3.10 compat: normalise space separator to 'T'
            normalized = s.replace(" ", "T")
            return datetime.datetime.fromisoformat(normalized)

    start = _parse(start_str)
    end = _parse(end_str)
    return (end - start).total_seconds()
