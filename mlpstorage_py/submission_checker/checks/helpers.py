"""Shared pure-function helpers for Phase 2 check methods.

This module is LOG-FREE: helpers return status tuples and never call
``log_violation`` or ``self.log.error`` directly. Callers emit violations
using the standard ``BaseCheck.log_violation`` / ``warn_violation`` pattern
(Pitfall #11, PROJECT.md accumulate-don't-abort principle).

Exports:
  DF_HEADER_RE          — compiled regex matching the ``df`` header line (D-B1)
  _check_filesystem_separation — filesystem-separation helper (D-B1..B5)
  _pair_checkpoint_runs — write/read run pairing helper (D-D2)
  _parse_iso_gap        — ISO-timestamp gap helper (D-D2, CHKPT-03)

References:
  - D-B1..B7 in Phase 2 CONTEXT.md (df parsing, longest-prefix mount match)
  - D-D2 in Phase 2 CONTEXT.md (pairing write/read checkpoint runs)
  - RESEARCH.md §Shared Helpers
"""

import datetime
import os
import re


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

    # If either path cannot be matched to a mount → cannot determine violation; pass
    if data_mount is None or results_mount is None:
        return (True, True)

    # Same mount → violation
    return (data_mount != results_mount, True)


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
