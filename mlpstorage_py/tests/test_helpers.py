"""Tests for mlpstorage_py.submission_checker.checks.helpers.

Covers:
  - _check_filesystem_separation (D-B1..B5)
  - _pair_checkpoint_runs (D-D2)
  - _parse_iso_gap (D-D2, CHKPT-03)

Run with:
    pytest mlpstorage_py/tests/test_helpers.py -v
"""

import os
import sys

import pytest

from mlpstorage_py.submission_checker.checks.helpers import (
    DF_HEADER_RE,
    _check_filesystem_separation,
    _pair_checkpoint_runs,
    _parse_iso_gap,
)


# ---------------------------------------------------------------------------
# Helpers for building df logfiles
# ---------------------------------------------------------------------------

_DF_HEADER = "Filesystem     1K-blocks  Used  Available  Use%  Mounted on\n"
_DF_HEADER_H = "Filesystem      Size  Used Avail Use% Mounted on\n"


def _make_logfile(tmp_path, content: str, name: str = "run.stdout.log") -> str:
    """Write *content* to a log file in *tmp_path* and return the path."""
    path = str(tmp_path / name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _df_block(*mount_rows: tuple[str, str]) -> str:
    """Build a df output block from (device, mount) pairs."""
    lines = [_DF_HEADER]
    for device, mount in mount_rows:
        lines.append(f"{device}      100        50    50         50%   {mount}\n")
    lines.append("\n")  # blank line terminates block
    return "".join(lines)


# ---------------------------------------------------------------------------
# _check_filesystem_separation
# ---------------------------------------------------------------------------


class TestCheckFilesystemSeparation:
    """Tests for _check_filesystem_separation (D-B3..B5, D-B1, D-B2)."""

    # D-B3: silent-skip when paths are missing
    def test_silent_skip_missing_data_dir(self):
        """Returns (True, True) when data_dir / checkpoint_folder is absent (D-B3)."""
        ok, df_found = _check_filesystem_separation({"results_dir": "/r"}, "/no/log")
        assert (ok, df_found) == (True, True)

    def test_silent_skip_missing_results_dir(self):
        """Returns (True, True) when results_dir is absent (D-B3)."""
        ok, df_found = _check_filesystem_separation({"data_dir": "/d"}, "/no/log")
        assert (ok, df_found) == (True, True)

    # D-B4: df not found
    def test_df_not_found_logfile_missing(self, tmp_path):
        """Returns (False, False) when the logfile does not exist (D-B4)."""
        ok, df_found = _check_filesystem_separation(
            {"data_dir": "/d", "results_dir": "/r"},
            str(tmp_path / "nonexistent.log"),
        )
        assert (ok, df_found) == (False, False)

    def test_df_not_found_logfile_exists_no_header(self, tmp_path):
        """Returns (False, False) when logfile exists but has no df header (D-B4)."""
        logfile = _make_logfile(tmp_path, "Some log output\nNo df block here\n")
        ok, df_found = _check_filesystem_separation(
            {"data_dir": "/d", "results_dir": "/r"}, logfile
        )
        assert (ok, df_found) == (False, False)

    # Same mount → violation
    def test_same_mount_returns_false(self, tmp_path):
        """Both paths on /data mount → (False, True) — violation (D-B2)."""
        content = "Preamble\n" + _df_block(("/dev/sda1", "/data"), ("/dev/sda2", "/"))
        logfile = _make_logfile(tmp_path, content)
        ok, df_found = _check_filesystem_separation(
            {"data_dir": "/data/foo", "results_dir": "/data/bar"}, logfile
        )
        assert (ok, df_found) == (False, True)

    # Different mounts → pass
    def test_different_mounts_returns_true(self, tmp_path):
        """data_dir on /data, results_dir on / → (True, True) — pass (D-B2)."""
        content = _df_block(("/dev/sda1", "/data"), ("/dev/sda2", "/"))
        logfile = _make_logfile(tmp_path, content)
        ok, df_found = _check_filesystem_separation(
            {"data_dir": "/data/foo", "results_dir": "/var/results"}, logfile
        )
        assert (ok, df_found) == (True, True)

    # Longest-prefix discipline (D-B2)
    def test_longest_prefix_match(self, tmp_path):
        """data_dir under /data/fast → longest prefix wins over /data (D-B2)."""
        content = _df_block(
            ("/dev/sda1", "/"),
            ("/dev/sda2", "/data"),
            ("/dev/sda3", "/data/fast"),
        )
        logfile = _make_logfile(tmp_path, content)
        # data_dir → /data/fast, results_dir → /data (different → pass)
        ok, df_found = _check_filesystem_separation(
            {"data_dir": "/data/fast/x", "results_dir": "/data/y"}, logfile
        )
        assert (ok, df_found) == (True, True)

    # df -h tolerance (D-B1, CR-01): DF_HEADER_RE matches both `df` (1K-blocks +
    # "Available") and `df -h` (Size + "Avail") output. Real `df -h` on Linux
    # emits "Avail" — the regex's fourth column uses `Avail\w*` to accept both
    # spellings, and the second column uses `\S+` for "1K-blocks" vs "Size".
    def test_df_h_format_tolerated(self, tmp_path):
        """Real `df -h` output with 'Size' + 'Avail' matches (D-B1 / CR-01).

        Real Linux `df -h` produces literally "Avail", not "Available". This
        test pins that exact header form so the regex never silently regresses
        back to a "Available"-only literal.
        """
        content = (
            "Filesystem      Size  Used Avail Use% Mounted on\n"
            + "/dev/sda1      100G  50G  50G  50%  /data\n"
            + "/dev/sda2      100G  50G  50G  50%  /\n"
            + "\n"
        )
        logfile = _make_logfile(tmp_path, content)
        ok, df_found = _check_filesystem_separation(
            {"data_dir": "/data/foo", "results_dir": "/data/bar"}, logfile
        )
        assert (ok, df_found) == (False, True)

    def test_df_long_format_tolerated(self, tmp_path):
        """Original `df` (no -h) with '1K-blocks' + 'Available' still matches (D-B1)."""
        content = (
            "Filesystem     1K-blocks  Used  Available  Use%  Mounted on\n"
            + "/dev/sda1       1000000   500000   500000   50%   /data\n"
            + "/dev/sda2       1000000   500000   500000   50%   /\n"
            + "\n"
        )
        logfile = _make_logfile(tmp_path, content)
        ok, df_found = _check_filesystem_separation(
            {"data_dir": "/data/foo", "results_dir": "/data/bar"}, logfile
        )
        assert (ok, df_found) == (False, True)

    def test_df_header_at_eof_no_newline(self, tmp_path):
        """Header without trailing newline returns (False, False), no crash (CR-02).

        Prior to the fix, content.index('\\n', match.end()) raised ValueError
        when the df header was the last line with no rows. The fix uses
        str.find() and returns (False, False) so the caller emits the standard
        'df output not found' violation.
        """
        content = "Filesystem      Size  Used Avail Use% Mounted on"  # no \n
        logfile = _make_logfile(tmp_path, content)
        ok, df_found = _check_filesystem_separation(
            {"data_dir": "/data/foo", "results_dir": "/data/bar"}, logfile
        )
        assert (ok, df_found) == (False, False)

    # checkpoint_folder analog
    def test_checkpoint_folder_key(self, tmp_path):
        """Accepts 'checkpoint_folder' as the data-path key (CHKPT-06 surface)."""
        content = _df_block(("/dev/sda1", "/ckpt"), ("/dev/sda2", "/"))
        logfile = _make_logfile(tmp_path, content)
        ok, df_found = _check_filesystem_separation(
            {"checkpoint_folder": "/ckpt/model", "results_dir": "/var/results"}, logfile
        )
        assert (ok, df_found) == (True, True)

    # Symlink resolution (D-B2) — Linux only
    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks not portable on Windows")
    def test_symlink_resolves_to_mount(self, tmp_path):
        """Realpath follows symlinks when matching mounts (D-B2)."""
        # Create /data/real and a symlink /data/link → /data/real
        real_dir = tmp_path / "data" / "real"
        real_dir.mkdir(parents=True)
        link_dir = tmp_path / "data" / "link"
        os.symlink(str(real_dir), str(link_dir))

        # Both data_dir (via symlink) and results_dir (direct) resolve to the same
        # /data mount. Build a df block with the tmp_path /data prefix as the mount.
        data_mount = str(tmp_path / "data")
        content = _df_block(("/dev/sda1", data_mount), ("/dev/sda2", "/"))
        logfile = _make_logfile(tmp_path, content)

        ok, df_found = _check_filesystem_separation(
            {"data_dir": str(link_dir), "results_dir": str(real_dir)}, logfile
        )
        # Both resolve to the same mount → violation
        assert (ok, df_found) == (False, True)


# ---------------------------------------------------------------------------
# _pair_checkpoint_runs
# ---------------------------------------------------------------------------


def _make_entry(
    num_write: int,
    num_read: int,
    ts: str,
    extra_args: dict | None = None,
) -> tuple:
    """Build a (summary, metadata, timestamp) tuple for checkpoint-run pairing tests."""
    args = {"num_checkpoints_write": num_write, "num_checkpoints_read": num_read}
    if extra_args:
        args.update(extra_args)
    return ({}, {"args": args}, ts)


class TestPairCheckpointRuns:
    """Tests for _pair_checkpoint_runs (D-D2)."""

    def test_no_split_mode_runs_returns_empty(self):
        """All combined-mode entries → returns [] (D-D2)."""
        entries = [
            _make_entry(10, 10, "20250101_120000"),
            _make_entry(10, 10, "20250101_130000"),
        ]
        assert _pair_checkpoint_runs(entries) == []

    def test_single_pair(self):
        """One write-only + one read-only → one pair (D-D2)."""
        write = _make_entry(10, 0, "20250101_120000")
        read = _make_entry(0, 10, "20250101_130000")
        result = _pair_checkpoint_runs([write, read])
        assert len(result) == 1
        assert result[0] == (write, read)

    def test_multiple_pairs_sorted_by_timestamp(self):
        """Multiple pairs sorted by write timestamp (D-D2)."""
        write1 = _make_entry(10, 0, "20250101_120000")
        write2 = _make_entry(10, 0, "20250101_140000")
        read1 = _make_entry(0, 10, "20250101_130000")
        read2 = _make_entry(0, 10, "20250101_150000")
        result = _pair_checkpoint_runs([write2, read1, write1, read2])
        assert result == [(write1, read1), (write2, read2)]

    def test_unequal_counts_zip_truncates(self):
        """2 write-only, 1 read-only → 1 pair (zip truncation per Gray Area 2)."""
        write1 = _make_entry(10, 0, "20250101_120000")
        write2 = _make_entry(10, 0, "20250101_140000")
        read1 = _make_entry(0, 10, "20250101_130000")
        result = _pair_checkpoint_runs([write1, write2, read1])
        assert len(result) == 1
        assert result[0] == (write1, read1)

    def test_combined_entries_ignored_split_entries_paired(self):
        """Combined-mode entry is dropped; write-only + read-only are paired (D-D2)."""
        combined = _make_entry(10, 10, "20250101_110000")
        write = _make_entry(10, 0, "20250101_120000")
        read = _make_entry(0, 10, "20250101_130000")
        result = _pair_checkpoint_runs([combined, write, read])
        assert len(result) == 1
        assert result[0] == (write, read)

    def test_none_metadata_treated_as_combined_mode(self):
        """Entry with metadata=None is silently dropped (defensive D-D2)."""
        none_entry = ({}, None, "20250101_110000")
        write = _make_entry(10, 0, "20250101_120000")
        read = _make_entry(0, 10, "20250101_130000")
        result = _pair_checkpoint_runs([none_entry, write, read])
        assert len(result) == 1

    def test_all_combined_returns_empty_list(self):
        """Returns [] not None when there are no split-mode runs."""
        result = _pair_checkpoint_runs([])
        assert result == []


# ---------------------------------------------------------------------------
# _parse_iso_gap
# ---------------------------------------------------------------------------


class TestParseIsoGap:
    """Tests for _parse_iso_gap (D-D2)."""

    def test_space_form_parses(self):
        """Space-separated form 'YYYY-MM-DD HH:MM:SS' is parsed correctly."""
        gap = _parse_iso_gap("2025-01-11 14:30:22", "2025-01-11 14:35:45")
        assert gap == pytest.approx(323.0)  # 5*60 + 23

    def test_t_form_parses(self):
        """ISO T-form 'YYYY-MM-DDTHH:MM:SS' is parsed correctly."""
        gap = _parse_iso_gap("2025-01-11T14:30:22", "2025-01-11T14:35:45")
        assert gap == pytest.approx(323.0)

    def test_invalid_start_raises_value_error(self):
        """Unparseable start string raises ValueError."""
        with pytest.raises(ValueError):
            _parse_iso_gap("garbage", "2025-01-11 14:35:45")

    def test_invalid_end_raises_value_error(self):
        """Unparseable end string raises ValueError."""
        with pytest.raises(ValueError):
            _parse_iso_gap("2025-01-11 14:30:22", "not-a-date")

    def test_negative_gap_allowed(self):
        """Returns negative float when end < start (directionality is caller's concern)."""
        gap = _parse_iso_gap("2025-01-11 14:35:45", "2025-01-11 14:30:22")
        assert gap == pytest.approx(-323.0)

    def test_zero_gap(self):
        """Returns 0.0 when start == end."""
        gap = _parse_iso_gap("2025-01-11 14:30:00", "2025-01-11 14:30:00")
        assert gap == pytest.approx(0.0)

    def test_large_gap_hours(self):
        """Correctly handles multi-hour gaps."""
        gap = _parse_iso_gap("2025-01-11 00:00:00", "2025-01-11 02:00:00")
        assert gap == pytest.approx(2 * 3600.0)
