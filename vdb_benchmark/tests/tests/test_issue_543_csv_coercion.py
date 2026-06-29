"""
Issue #543: simple_bench.calculate_statistics() must not crash on malformed
per-process CSV rows.

Reporter scenario: 16 worker processes ran a 60s benchmark, finished writing
their per-process ``milvus_benchmark_p*.csv`` files, and statistics
calculation crashed with::

    TypeError: unsupported operand type(s) for +: 'float' and 'str'
        at (all_data["timestamp"] + all_data["batch_time_seconds"]).max()

The root cause was that a small number of rows in two of the CSVs had a
``batch_time_seconds`` value of the string ``True`` instead of a float
duration — once any value in that column is non-numeric, pandas falls back
to ``object`` dtype for the whole column and the arithmetic on line 1042
explodes.

The fix is reader-side defense in ``calculate_statistics()``: coerce the
numeric columns with ``pd.to_numeric(..., errors='coerce')``, drop rows
that fail to coerce, log a clear diagnostic naming the dropped count and
the source files, and proceed with the valid rows. The benchmark run
should succeed as long as enough good rows remain.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

# Allow ``from vdbbench...`` from the source tree without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "vdb_benchmark"))

from vdbbench.simple_bench import calculate_statistics, csv_fields  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows):
    """Write a per-process CSV using the production header order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _good_row(process_id, batch_id, timestamp, batch_size=100,
              batch_time=0.5, avg_query=0.005, success=True):
    return {
        "process_id": process_id,
        "batch_id": batch_id,
        "timestamp": timestamp,
        "batch_size": batch_size,
        "batch_time_seconds": batch_time,
        "avg_query_time_seconds": avg_query,
        "success": success,
    }


# Exact shape the reporter observed in milvus_benchmark_p2.csv and p8.csv:
# process_id empty, batch_id=100, the trailing two numeric fields and
# 'success' empty, batch_time_seconds set to the string 'True'.
_REPORTER_MALFORMED_ROW = {
    "process_id": "",
    "batch_id": 100,
    "timestamp": 0.43958,
    "batch_size": 0.004396,
    "batch_time_seconds": True,
    "avg_query_time_seconds": "",
    "success": "",
}


# ---------------------------------------------------------------------------
# Reproducer: the exact shape from issue #543
# ---------------------------------------------------------------------------


def test_calculate_statistics_recovers_from_reporter_malformed_row(tmp_path):
    """The reporter's exact CSV shape must not crash and must return stats.

    Two worker CSVs (p2 and p8) each contain one malformed row matching the
    issue body, surrounded by good rows. The other workers' CSVs are clean.
    """
    # 14 clean CSVs
    for pid in [0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]:
        _write_csv(
            tmp_path / f"milvus_benchmark_p{pid}.csv",
            [_good_row(pid, b, b * 0.5) for b in range(1, 11)],
        )

    # 2 CSVs with one malformed row each (exact reporter shape)
    for pid in [2, 8]:
        rows = [_good_row(pid, b, b * 0.5) for b in range(1, 11)]
        rows.insert(5, _REPORTER_MALFORMED_ROW)
        _write_csv(tmp_path / f"milvus_benchmark_p{pid}.csv", rows)

    # Must not raise
    stats = calculate_statistics(str(tmp_path))

    # No 'error' key — calculation succeeded
    assert "error" not in stats, (
        f"calculate_statistics returned an error: {stats!r}"
    )

    # Stats should reflect ~16 * 10 = 160 good rows (2 malformed dropped)
    assert stats.get("total_queries", 0) > 0
    assert stats.get("total_time_seconds", 0) > 0


# ---------------------------------------------------------------------------
# Per-malformation coverage — any one bad numeric field should not crash
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_field,bad_value", [
    ("timestamp", "True"),
    ("batch_time_seconds", "True"),
    ("batch_time_seconds", "not-a-number"),
    ("batch_size", "True"),
    ("avg_query_time_seconds", "True"),
])
def test_calculate_statistics_drops_rows_with_non_numeric_values(
    tmp_path, bad_field, bad_value
):
    """A single non-numeric value in any numeric column must not crash."""
    good = [_good_row(0, b, b * 0.5) for b in range(1, 11)]
    malformed = _good_row(0, 99, 99.0)
    malformed[bad_field] = bad_value
    good.insert(5, malformed)

    _write_csv(tmp_path / "milvus_benchmark_p0.csv", good)

    stats = calculate_statistics(str(tmp_path))
    assert "error" not in stats
    assert stats.get("total_queries", 0) > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_calculate_statistics_returns_error_when_every_row_is_malformed(tmp_path):
    """If no good rows remain, return a clear error rather than crash."""
    bad_rows = [_REPORTER_MALFORMED_ROW for _ in range(10)]
    _write_csv(tmp_path / "milvus_benchmark_p0.csv", bad_rows)

    stats = calculate_statistics(str(tmp_path))
    # Should return an error dict, not raise
    assert "error" in stats


def test_calculate_statistics_unchanged_for_clean_csvs(tmp_path):
    """Sanity: clean CSVs still produce a complete stats dict."""
    for pid in range(4):
        _write_csv(
            tmp_path / f"milvus_benchmark_p{pid}.csv",
            [_good_row(pid, b, b * 0.5) for b in range(1, 11)],
        )

    stats = calculate_statistics(str(tmp_path))
    assert "error" not in stats
    # Pre-fix code already produced these on clean input.
    assert stats["total_queries"] > 0
    assert stats["total_time_seconds"] > 0
