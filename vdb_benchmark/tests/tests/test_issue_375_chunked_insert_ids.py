"""
Regression tests for issue #375:

    vdb benchmark shows a very low recall@10 because the flat_gt
    collection size is too small.

Root cause: ``load_vdb.insert_data`` computed primary keys as
``range(batch_start, batch_end)`` based on the *chunk-local* index,
so every chunk re-used IDs ``0..chunk_size-1``. With ``num_vectors=1M``
and ``chunk_size=10k`` the source collection ended up with only 10k
unique PKs (and 99 duplicates per PK), which in turn made the
``flat_gt`` collection only 10k rows — about 1% of the source — and
drove recall@10 down to ~0.009.

The fix adds a ``start_id`` offset to ``insert_data`` and threads a
running ``global_id_offset`` through the chunked path in ``main``.
These tests verify the IDs are globally unique across chunks, and that
the legacy default (``start_id=0``) still works for the single-chunk
path.
"""
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Make the package importable regardless of where pytest is invoked from.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# We import the function under test from the real module. We do NOT import
# the module-level argparse / Milvus connect code — those run only inside
# ``main()``. The import itself is cheap.
from vdbbench.load_vdb import insert_data  # noqa: E402


def _captured_ids(mock_collection):
    """Concatenate every IDs list passed to ``collection.insert``."""
    captured = []
    for call in mock_collection.insert.call_args_list:
        # ``insert`` is called as ``collection.insert([ids, batch_vectors])``.
        args, _kwargs = call
        payload = args[0]
        ids = payload[0]
        captured.extend(list(ids))
    return captured


class TestInsertDataIdOffset:
    """Verify primary-key uniqueness across chunked inserts."""

    def test_default_start_id_preserves_legacy_behavior(self):
        """When ``start_id`` is omitted, IDs start at 0 — same as before #375."""
        collection = MagicMock()
        vectors = np.zeros((100, 8), dtype=np.float32)

        total, _elapsed = insert_data(collection, vectors, batch_size=25)

        assert total == 100
        ids = _captured_ids(collection)
        assert ids == list(range(0, 100))

    def test_start_id_offsets_all_batches(self):
        """A non-zero ``start_id`` shifts every batch's IDs by that offset."""
        collection = MagicMock()
        vectors = np.zeros((50, 4), dtype=np.float32)

        insert_data(collection, vectors, batch_size=10, start_id=1000)

        ids = _captured_ids(collection)
        assert ids == list(range(1000, 1050))

    def test_three_chunks_produce_globally_unique_ids(self):
        """
        Exact reproduction of issue #375: simulate the chunked path in
        ``main()`` with three chunks. Before the fix, every chunk re-used
        IDs 0..chunk_size-1 and the union had only ``chunk_size`` unique
        values; after the fix the union has ``3 * chunk_size`` unique values.
        """
        collection = MagicMock()
        chunk_size = 1000
        batch_size = 100
        num_chunks = 3

        global_offset = 0
        for _ in range(num_chunks):
            chunk = np.zeros((chunk_size, 4), dtype=np.float32)
            insert_data(collection, chunk, batch_size=batch_size, start_id=global_offset)
            global_offset += chunk_size

        ids = _captured_ids(collection)
        assert len(ids) == num_chunks * chunk_size
        # The critical assertion the original code would fail:
        assert len(set(ids)) == num_chunks * chunk_size, (
            "Duplicate primary keys across chunks — issue #375 regression."
        )
        assert min(ids) == 0
        assert max(ids) == num_chunks * chunk_size - 1

    def test_uneven_final_chunk(self):
        """The final chunk is usually smaller than ``chunk_size``."""
        collection = MagicMock()
        # 2500 vectors total, chunks of 1000 → 1000, 1000, 500
        chunks = [1000, 1000, 500]
        global_offset = 0
        for n in chunks:
            chunk = np.zeros((n, 4), dtype=np.float32)
            insert_data(collection, chunk, batch_size=300, start_id=global_offset)
            global_offset += n

        ids = _captured_ids(collection)
        assert ids == list(range(0, 2500))
        assert len(set(ids)) == 2500

    def test_batch_size_larger_than_chunk(self):
        """``batch_size`` >= len(vectors) should still produce one batch with the offset applied."""
        collection = MagicMock()
        vectors = np.zeros((42, 4), dtype=np.float32)

        insert_data(collection, vectors, batch_size=1000, start_id=500)

        assert collection.insert.call_count == 1
        ids = _captured_ids(collection)
        assert ids == list(range(500, 542))


class TestFlatGtCoverageGuard:
    """
    Sanity-check the *intent* of the coverage guard added to
    ``enhanced_bench.create_flat_collection``: a flat_gt collection
    that covers far fewer entities than the source should be flagged.

    We assert the threshold here rather than invoking Milvus, so this
    test runs in CI with no external dependencies.
    """

    @pytest.mark.parametrize(
        "flat_count,source_count,should_pass",
        [
            (1_000_000, 1_000_000, True),    # exact match
            (995_000, 1_000_000, True),      # 99.5%, within tolerance
            (10_000, 1_000_000, False),      # the issue #375 failure mode
            (100_000, 1_000_000, False),     # only 10%, still wrong
            (0, 1_000_000, False),           # empty
        ],
    )
    def test_coverage_threshold(self, flat_count, source_count, should_pass):
        coverage = flat_count / source_count if source_count else 0.0
        passes = coverage >= 0.99
        assert passes is should_pass
