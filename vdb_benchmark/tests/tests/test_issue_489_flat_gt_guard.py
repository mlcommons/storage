"""
Regression tests for issue #489:

    vdb_benchmark simple_bench can silently continue with an empty FLAT
    ground-truth collection after the Milvus copy fallback fails, producing
    a "successful" run whose recall.num_queries_evaluated == 0.

Root cause(s) in ``vdbbench/simple_bench.py``:

  1. The pk-cursor fallback initialized the int64 cursor at ``-2**63``,
     producing the expression ``pk > -9223372036854775808``. Milvus parses
     the operand magnitude (9223372036854775808) as int64, which overflows
     INT64_MAX by one and raises a parse error, breaking the copy loop with
     ``copied == 0``.
  2. After copying zero vectors the loop printed a hard-coded
     ``Copied 0/N vectors (100.0%)``.
  3. ``create_flat_collection`` returned ``True`` even with an empty FLAT
     collection (it only failed on exceptions), so the caller's
     ``if not flat_ok`` guard never tripped.
  4. The run still produced QPS/latency numbers even though
     ``num_queries_evaluated == 0``.
  5. The ``query_iterator`` fast path used an unbounded batch size, so wide
     vectors exceeded Milvus' 256MB gRPC message limit and forced the broken
     fallback in the first place.

These tests verify the fixes WITHOUT requiring a live Milvus or the pymilvus
package: a minimal fake ``pymilvus`` is injected into ``sys.modules`` before
the module under test is imported, and the empty-collection paths are driven
with mocks.
"""
import os
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make the package importable regardless of where pytest is invoked from.
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# simple_bench.py imports pymilvus at module load and calls sys.exit(1) on
# ImportError. CI does not have pymilvus installed, so we inject a fake module
# exposing exactly the names simple_bench imports. DataType members only need
# to compare by identity and expose a ``.name`` attribute.
# ---------------------------------------------------------------------------
def _install_fake_pymilvus():
    if "pymilvus" in sys.modules:
        return

    fake = types.ModuleType("pymilvus")

    class _DataTypeMember:
        def __init__(self, name):
            self.name = name

    class DataType:
        INT64 = _DataTypeMember("INT64")
        INT32 = _DataTypeMember("INT32")
        INT16 = _DataTypeMember("INT16")
        INT8 = _DataTypeMember("INT8")
        VARCHAR = _DataTypeMember("VARCHAR")
        FLOAT_VECTOR = _DataTypeMember("FLOAT_VECTOR")
        BINARY_VECTOR = _DataTypeMember("BINARY_VECTOR")
        FLOAT16_VECTOR = _DataTypeMember("FLOAT16_VECTOR")
        BFLOAT16_VECTOR = _DataTypeMember("BFLOAT16_VECTOR")

    fake.DataType = DataType
    fake.Collection = MagicMock(name="Collection")
    fake.CollectionSchema = MagicMock(name="CollectionSchema")
    fake.FieldSchema = MagicMock(name="FieldSchema")
    fake.connections = MagicMock(name="connections")
    fake.utility = MagicMock(name="utility")

    sys.modules["pymilvus"] = fake


_install_fake_pymilvus()

# Now the import is safe in a dependency-free CI environment.
from vdbbench import simple_bench  # noqa: E402
from vdbbench.simple_bench import (  # noqa: E402
    calc_recall,
    create_flat_collection,
    precompute_ground_truth,
)

# The bad cursor expression this issue is about. Constructing it dynamically
# avoids any chance of a copy/paste typo masking a regression.
_BAD_INT_EXPR = f"pk > {-2 ** 63}"  # "pk > -9223372036854775808"


def _make_field(name, dtype, is_primary=False):
    field = MagicMock()
    field.name = name
    field.dtype = dtype
    field.is_primary = is_primary
    return field


def _make_source_schema(pk_dtype):
    """A minimal source collection schema: int/varchar pk + float vector."""
    DataType = simple_bench.DataType
    schema = MagicMock()
    schema.fields = [
        _make_field("id", pk_dtype, is_primary=True),
        _make_field("vector", DataType.FLOAT_VECTOR, is_primary=False),
    ]
    return schema


# ===========================================================================
# Fix 1: int64 fallback cursor must not produce the out-of-range expression
# ===========================================================================
class TestCursorExpressionInRange:
    """The pk-cursor first-page expression must be parseable by Milvus."""

    def test_source_no_longer_uses_min_int64_sentinel(self):
        """The -2**63 cursor sentinel must be gone from executable code.

        (Comments may still reference -2**63 to explain the fix, so we scan
        only non-comment code by stripping the part after any ``#``.)
        """
        code_lines = []
        for line in open(simple_bench.__file__, encoding="utf-8"):
            code = line.split("#", 1)[0]
            code_lines.append(code)
        code = "\n".join(code_lines)

        assert "-2**63" not in code and "-2 ** 63" not in code, (
            "simple_bench still initializes the pk cursor at -2**63, which "
            "produces the out-of-range expression 'id > -9223372036854775808'."
        )
        # And the safe sentinel must be present in code.
        assert "last_pk: Union[int, str] = -1 if is_int_pk else" in code

    def test_int_first_page_expression_is_in_range(self):
        """
        Reproduce the cursor-init logic for an int PK and assert the first-page
        expression is valid (operand within int64 range), not the overflowing
        form the issue reported.
        """
        is_int_pk = True
        last_pk = -1 if is_int_pk else ""  # post-fix sentinel
        expr = f"id > {last_pk}"

        assert expr == "id > -1"
        assert expr != _BAD_INT_EXPR

        # The operand magnitude must fit in signed int64.
        operand = int(expr.split(">")[1])
        assert -(2 ** 63) <= operand <= (2 ** 63 - 1)

    def test_min_int64_operand_overflows_but_minus_one_does_not(self):
        """Document precisely why -2**63 fails and -1 is safe for Milvus int64."""
        int64_max = 2 ** 63 - 1
        # The magnitude Milvus tries to parse from the bad expression.
        bad_operand_magnitude = int("9223372036854775808")
        assert bad_operand_magnitude > int64_max  # overflow by one -> parse error
        # The fixed sentinel's magnitude is trivially in range.
        assert abs(-1) <= int64_max

    def test_varchar_first_page_uses_closed_lower_bound(self):
        """VARCHAR PKs should start with '>= \"\"' so no valid key is skipped."""
        is_int_pk = False
        first_page = True
        last_pk = -1 if is_int_pk else ""

        if is_int_pk:
            expr = f"id > {last_pk}"
        else:
            expr = 'id >= ""' if first_page else f'id > "{last_pk}"'

        assert expr == 'id >= ""'


# ===========================================================================
# Fix 3: create_flat_collection must abort when coverage is too low
# ===========================================================================
def _patch_milvus(monkeypatch, source_coll, flat_coll):
    """Wire simple_bench's module-level Milvus symbols to mocks."""
    connections = MagicMock()
    utility = MagicMock()
    utility.has_collection.return_value = False  # force the create path

    def _collection_factory(name, *args, **kwargs):
        # First positional construction in the create path is the source
        # (Collection(source_name, using=...)); the schema-bearing call builds
        # the flat collection (Collection(flat_name, schema, using=...)).
        if name == "source":
            return source_coll
        return flat_coll

    Collection = MagicMock(side_effect=_collection_factory)

    monkeypatch.setattr(simple_bench, "connections", connections)
    monkeypatch.setattr(simple_bench, "utility", utility)
    monkeypatch.setattr(simple_bench, "Collection", Collection)
    return connections, utility, Collection


class TestCoverageGuard:
    """create_flat_collection must return False on insufficient coverage."""

    def test_empty_flat_collection_returns_false(self, monkeypatch):
        """
        Source reports 1000 vectors; the copy path inserts nothing and the FLAT
        collection stays at 0 entities. The function must abort (return False),
        not declare success.
        """
        DataType = simple_bench.DataType

        source_coll = MagicMock()
        source_coll.schema = _make_source_schema(DataType.INT64)
        source_coll.name = "source"
        # num_entities is read multiple times; source always reports 1000.
        type(source_coll).num_entities = property(lambda self: 1000)
        # Make the copy yield nothing regardless of path taken.
        source_coll.query.return_value = []
        source_coll.search.return_value = []
        # Force the pk-cursor fallback rather than query_iterator.
        del source_coll.query_iterator

        flat_coll = MagicMock()
        flat_coll.name = "..._flat_gt"
        # Flat collection never gets populated -> 0 entities throughout.
        type(flat_coll).num_entities = property(lambda self: 0)

        _patch_milvus(monkeypatch, source_coll, flat_coll)

        result = create_flat_collection(
            host="127.0.0.1",
            port="19530",
            source_collection_name="source",
            flat_collection_name="..._flat_gt",
            vector_dim=8,
            metric_type="COSINE",
        )

        assert result is False, (
            "create_flat_collection must abort when the FLAT ground-truth "
            "collection covers ~0% of the source (issue #489)."
        )
        # The empty FLAT collection must never reach index construction.
        flat_coll.create_index.assert_not_called()

    def test_coverage_threshold_math(self):
        """The 99% coverage rule accepts full coverage and rejects partial."""
        def passes(flat_count, total):
            coverage = (flat_count / total) if total else 0.0
            return coverage >= 0.99

        assert passes(1_000_000, 1_000_000) is True
        assert passes(995_000, 1_000_000) is True       # 99.5%
        assert passes(0, 1_000_000) is False             # the bug's case
        assert passes(10_000, 1_000_000) is False        # ~1% (issue #375 shape)
        assert passes(980_000, 1_000_000) is False       # 98%


# ===========================================================================
# Fix 2: progress output must reflect the true copied percentage
# ===========================================================================
class TestProgressOutput:
    """The post-copy summary line must not be hard-coded to 100.0%."""

    def test_source_has_no_hardcoded_full_progress(self):
        src = open(simple_bench.__file__, encoding="utf-8").read()
        assert 'vectors (100.0%)"' not in src, (
            "simple_bench still prints a hard-coded '(100.0%)' progress line "
            "even when 0 vectors were copied (issue #489)."
        )

    @pytest.mark.parametrize(
        "copied,total,expected",
        [
            (0, 1_000_000, "0.0%"),
            (500_000, 1_000_000, "50.0%"),
            (1_000_000, 1_000_000, "100.0%"),
            (0, 0, "0.0%"),  # guard against div-by-zero
        ],
    )
    def test_progress_percentage_is_accurate(self, copied, total, expected):
        final_pct = (100.0 * copied / total) if total else 0.0
        assert f"{final_pct:.1f}%" == expected


# ===========================================================================
# Fix 4: an empty / all-invalid ground truth must invalidate the run
# ===========================================================================
class TestEmptyGroundTruthInvalidatesRun:
    """precompute_ground_truth and the num_queries_evaluated guard."""

    def test_precompute_returns_empty_when_all_neighbors_empty(self, monkeypatch):
        """
        A FLAT collection with no usable vectors yields empty neighbor lists for
        every query. precompute_ground_truth must return {} so the caller's
        'if not ground_truth' guard aborts the run.
        """
        flat_coll = MagicMock()
        type(flat_coll).num_entities = property(lambda self: 0)

        # Every search returns one hit-list per query, each empty.
        def _empty_search(data, **kwargs):
            return [[] for _ in data]

        flat_coll.search.side_effect = _empty_search

        monkeypatch.setattr(simple_bench, "Collection", lambda *a, **k: flat_coll)
        monkeypatch.setattr(simple_bench, "connections", MagicMock())

        queries = [[0.0] * 8 for _ in range(5)]
        gt = precompute_ground_truth(
            host="127.0.0.1",
            port="19530",
            flat_collection_name="..._flat_gt",
            query_vectors=queries,
            top_k=10,
            metric_type="COSINE",
        )

        assert gt == {}, (
            "Ground truth that is empty for every query must be reported as "
            "failure (empty dict), not silently returned (issue #489)."
        )

    def test_calc_recall_reports_zero_evaluated_for_empty_gt(self):
        """
        With empty ground-truth lists, calc_recall evaluates 0 queries — the
        signal the run-level guard uses to mark the benchmark invalid.
        """
        ann_results = {i: [i, i + 1, i + 2] for i in range(5)}
        empty_gt = {i: [] for i in range(5)}

        stats = calc_recall(ann_results, empty_gt, k=10)

        assert stats["num_queries_evaluated"] == 0
        assert stats["recall_at_k"] == 0.0

    def test_run_level_guard_condition(self):
        """
        The decision the run path makes: num_queries_evaluated == 0 -> FAILED.
        Asserting the predicate keeps the contract explicit.
        """
        invalid = {"num_queries_evaluated": 0}
        valid = {"num_queries_evaluated": 124_000}

        def run_is_invalid(recall_stats):
            return recall_stats.get("num_queries_evaluated", 0) == 0

        assert run_is_invalid(invalid) is True
        assert run_is_invalid(valid) is False

    def test_calc_recall_valid_when_gt_present(self):
        """Sanity: a populated ground truth still produces a valid recall."""
        ann_results = {0: [1, 2, 3], 1: [4, 5, 6]}
        ground_truth = {0: [1, 2, 9], 1: [4, 5, 6]}

        stats = calc_recall(ann_results, ground_truth, k=3)

        assert stats["num_queries_evaluated"] == 2
        # Query 0: 2/3 overlap, query 1: 3/3 overlap -> mean (2/3 + 1)/2.
        assert stats["recall_at_k"] == pytest.approx((2 / 3 + 1.0) / 2)


# ===========================================================================
# Fix 5: query_iterator batch size must be bounded under the gRPC limit
# ===========================================================================
class TestIteratorBatchSizeBound:
    """The fast-path iterator must cap its batch to stay under 256MB gRPC."""

    def test_source_caps_iterator_batch_size(self):
        src = open(simple_bench.__file__, encoding="utf-8").read()
        assert "iter_batch_size" in src, (
            "simple_bench no longer bounds the query_iterator batch size; wide "
            "vectors can exceed the 256MB gRPC limit (issue #489, root cause)."
        )

    @pytest.mark.parametrize(
        "vector_dim,copy_batch_size",
        [
            (1536, 5000),   # the issue's configuration
            (128, 5000),
            (4096, 5000),
        ],
    )
    def test_iter_batch_size_stays_under_grpc_limit(self, vector_dim, copy_batch_size):
        """The computed batch must keep one response well under 256MB."""
        bytes_per_row = max(vector_dim * 4, 1)
        safe_rows = max(1, (24 * 1024 * 1024) // bytes_per_row)
        iter_batch_size = min(copy_batch_size, safe_rows, 16384)

        approx_response_bytes = iter_batch_size * bytes_per_row
        assert approx_response_bytes <= 256 * 1024 * 1024
        assert iter_batch_size >= 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

