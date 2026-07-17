"""
Regression tests for issue #805: AISAQ Recall@10 = 0.00 across all runs,
with result_verdict.json still reporting "valid".

Two defects are covered:

  1. ``create_flat_collection`` / ``validate_existing_flat_collection`` reused
     an existing FLAT ground-truth collection based on entity COUNT alone. A
     stale FLAT GT left over from a regenerated source collection (same size,
     different vectors) was silently paired with the new ANN collection,
     making every ANN result miss the GT set — recall exactly 0.00 on every
     query while QPS/latency stayed healthy.
     Fix: ``verify_flat_matches_source`` samples PKs from the FLAT collection
     and compares their vectors element-wise against the source before reuse.

  2. ``emit_result_verdict`` keyed only on FLAT coverage and the number of
     evaluated queries, so an all-zero-recall run was marked "valid".
     Fix: a zero-recall gate marks the run invalid when every evaluated query
     scored recall 0.0.

These tests are dependency-free: they install a fake ``pymilvus`` before
importing ``vdbbench`` so they run in CI without a Milvus client.
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock

import pytest


def _install_fake_pymilvus():
    if "pymilvus" in sys.modules:
        return
    fake = MagicMock(name="pymilvus")
    fake.connections = MagicMock(name="connections")
    sys.modules["pymilvus"] = fake


_install_fake_pymilvus()

from vdbbench.simple_bench import (  # noqa: E402
    FlatSetupResult,
    emit_result_verdict,
    verify_flat_matches_source,
)


# ---------------------------------------------------------------------------
# Fakes for the content-verification helper
# ---------------------------------------------------------------------------


class _FakeField:
    def __init__(self, name, dtype, is_primary=False):
        self.name = name
        self.dtype = dtype
        self.is_primary = is_primary


class _FakeSchema:
    def __init__(self, fields):
        self.fields = fields


class _FakeCollection:
    """Minimal stand-in exposing schema + query() over an id->vector map."""

    def __init__(self, name, vectors_by_pk, pk_field="id", vec_field="vector"):
        from pymilvus import DataType  # the fake module

        # The fake pymilvus DataType attributes are MagicMock sentinels, which
        # is fine: _detect_schema_fields compares identity against them.
        self.name = name
        self._pk_field = pk_field
        self._vec_field = vec_field
        self._data = dict(vectors_by_pk)
        self.schema = _FakeSchema(
            [
                _FakeField(pk_field, DataType.INT64, is_primary=True),
                _FakeField(vec_field, DataType.FLOAT_VECTOR),
            ]
        )

    def load(self):
        pass

    def query(self, expr, output_fields, limit):
        # Support the two expression shapes the helper emits:
        #   "<pk> > -1"          -> first-page sample
        #   "<pk> in [1, 2, 3]"  -> targeted fetch
        if " in " in expr:
            wanted = json.loads(expr.split(" in ", 1)[1])
            pks = [pk for pk in wanted if pk in self._data]
        else:
            pks = sorted(self._data.keys())
        rows = [
            {self._pk_field: pk, self._vec_field: self._data[pk]}
            for pk in pks[:limit]
        ]
        return rows


def _vec(seed, dim=4):
    return [float(seed) + 0.1 * i for i in range(dim)]


class TestVerifyFlatMatchesSource:
    def test_matching_content_is_accepted(self):
        data = {pk: _vec(pk) for pk in range(10)}
        source = _FakeCollection("src", data)
        flat = _FakeCollection("src_flat_gt", data)

        matches, detail = verify_flat_matches_source(source, flat)

        assert matches is True
        assert "match" in detail

    def test_stale_content_with_equal_count_is_rejected(self):
        # Same PKs, same COUNT, different vectors — the issue #805 scenario:
        # the source collection was regenerated but the old FLAT GT survived.
        old_data = {pk: _vec(pk) for pk in range(10)}
        new_data = {pk: _vec(pk + 100) for pk in range(10)}
        source = _FakeCollection("src", new_data)
        flat = _FakeCollection("src_flat_gt", old_data)

        matches, detail = verify_flat_matches_source(source, flat)

        assert matches is False
        assert "mismatch" in detail

    def test_pk_missing_from_source_is_rejected(self):
        source = _FakeCollection("src", {pk: _vec(pk) for pk in range(5, 10)})
        flat = _FakeCollection("src_flat_gt", {pk: _vec(pk) for pk in range(5)})

        matches, detail = verify_flat_matches_source(source, flat)

        assert matches is False

    def test_verification_error_is_treated_as_mismatch(self):
        class _Broken(_FakeCollection):
            def query(self, *args, **kwargs):
                raise RuntimeError("boom")

        data = {pk: _vec(pk) for pk in range(3)}
        source = _FakeCollection("src", data)
        flat = _Broken("src_flat_gt", data)

        matches, detail = verify_flat_matches_source(source, flat)

        assert matches is False
        assert "verification failed" in detail

    def test_empty_flat_sample_is_rejected(self):
        source = _FakeCollection("src", {pk: _vec(pk) for pk in range(3)})
        flat = _FakeCollection("src_flat_gt", {})

        matches, _ = verify_flat_matches_source(source, flat)

        assert matches is False


# ---------------------------------------------------------------------------
# Zero-recall verdict gate
# ---------------------------------------------------------------------------


def _full_coverage_flat_result():
    return FlatSetupResult(
        ok=True,
        coverage=1.0,
        total_vectors=1000,
        copied_vectors=1000,
        reused=True,
    )


def _emit(tmp_dir, recall_stats, num_queries=1000):
    verdict = emit_result_verdict(
        tmp_dir,
        flat_result=_full_coverage_flat_result(),
        num_queries_evaluated=num_queries,
        recall_stats=recall_stats,
    )
    with open(os.path.join(tmp_dir, "result_verdict.json"), encoding="utf-8") as f:
        payload = json.load(f)
    return verdict, payload


class TestZeroRecallVerdict:
    def test_all_zero_recall_is_invalid(self):
        recall_stats = {
            "recall_at_k": 0.0,
            "mean_recall": 0.0,
            "min_recall": 0.0,
            "max_recall": 0.0,
            "num_queries_evaluated": 1000,
        }
        with tempfile.TemporaryDirectory() as tmp:
            verdict, payload = _emit(tmp, recall_stats)

        assert verdict.startswith("invalid")
        assert "recall is 0.0" in verdict
        assert payload["valid"] is False
        assert payload["recall_summary"]["max_recall"] == 0.0

    def test_nonzero_recall_stays_valid(self):
        recall_stats = {
            "recall_at_k": 0.57,
            "mean_recall": 0.57,
            "min_recall": 0.1,
            "max_recall": 1.0,
            "num_queries_evaluated": 1000,
        }
        with tempfile.TemporaryDirectory() as tmp:
            verdict, payload = _emit(tmp, recall_stats)

        assert verdict == "valid"
        assert payload["valid"] is True
        assert payload["recall_summary"]["recall_at_k"] == 0.57

    def test_low_but_nonzero_recall_stays_valid(self):
        # The gate targets the degenerate all-zero case only; a poorly tuned
        # index with tiny-but-real recall is a quality problem, not a broken
        # measurement, and remains a "valid" run.
        recall_stats = {
            "recall_at_k": 0.001,
            "mean_recall": 0.001,
            "min_recall": 0.0,
            "max_recall": 0.1,
            "num_queries_evaluated": 1000,
        }
        with tempfile.TemporaryDirectory() as tmp:
            verdict, _ = _emit(tmp, recall_stats)

        assert verdict == "valid"

    def test_missing_recall_stats_preserves_legacy_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            verdict = emit_result_verdict(
                tmp,
                flat_result=_full_coverage_flat_result(),
                num_queries_evaluated=1000,
            )
            with open(
                os.path.join(tmp, "result_verdict.json"), encoding="utf-8"
            ) as f:
                payload = json.load(f)

        assert verdict == "valid"
        assert payload["recall_summary"] is None

    def test_zero_queries_still_takes_precedence(self):
        recall_stats = {"max_recall": 0.0}
        with tempfile.TemporaryDirectory() as tmp:
            verdict = emit_result_verdict(
                tmp,
                flat_result=_full_coverage_flat_result(),
                num_queries_evaluated=0,
                recall_stats=recall_stats,
            )

        assert verdict == "invalid: 0 queries had valid ground truth"

    def test_malformed_recall_stats_do_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            verdict = emit_result_verdict(
                tmp,
                flat_result=_full_coverage_flat_result(),
                num_queries_evaluated=1000,
                recall_stats={"max_recall": "not-a-number"},
            )

        assert verdict == "valid"
