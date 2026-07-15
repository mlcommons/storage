"""
Regression tests for issue #625:

    Recall computed over i.i.d. uniformly random vectors is not meaningful:
    at 1536-d the query-to-corpus cosine similarity concentrates around a
    mean of ~0.75 with std ~0.008 (relative contrast ~1.1), and the gap
    between the k-th and (k+1)-th neighbor is routinely below float32
    matmul noise. Recall@k is then non-discriminative between backends /
    index configurations, and exact set-intersection recall scores
    numerically-tied neighbors as misses.

Fixes under test:

  1. ``query_mode='planted'`` (``vdbbench/benchmark/generator.py``):
     queries are perturbations of database vectors, so every query has a
     genuine planted near neighbor. In the orchestrator path the base
     vectors are reproduced bit-exactly from the dataset seed; in the
     simple_bench path they are fetched from the live collection.
  2. Tie-aware epsilon recall (``search_runner._recall_at_k``,
     ``simple_bench.calc_recall``, ``enhanced_bench.calc_recall``):
     a returned neighbor within epsilon of the k-th ground-truth score is
     credited rather than scored as a miss. Exact recall is always also
     reported; ``epsilon=0`` preserves historical behavior bit-for-bit.

These tests verify the fixes WITHOUT requiring a live Milvus or the
pymilvus package: a minimal fake ``pymilvus`` is injected into
``sys.modules`` before the modules under test are imported (same pattern
as the issue #489 / #572 tests).
"""
import os
import sys
import types
from unittest.mock import MagicMock

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
# ImportError. CI does not have pymilvus installed, so we inject a fake
# module exposing exactly the names simple_bench imports (see #489 tests).
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

from vdbbench.benchmark.generator import (  # noqa: E402
    DEFAULT_QUERY_NOISE,
    QUERY_MODES,
    VectorGenerator,
    _generate_block,
    generate_query_vectors,
    plant_queries,
)
from vdbbench.benchmark.ground_truth import GroundTruthBuilder  # noqa: E402
from vdbbench.benchmark.orchestrator import BenchmarkConfig  # noqa: E402
from vdbbench.benchmark.search_runner import _recall_at_k  # noqa: E402
from vdbbench import simple_bench  # noqa: E402
from vdbbench.simple_bench import calc_recall  # noqa: E402


DIM = 64
SEED_DATA = 42
SEED_QUERY = 99


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ===========================================================================
# 1. Planted query generation (generator.py)
# ===========================================================================


class TestPlantedQueryGeneration:
    """query_mode='planted' plants a genuine near neighbor per query."""

    def test_independent_mode_is_bit_exact_with_historical_behavior(self):
        """Default mode must reproduce the pre-#625 query stream exactly,
        so existing artifacts / GT caches remain comparable."""
        legacy_rng = np.random.RandomState(SEED_QUERY)
        legacy = legacy_rng.random((50, DIM)).astype(np.float32)
        norms = np.linalg.norm(legacy, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        legacy = legacy / norms

        current = generate_query_vectors(50, DIM, seed=SEED_QUERY)
        np.testing.assert_array_equal(current, legacy)

    def test_planted_bases_match_stored_block0_vectors_bit_exactly(self):
        """Planted queries must be derived from the *stored* vectors: a
        fresh RandomState(dataset_seed) draw equals the first rows of the
        producer's block 0, with noise=0 recovering them exactly."""
        gen = VectorGenerator(
            total_vectors=200, dimension=DIM, block_size=100, seed=SEED_DATA,
        )
        gen.start()
        block0 = gen.queue.get()
        # Drain the producer so the thread exits cleanly.
        while gen.queue.get() is not None:
            pass
        gen.join()

        nq = 30
        planted_zero_noise = generate_query_vectors(
            nq, DIM, seed=SEED_QUERY,
            query_mode="planted", dataset_seed=SEED_DATA, query_noise=0.0,
        )
        np.testing.assert_array_almost_equal(
            planted_zero_noise, block0.vectors[:nq], decimal=6,
        )

    def test_planted_queries_are_near_their_base_and_normalized(self):
        nq = 30
        base_rng = np.random.RandomState(SEED_DATA)
        base = _generate_block(nq, DIM, "uniform", base_rng)

        planted = generate_query_vectors(
            nq, DIM, seed=SEED_QUERY,
            query_mode="planted", dataset_seed=SEED_DATA,
            query_noise=DEFAULT_QUERY_NOISE,
        )

        # Unit-norm output
        np.testing.assert_allclose(
            np.linalg.norm(planted, axis=1), 1.0, atol=1e-5,
        )
        # Each query is close to its base (cosine >> random baseline) but
        # not identical to it.
        cos_to_base = np.sum(planted * base, axis=1)
        assert np.all(cos_to_base > 0.99)
        assert not np.allclose(planted, base)

    def test_planted_queries_are_deterministic(self):
        a = generate_query_vectors(
            20, DIM, seed=SEED_QUERY, query_mode="planted",
            dataset_seed=SEED_DATA,
        )
        b = generate_query_vectors(
            20, DIM, seed=SEED_QUERY, query_mode="planted",
            dataset_seed=SEED_DATA,
        )
        np.testing.assert_array_equal(a, b)

    def test_invalid_query_mode_raises(self):
        with pytest.raises(ValueError, match="query_mode"):
            generate_query_vectors(10, DIM, query_mode="clustered")
        assert "planted" in QUERY_MODES and "independent" in QUERY_MODES

    def test_planted_queries_restore_relative_contrast(self):
        """The core #625 fix: nearest-neighbor contrast must be
        overwhelming for planted queries and negligible for independent
        ones, on the same uniform corpus."""
        n_db, nq = 2000, 20
        db = _generate_block(n_db, DIM, "uniform", np.random.RandomState(SEED_DATA))

        independent = generate_query_vectors(nq, DIM, seed=SEED_QUERY)
        planted = generate_query_vectors(
            nq, DIM, seed=SEED_QUERY,
            query_mode="planted", dataset_seed=SEED_DATA,
        )

        def relative_contrast(queries):
            sims = queries @ db.T
            nn = sims.max(axis=1)
            mean = sims.mean(axis=1)
            # cosine distance contrast: d_mean / d_nn
            return float(np.mean((1 - mean) / (1 - nn)))

        rc_independent = relative_contrast(independent)
        rc_planted = relative_contrast(planted)

        assert rc_planted > 5 * rc_independent
        assert rc_planted > 10.0

    def test_planted_ground_truth_rank1_is_the_base_vector(self):
        """With small noise, GT rank-1 for query q must be db id q (the
        perturbed base), proving the neighbor really was planted."""
        n_db, nq = 1000, 25
        gen = VectorGenerator(
            total_vectors=n_db, dimension=DIM, block_size=n_db, seed=SEED_DATA,
        )
        gen.start()
        block = gen.queue.get()
        while gen.queue.get() is not None:
            pass
        gen.join()

        planted = generate_query_vectors(
            nq, DIM, seed=SEED_QUERY,
            query_mode="planted", dataset_seed=SEED_DATA, query_noise=0.02,
        )
        builder = GroundTruthBuilder(planted, k=10)
        builder.update(block)
        ids, sims = builder.build_with_similarities()

        np.testing.assert_array_equal(ids[:, 0], np.arange(nq))
        # Similarities are sorted closest-first
        assert np.all(np.diff(sims, axis=1) <= 1e-6)


# ===========================================================================
# 2. Orchestrator config validation
# ===========================================================================


class TestOrchestratorConfigValidation:
    def test_config_accepts_new_fields_from_yaml_sections(self):
        cfg = BenchmarkConfig.from_dict(
            {
                "dataset": {
                    "num_vectors": 1000,
                    "dimension": DIM,
                    "query_mode": "planted",
                    "query_noise": 0.1,
                    "recall_epsilon": 1e-4,
                },
            }
        )
        assert cfg.query_mode == "planted"
        assert cfg.query_noise == pytest.approx(0.1)
        assert cfg.recall_epsilon == pytest.approx(1e-4)

    def test_defaults_preserve_historical_behavior(self):
        cfg = BenchmarkConfig()
        assert cfg.query_mode == "independent"
        assert cfg.recall_epsilon == 0.0

    def test_planted_mode_rejects_queries_exceeding_block_size(self):
        from vdbbench.benchmark.orchestrator import BenchmarkOrchestrator

        cfg = BenchmarkConfig(
            query_mode="planted",
            num_query_vectors=200_000,
            block_size=100_000,
        )
        orch = BenchmarkOrchestrator(cfg, backend=MagicMock())
        with pytest.raises(ValueError, match="block_size"):
            orch.run()

    def test_invalid_query_mode_rejected_by_orchestrator(self):
        from vdbbench.benchmark.orchestrator import BenchmarkOrchestrator

        cfg = BenchmarkConfig(query_mode="bogus")
        orch = BenchmarkOrchestrator(cfg, backend=MagicMock())
        with pytest.raises(ValueError, match="query_mode"):
            orch.run()


# ===========================================================================
# 3. Epsilon recall -- search_runner._recall_at_k
# ===========================================================================


class TestEpsilonRecallSearchRunner:
    def _make_case(self):
        """One query, k=3. GT window of 5. ANN returns id 40 (rank 4,
        tied with rank 3 within 1e-5) instead of id 30."""
        truth_ids = np.array([[10, 20, 30, 40, 50]], dtype=np.int64)
        truth_sims = np.array(
            [[0.900, 0.850, 0.800, 0.800 - 5e-6, 0.500]], dtype=np.float32,
        )
        predicted = np.array([[10, 20, 40]], dtype=np.int64)
        return predicted, truth_ids, truth_sims

    def test_epsilon_zero_matches_historical_exact_recall(self):
        predicted, truth_ids, truth_sims = self._make_case()
        exact = _recall_at_k(predicted, truth_ids, k=3)
        with_sims = _recall_at_k(
            predicted, truth_ids, k=3,
            truth_similarities=truth_sims, epsilon=0.0,
        )
        assert exact == pytest.approx(2 / 3)
        assert with_sims == pytest.approx(exact)

    def test_epsilon_credits_float32_level_ties(self):
        predicted, truth_ids, truth_sims = self._make_case()
        recall = _recall_at_k(
            predicted, truth_ids, k=3,
            truth_similarities=truth_sims, epsilon=1e-4,
        )
        assert recall == pytest.approx(1.0)

    def test_epsilon_does_not_credit_genuine_misses(self):
        truth_ids = np.array([[10, 20, 30, 40, 50]], dtype=np.int64)
        truth_sims = np.array(
            [[0.900, 0.850, 0.800, 0.600, 0.500]], dtype=np.float32,
        )
        predicted = np.array([[10, 20, 40]], dtype=np.int64)  # 40 is far
        recall = _recall_at_k(
            predicted, truth_ids, k=3,
            truth_similarities=truth_sims, epsilon=1e-4,
        )
        assert recall == pytest.approx(2 / 3)

    def test_epsilon_recall_capped_at_one(self):
        """When several ties are all returned, credited hits are capped at
        k so recall never exceeds 1.0."""
        truth_ids = np.array([[10, 20, 30, 40]], dtype=np.int64)
        truth_sims = np.array(
            [[0.800, 0.800, 0.800, 0.800]], dtype=np.float32,
        )
        predicted = np.array([[10, 20, 30, 40]], dtype=np.int64)
        recall = _recall_at_k(
            predicted, truth_ids, k=3,
            truth_similarities=truth_sims, epsilon=1e-4,
        )
        assert recall == pytest.approx(1.0)

    def test_missing_similarities_falls_back_to_exact(self):
        predicted, truth_ids, _ = self._make_case()
        recall = _recall_at_k(
            predicted, truth_ids, k=3,
            truth_similarities=None, epsilon=1e-4,
        )
        assert recall == pytest.approx(2 / 3)


# ===========================================================================
# 4. Epsilon recall -- simple_bench.calc_recall (dict-based)
# ===========================================================================


class TestEpsilonRecallSimpleBench:
    def test_defaults_bitwise_compatible_with_pre_625_schema(self):
        """No epsilon / no scores: values and legacy keys unchanged; new
        keys report exact recall and epsilon=0 (verdict logic from #489 /
        #572 keys on num_queries_evaluated, which must be intact)."""
        ann = {0: [1, 2, 3], 1: [4, 5, 6]}
        gt = {0: [1, 2, 9], 1: [4, 5, 6]}
        stats = calc_recall(ann, gt, k=3)
        assert stats["recall_at_k"] == pytest.approx((2 / 3 + 1.0) / 2)
        assert stats["recall_at_k_exact"] == stats["recall_at_k"]
        assert stats["recall_epsilon"] == 0.0
        assert stats["num_queries_evaluated"] == 2
        assert "per_query_recall" in stats and "recall_by_query" in stats

    def test_epsilon_credits_ties_cosine_higher_is_better(self):
        ann = {0: [1, 2, 44]}
        gt = {0: [1, 2, 3, 44, 5]}
        scores = {0: [0.9, 0.85, 0.8, 0.8 - 5e-6, 0.5]}
        stats = calc_recall(
            ann, gt, k=3,
            ground_truth_scores=scores, epsilon=1e-4, higher_is_better=True,
        )
        assert stats["recall_at_k"] == pytest.approx(1.0)
        assert stats["recall_at_k_exact"] == pytest.approx(2 / 3)
        assert stats["recall_epsilon"] == pytest.approx(1e-4)

    def test_epsilon_credits_ties_l2_lower_is_better(self):
        ann = {0: [1, 2, 44]}
        gt = {0: [1, 2, 3, 44, 5]}
        # L2: distance, lower = closer; rank-4 tied with rank-3
        scores = {0: [0.10, 0.15, 0.20, 0.20 + 5e-6, 0.90]}
        stats = calc_recall(
            ann, gt, k=3,
            ground_truth_scores=scores, epsilon=1e-4, higher_is_better=False,
        )
        assert stats["recall_at_k"] == pytest.approx(1.0)
        assert stats["recall_at_k_exact"] == pytest.approx(2 / 3)

    def test_empty_gt_still_reports_zero_evaluated(self):
        """#489 guard interplay: empty GT must keep reporting
        num_queries_evaluated == 0 so the verdict aborts the run."""
        stats = calc_recall({0: [1, 2, 3]}, {0: []}, k=3, epsilon=1e-4)
        assert stats["num_queries_evaluated"] == 0
        assert stats["recall_at_k"] == 0.0
        assert stats["recall_at_k_exact"] == 0.0

    def test_enhanced_bench_calc_recall_matches(self):
        """enhanced_bench duplicates calc_recall; the epsilon semantics of
        the two implementations must agree."""
        from vdbbench import enhanced_bench

        ann = {0: [1, 2, 44]}
        gt = {0: [1, 2, 3, 44, 5]}
        scores = {0: [0.9, 0.85, 0.8, 0.8 - 5e-6, 0.5]}
        kwargs = dict(
            ground_truth_scores=scores, epsilon=1e-4, higher_is_better=True,
        )
        s1 = simple_bench.calc_recall(ann, gt, 3, **kwargs)
        s2 = enhanced_bench.calc_recall(ann, gt, 3, **kwargs)
        assert s1["recall_at_k"] == pytest.approx(s2["recall_at_k"])
        assert s1["recall_at_k_exact"] == pytest.approx(s2["recall_at_k_exact"])


# ===========================================================================
# 5. precompute_ground_truth captures scores (simple_bench, mocked Milvus)
# ===========================================================================


class TestGroundTruthScoreCapture:
    def _fake_hits(self, ids_scores):
        hits = []
        for hid, dist in ids_scores:
            h = MagicMock()
            h.id = hid
            h.distance = dist
            hits.append(h)
        return hits

    def test_scores_out_is_filled_and_aligned(self, monkeypatch):
        fake_coll = MagicMock()
        fake_coll.num_entities = 100
        fake_coll.search.return_value = [
            self._fake_hits([(7, 0.99), (3, 0.98), (5, 0.97)]),
        ]
        monkeypatch.setattr(
            simple_bench, "Collection", MagicMock(return_value=fake_coll),
        )
        monkeypatch.setattr(simple_bench, "open_connection", MagicMock())

        scores: dict = {}
        gt = simple_bench.precompute_ground_truth(
            host="h", port="p",
            flat_collection_name="flat",
            query_vectors=[[0.0] * DIM],
            top_k=3,
            scores_out=scores,
        )
        assert gt == {0: [7, 3, 5]}
        assert scores == {0: [pytest.approx(0.99), pytest.approx(0.98),
                              pytest.approx(0.97)]}

    def test_return_type_unchanged_without_scores_out(self, monkeypatch):
        """#489 test compatibility: callers that don't pass scores_out get
        the same Dict[int, List[int]] as before."""
        fake_coll = MagicMock()
        fake_coll.num_entities = 100
        fake_coll.search.return_value = [self._fake_hits([(1, 0.9)])]
        monkeypatch.setattr(
            simple_bench, "Collection", MagicMock(return_value=fake_coll),
        )
        monkeypatch.setattr(simple_bench, "open_connection", MagicMock())

        gt = simple_bench.precompute_ground_truth(
            host="h", port="p",
            flat_collection_name="flat",
            query_vectors=[[0.0] * DIM],
            top_k=1,
        )
        assert isinstance(gt, dict)
        assert gt == {0: [1]}


# ===========================================================================
# 6. GroundTruthBuilder similarities + end-to-end epsilon path
# ===========================================================================


class TestGroundTruthSimilarities:
    def test_build_with_similarities_alignment(self):
        rng = np.random.RandomState(SEED_DATA)
        db = _normalize(rng.normal(0, 1, (500, DIM)).astype(np.float32))
        queries = _normalize(rng.normal(0, 1, (10, DIM)).astype(np.float32))

        builder = GroundTruthBuilder(queries, k=5)
        from vdbbench.benchmark.generator import VectorBlock

        builder.update(VectorBlock(
            ids=np.arange(500, dtype=np.int64), vectors=db, block_index=0,
        ))
        ids, sims = builder.build_with_similarities()

        assert ids.shape == (10, 5) and sims.shape == (10, 5)
        # Verify against brute force
        ref = queries @ db.T
        for q in range(10):
            expected = np.argsort(-ref[q])[:5]
            np.testing.assert_array_equal(ids[q], expected)
            np.testing.assert_allclose(sims[q], ref[q][expected], atol=1e-6)

    def test_build_still_returns_ids_only(self):
        """Backward compatibility: build() keeps its historical contract."""
        rng = np.random.RandomState(SEED_DATA)
        db = _normalize(rng.normal(0, 1, (100, DIM)).astype(np.float32))
        queries = _normalize(rng.normal(0, 1, (4, DIM)).astype(np.float32))

        builder = GroundTruthBuilder(queries, k=3)
        from vdbbench.benchmark.generator import VectorBlock

        builder.update(VectorBlock(
            ids=np.arange(100, dtype=np.int64), vectors=db, block_index=0,
        ))
        out = builder.build()
        assert isinstance(out, np.ndarray)
        assert out.shape == (4, 3)
        assert out.dtype == np.int64
