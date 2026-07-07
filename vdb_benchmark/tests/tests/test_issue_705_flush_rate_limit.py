"""
Regression tests for issue #705:

    VectorDB datagen fails with gRPC RateLimiter errors when --npernode > 2.

Root cause: every MPI rank in ``vdbbench/mpi_wrapper.py`` called
``flush_collection`` on the same collection. Milvus 2.4+ ships
``quotaAndLimits.flushRate.collection.max: 0.1`` (one flush per 10 s per
collection), and pymilvus's built-in rate-limit retry budget
(retry_times=75, back-off capped at 3 s => ~210 s) admits only ~21 flush
tokens. With 7 hosts x --npernode 4 = 28 concurrent flushes, tail ranks
exhaust the budget and raise MilvusException code 8; 14 ranks
(--npernode 2) squeak through, matching the reported threshold.

Fixes under test:

  1. ``load_vdb.flush_collection`` retries flushes rejected by the rate
     limiter (code 8), waits out the limiter period between attempts,
     passes a time budget via ``timeout=``, respects its deadline, and
     returns the flush wall time.
  2. ``mpi_wrapper._load_phase`` no longer flushes per rank; the collection
     is flushed exactly once by rank 0 after ``comm.gather``. Per-rank
     payloads keep ``flush_seconds`` (now 0.0) for schema stability and the
     load summary carries ``collection_flush_seconds``.

These tests run WITHOUT a live Milvus or the pymilvus package: a minimal
fake ``pymilvus`` is injected into ``sys.modules`` before importing the
module under test.
"""
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Make the package importable regardless of where pytest is invoked from.
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# load_vdb.py imports pymilvus at module load. CI may not have pymilvus
# installed, so inject a fake module exposing exactly the names it imports.
# ---------------------------------------------------------------------------
class _FakeMilvusException(Exception):
    def __init__(self, code=0, message=""):
        super().__init__(message)
        self.code = code
        self.compatible_code = code
        self.message = message


def _install_fake_pymilvus():
    if "pymilvus" in sys.modules:
        mod = sys.modules["pymilvus"]
        if not hasattr(mod, "MilvusException"):
            mod.MilvusException = _FakeMilvusException
        return

    fake = types.ModuleType("pymilvus")
    fake.connections = MagicMock()
    fake.Collection = MagicMock()
    fake.FieldSchema = MagicMock()
    fake.CollectionSchema = MagicMock()
    fake.DataType = MagicMock()
    fake.utility = MagicMock()
    fake.MilvusException = _FakeMilvusException
    sys.modules["pymilvus"] = fake


_install_fake_pymilvus()

from vdbbench import load_vdb  # noqa: E402

MilvusException = sys.modules["pymilvus"].MilvusException


def _rate_limit_exc():
    return MilvusException(
        code=8,
        message=(
            "failed to flush collection: rate limit exceeded[rate=0.1], "
            "request is rejected by grpc RateLimiter middleware, please "
            "retry later"
        ),
    )


class _FlushRecorder:
    """Fake Collection whose flush() raises RateLimit N times, then succeeds."""

    def __init__(self, rate_limit_rejections=0):
        self.remaining_rejections = rate_limit_rejections
        self.calls = []

    def flush(self, timeout=None, **kwargs):
        self.calls.append({"timeout": timeout, **kwargs})
        if self.remaining_rejections > 0:
            self.remaining_rejections -= 1
            raise _rate_limit_exc()


class TestIsRateLimitError:
    def test_detects_code_8(self):
        assert load_vdb._is_rate_limit_error(MilvusException(code=8, message="x"))

    def test_detects_message_without_code(self):
        # Real pymilvus exposes ``code`` as a read-only property, so build
        # the non-rate-limit code state via the constructor only. code=0
        # exercises the message-based fallback in _is_rate_limit_error.
        exc = MilvusException(code=0, message="rate limit exceeded[rate=0.1]")
        assert load_vdb._is_rate_limit_error(exc)

    def test_rejects_other_errors(self):
        exc = MilvusException(code=1, message="collection not found")
        assert not load_vdb._is_rate_limit_error(exc)


class TestFlushCollectionRateLimitRetry:
    def test_succeeds_first_try_and_returns_duration(self):
        coll = _FlushRecorder(rate_limit_rejections=0)
        duration = load_vdb.flush_collection(coll, max_wait_s=60)
        assert len(coll.calls) == 1
        assert isinstance(duration, float) and duration >= 0.0

    def test_passes_time_budget_to_pymilvus(self):
        """timeout= switches pymilvus's retry loop from a 75-attempt budget
        to a time budget; verify it is forwarded and positive."""
        coll = _FlushRecorder()
        load_vdb.flush_collection(coll, max_wait_s=120)
        assert coll.calls[0]["timeout"] is not None
        assert coll.calls[0]["timeout"] >= 30.0

    def test_retries_after_rate_limit_rejection(self):
        coll = _FlushRecorder(rate_limit_rejections=3)
        with patch.object(load_vdb.time, "sleep") as mock_sleep:
            duration = load_vdb.flush_collection(coll, max_wait_s=600)
        assert len(coll.calls) == 4  # 3 rejections + 1 success
        assert mock_sleep.call_count == 3
        # Each wait must cover the limiter period (one flush per 10 s).
        for c in mock_sleep.call_args_list:
            assert c.args[0] >= load_vdb.MILVUS_FLUSH_LIMITER_PERIOD_S
        assert duration >= 0.0

    def test_gives_up_after_deadline(self):
        coll = _FlushRecorder(rate_limit_rejections=10**6)
        fake_now = [1000.0]

        def fake_time():
            return fake_now[0]

        def fake_sleep(s):
            fake_now[0] += s

        with patch.object(load_vdb.time, "time", side_effect=fake_time), \
             patch.object(load_vdb.time, "sleep", side_effect=fake_sleep):
            with pytest.raises(MilvusException):
                load_vdb.flush_collection(coll, max_wait_s=45)
        # Bounded attempts: 45 s deadline / >=10 s limiter waits.
        assert len(coll.calls) <= 6

    def test_non_rate_limit_errors_propagate_immediately(self):
        class _Boom:
            def flush(self, timeout=None, **kwargs):
                raise MilvusException(code=1, message="collection not found")

        with patch.object(load_vdb.time, "sleep") as mock_sleep:
            with pytest.raises(MilvusException):
                load_vdb.flush_collection(_Boom(), max_wait_s=600)
        mock_sleep.assert_not_called()


class TestMpiWrapperSingleFlush:
    """Source-level guards: the per-rank flush must not come back."""

    @classmethod
    def setup_class(cls):
        path = os.path.join(ROOT, "vdbbench", "mpi_wrapper.py")
        with open(path) as f:
            cls.src = f.read()

    def test_no_flush_between_insert_loop_and_gather(self):
        """flush_collection must not be called in the per-rank insert
        section (before comm.gather); it runs once on rank 0 afterwards."""
        insert_section = self.src.split("comm.gather(payload, root=0)")[0]
        # The per-rank section may reference flush_collection in imports or
        # comments, but must not call it.
        per_rank_calls = [
            line
            for line in insert_section.splitlines()
            if "flush_collection(" in line
            and "import" not in line
            and not line.strip().startswith("#")
            and "from" not in line
        ]
        assert per_rank_calls == [], (
            "Per-rank flush reintroduced; this re-triggers Milvus's "
            "per-collection flush rate limiter at scale (issue #705): "
            f"{per_rank_calls}"
        )

    def test_single_flush_after_gather(self):
        post_gather = self.src.split("comm.gather(payload, root=0)", 1)[1]
        assert "flush_collection(" in post_gather
        assert "collection_flush_seconds" in post_gather

    def test_rank_payload_keeps_flush_seconds_field(self):
        """Schema stability: rank_stats entries still carry flush_seconds."""
        assert '"flush_seconds": 0.0' in self.src
