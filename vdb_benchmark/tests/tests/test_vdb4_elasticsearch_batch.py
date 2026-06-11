"""Regression test for VDB-4: Elasticsearch search() batch path.

Isolated from the other VDB tests so it does not transitively import the
Milvus backend (pymilvus): only the Elasticsearch backend is needed here.

* len == 1 -> single fast-path self._client.search() call, shape [[...]].
* len  > 1 -> one self._client.msearch() call, results unpacked from
             response["responses"] in query order.
"""
from __future__ import annotations

from unittest import mock

import numpy as np

from vdbbench.benchmark.backends.elasticsearch.backend import ElasticsearchBackend


def _hit(_id):
    return {"_id": str(_id)}


def _make_backend():
    be = ElasticsearchBackend()
    be._client = mock.MagicMock()
    return be


def test_vdb4_single_query_uses_search_fastpath():
    be = _make_backend()
    be._client.search.return_value = {"hits": {"hits": [_hit(7), _hit(3)]}}

    qv = np.zeros((1, 8), dtype=np.float32)
    out = be.search(name="idx", query_vectors=qv, top_k=2)

    assert out == [[7, 3]]
    be._client.search.assert_called_once()
    be._client.msearch.assert_not_called()


def test_vdb4_multi_query_uses_single_msearch_call():
    be = _make_backend()
    # Two query vectors -> two response entries, preserving order.
    be._client.msearch.return_value = {
        "responses": [
            {"hits": {"hits": [_hit(1), _hit(2)]}},
            {"hits": {"hits": [_hit(9)]}},
        ]
    }

    qv = np.zeros((2, 8), dtype=np.float32)
    out = be.search(name="idx", query_vectors=qv, top_k=2)

    assert out == [[1, 2], [9]]
    # Exactly one HTTP round-trip for the whole batch, not one per vector.
    be._client.msearch.assert_called_once()
    be._client.search.assert_not_called()

    # Body is header+body interleaved: 2 vectors -> 4 entries.
    body = be._client.msearch.call_args.kwargs.get("body")
    if body is None and be._client.msearch.call_args.args:
        body = be._client.msearch.call_args.args[0]
    assert len(body) == 4
    assert body[0] == {"index": "idx"}
    assert "knn" in body[1]
