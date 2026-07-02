"""Regression tests for index-specific Milvus search params."""

import pytest

from vdbbench.simple_bench import build_search_params


@pytest.mark.parametrize(
    ("index_type", "expected_params"),
    [
        ("HNSW", {"ef": 200}),
        ("hnsw", {"ef": 200}),
        ("DISKANN", {"search_list": 200}),
        ("diskann", {"search_list": 200}),
        ("AISAQ", {"search_list": 200}),
        ("aisaq", {"search_list": 200}),
        ("FLAT", {}),
        ("IVF_FLAT", {}),
        (None, {"ef": 200}),
        ("UNKNOWN", {"ef": 200}),
    ],
)
def test_build_search_params_uses_index_specific_effort_key(
    index_type,
    expected_params,
):
    assert build_search_params(
        index_type=index_type,
        metric_type="COSINE",
        search_ef=200,
        search_limit=10,
    ) == {"metric_type": "COSINE", "params": expected_params}


def test_build_search_params_clamps_effort_to_search_limit():
    assert build_search_params(
        index_type="DISKANN",
        metric_type="COSINE",
        search_ef=5,
        search_limit=10,
    ) == {"metric_type": "COSINE", "params": {"search_list": 10}}
