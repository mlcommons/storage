"""
Regression tests for issue #541:
`mlpstorage closed vectordb datagen file` silently ignored --collection
(and --dimension / --num-vectors) when a YAML --config was also supplied.

Root cause: the `is_default` map built in
``vdb_benchmark/vdbbench/load_vdb.py:parse_args`` and
``vdb_benchmark/vdbbench/mpi_wrapper.py:_set_load_is_default`` did not include
`collection_name`, `dimension`, or `num_vectors`. ``merge_config_with_args``
treats any key absent from `is_default` as "user did not set it on the CLI"
and lets the YAML overwrite it — so the user's explicit --collection got
clobbered by ``configs/vectordbbench/default.yaml``'s
``dataset.collection_name``.

See: https://github.com/mlcommons/storage/issues/541
"""
import argparse

import pytest

from vdbbench.config_loader import merge_config_with_args


YAML_CONFIG = {
    "dataset": {
        "collection_name": "mlps_1m_1shards_1536dim_uniform",
        "dimension": 1536,
        "num_vectors": 1_000_000,
        "num_shards": 1,
    },
}


@pytest.mark.parametrize("field, cli_value, yaml_value", [
    ("collection_name", "mlps_user_supplied_name", "mlps_1m_1shards_1536dim_uniform"),
    ("dimension", 768, 1536),
    ("num_vectors", 5_000_000, 1_000_000),
])
def test_cli_value_wins_over_yaml_for_required_args(field, cli_value, yaml_value):
    """CLI-supplied values for required args must beat YAML defaults."""
    args = argparse.Namespace(
        collection_name=None,
        dimension=None,
        num_vectors=None,
        num_shards=1,
    )
    setattr(args, field, cli_value)

    # Simulate the is_default map produced by load_vdb.parse_args() after the
    # fix in #541: the field that was passed on the CLI is_default=False.
    args.is_default = {
        "collection_name": args.collection_name is None,
        "dimension": args.dimension is None,
        "num_vectors": args.num_vectors is None,
        "num_shards": args.num_shards == 1,
    }

    merged = merge_config_with_args(YAML_CONFIG, args)
    assert getattr(merged, field) == cli_value, (
        f"YAML clobbered CLI {field}: expected {cli_value!r}, got {getattr(merged, field)!r}"
    )


@pytest.mark.parametrize("field, yaml_value", [
    ("collection_name", "mlps_1m_1shards_1536dim_uniform"),
    ("dimension", 1536),
    ("num_vectors", 1_000_000),
])
def test_yaml_fills_required_args_when_cli_omitted(field, yaml_value):
    """When the user does NOT pass a flag, YAML must still fill it in."""
    args = argparse.Namespace(
        collection_name=None,
        dimension=None,
        num_vectors=None,
        num_shards=1,
    )
    args.is_default = {
        "collection_name": True,
        "dimension": True,
        "num_vectors": True,
        "num_shards": True,
    }

    merged = merge_config_with_args(YAML_CONFIG, args)
    assert getattr(merged, field) == yaml_value


def test_load_vdb_parse_args_includes_required_fields_in_is_default(monkeypatch):
    """Direct check on load_vdb.parse_args(): the is_default map must include
    collection_name, dimension, and num_vectors (issue #541 regression)."""
    import sys

    from vdbbench import load_vdb

    argv = [
        "load-vdb",
        "--collection-name", "mlps_test",
        "--dimension", "768",
        "--num-vectors", "1000",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = load_vdb.parse_args()
    for field in ("collection_name", "dimension", "num_vectors"):
        assert field in args.is_default, (
            f"is_default missing {field!r}; merge_config_with_args will "
            f"let YAML overwrite the CLI value"
        )
        assert args.is_default[field] is False, (
            f"is_default[{field!r}] should be False after explicit CLI flag"
        )


def test_mpi_wrapper_set_load_is_default_includes_required_fields():
    """vdbbench.mpi_wrapper._set_load_is_default must also track the three
    required default-less args (issue #541 regression)."""
    from vdbbench.mpi_wrapper import _set_load_is_default

    args = argparse.Namespace(
        host="vdb-host",
        port="19530",
        collection_name="mlps_user_supplied_name",
        dimension=768,
        num_vectors=5_000_000,
        num_shards=16,
        vector_dtype="FLOAT_VECTOR",
        distribution="uniform",
        batch_size=10000,
        chunk_size=1000000,
        index_type="DISKANN",
        metric_type="COSINE",
        max_degree=16,
        search_list_size=200,
        M=16,
        ef_construction=200,
        inline_pq=16,
        monitor_interval=5,
        compact=False,
        force=False,
    )
    _set_load_is_default(args)
    for field in ("collection_name", "dimension", "num_vectors"):
        assert field in args.is_default, f"is_default missing {field!r}"
        assert args.is_default[field] is False, (
            f"is_default[{field!r}] should be False after non-None value"
        )
