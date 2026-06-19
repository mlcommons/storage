"""Elasticsearch backend package.

Exposes :class:`ElasticsearchBackend` and :func:`backend_descriptor` for
automatic registration by the backend registry.

Requires the ``elasticsearch`` Python package::

    pip install elasticsearch
"""

from ..base import BackendDescriptor, IndexDescriptor, ParamDescriptor
from .backend import ElasticsearchBackend

__all__ = ["ElasticsearchBackend", "backend_descriptor"]


def backend_descriptor() -> BackendDescriptor:
    """Return the capability descriptor for the Elasticsearch backend."""
    return BackendDescriptor(
        name="elasticsearch",
        display_name="Elasticsearch",
        description=(
            "Elasticsearch with dense vector support for approximate and "
            "exact k-nearest-neighbor search.  Uses the kNN search API "
            "introduced in Elasticsearch 8.x with HNSW and brute-force "
            "(exact) retrieval.  Requires a running Elasticsearch cluster "
            "and the elasticsearch-py Python package."
        ),
        backend_class=ElasticsearchBackend,
        supported_metrics=["COSINE", "L2", "IP"],
        supported_indexes=[
            IndexDescriptor(
                name="HNSW",
                description=(
                    "Hierarchical Navigable Small World graph index.  "
                    "Default dense-vector index type in Elasticsearch 8.x."
                ),
                build_params=[
                    ParamDescriptor(
                        name="m",
                        description=(
                            "Max number of connections per node.  Higher "
                            "values improve recall at the cost of memory."
                        ),
                        type="int",
                        default=16,
                    ),
                    ParamDescriptor(
                        name="ef_construction",
                        description=(
                            "Search width during index construction.  "
                            "Higher values improve recall at the cost of "
                            "build time."
                        ),
                        type="int",
                        default=100,
                    ),
                ],
                search_params=[
                    ParamDescriptor(
                        name="num_candidates",
                        description=(
                            "Number of candidate vectors to consider per "
                            "shard during kNN search.  Higher values improve "
                            "recall at the cost of latency."
                        ),
                        type="int",
                        default=100,
                    ),
                ],
            ),
            IndexDescriptor(
                name="FLAT",
                description=(
                    "Brute-force exact search via script_score queries.  "
                    "Perfect recall but O(n) per query."
                ),
                build_params=[],
                search_params=[],
            ),
        ],
        connection_params=[
            ParamDescriptor(
                name="host",
                description="Elasticsearch server URL (e.g. http://localhost:9200).",
                type="str",
                default="http://localhost:9200",
            ),
            ParamDescriptor(
                name="api_key",
                description="API key for authentication (optional).",
                type="str",
                default=None,
            ),
            ParamDescriptor(
                name="cloud_id",
                description="Elastic Cloud deployment ID (optional, alternative to host).",
                type="str",
                default=None,
            ),
        ],
        active=True,
    )
