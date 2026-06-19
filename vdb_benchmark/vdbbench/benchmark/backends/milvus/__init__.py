"""Milvus backend package.

Exposes :class:`MilvusBackend` and :func:`backend_descriptor` for
automatic registration by the backend registry.
"""

from ..base import BackendDescriptor, IndexDescriptor, ParamDescriptor
from .backend import MilvusBackend

__all__ = ["MilvusBackend", "backend_descriptor"]


def backend_descriptor() -> BackendDescriptor:
    """Return the capability descriptor for the Milvus backend."""
    return BackendDescriptor(
        name="milvus",
        display_name="Milvus",
        description=(
            "Open-source vector database built for scalable similarity "
            "search.  Supports HNSW, DiskANN, AISAQ, and FLAT index types "
            "with COSINE, L2, and IP distance metrics.  Requires a running "
            "Milvus server (standalone or cluster) and the pymilvus Python "
            "package."
        ),
        backend_class=MilvusBackend,
        supported_metrics=["COSINE", "L2", "IP"],
        supported_indexes=[
            IndexDescriptor(
                name="HNSW",
                description=(
                    "Hierarchical Navigable Small World graph index.  "
                    "Good general-purpose choice balancing recall and speed."
                ),
                build_params=[
                    ParamDescriptor(
                        name="M",
                        description="Max number of connections per node.",
                        type="int",
                        default=16,
                    ),
                    ParamDescriptor(
                        name="efConstruction",
                        description="Search width during index construction.",
                        type="int",
                        default=200,
                    ),
                ],
                search_params=[
                    ParamDescriptor(
                        name="ef",
                        description="Search width at query time (higher = better recall).",
                        type="int",
                        default=128,
                    ),
                ],
            ),
            IndexDescriptor(
                name="DISKANN",
                description=(
                    "Microsoft DiskANN -- SSD-friendly graph index for "
                    "large-scale datasets that exceed RAM."
                ),
                build_params=[
                    ParamDescriptor(
                        name="MaxDegree",
                        description="Maximum out-degree of each graph node.",
                        type="int",
                        default=64,
                    ),
                    ParamDescriptor(
                        name="SearchListSize",
                        description="Candidate-list size during index build.",
                        type="int",
                        default=200,
                    ),
                ],
                search_params=[
                    ParamDescriptor(
                        name="search_list",
                        description="Candidate-list size at query time.",
                        type="int",
                        default=200,
                    ),
                ],
            ),
            IndexDescriptor(
                name="AISAQ",
                description=(
                    "Approximate Inference with Scalar and Additive "
                    "Quantization -- a compressed index format."
                ),
                build_params=[
                    ParamDescriptor(
                        name="inline_pq",
                        description="Product-quantization sub-vector count.",
                        type="int",
                        default=16,
                    ),
                    ParamDescriptor(
                        name="max_degree",
                        description="Maximum out-degree of each graph node.",
                        type="int",
                        default=32,
                    ),
                    ParamDescriptor(
                        name="search_list_size",
                        description="Candidate-list size during build.",
                        type="int",
                        default=100,
                    ),
                ],
                search_params=[],
            ),
            IndexDescriptor(
                name="FLAT",
                description=(
                    "Brute-force exact search (no indexing).  "
                    "Perfect recall but O(n) per query."
                ),
                build_params=[],
                search_params=[],
            ),
        ],
        connection_params=[
            ParamDescriptor(
                name="host",
                description="Milvus server hostname or IP.",
                type="str",
                default="127.0.0.1",
            ),
            ParamDescriptor(
                name="port",
                description="Milvus gRPC port.",
                type="str",
                default="19530",
            ),
            ParamDescriptor(
                name="max_message_length",
                description="Max gRPC message size in bytes.",
                type="int",
                default=514_983_574,
            ),
        ],
    )
