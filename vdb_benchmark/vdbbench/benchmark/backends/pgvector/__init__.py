"""pgvector backend package.

Exposes :class:`PGVectorBackend` and :func:`backend_descriptor` for
automatic registration by the backend registry.
"""

from ..base import BackendDescriptor, IndexDescriptor, ParamDescriptor
from .backend import PGVectorBackend

__all__ = ["PGVectorBackend", "backend_descriptor"]


def backend_descriptor() -> BackendDescriptor:
    """Return the capability descriptor for the pgvector backend."""
    return BackendDescriptor(
        name="pgvector",
        display_name="pgvector (PostgreSQL)",
        description=(
            "PostgreSQL extension for vector similarity search.  Uses "
            "standard SQL with the pgvector extension for HNSW and IVFFlat "
            "indexes.  Supports COSINE, L2, and IP distance metrics.  "
            "Requires a PostgreSQL server with the vector extension "
            "installed and the psycopg2-binary + pgvector Python packages."
        ),
        backend_class=PGVectorBackend,
        supported_metrics=["COSINE", "L2", "IP"],
        supported_indexes=[
            IndexDescriptor(
                name="HNSW",
                description=(
                    "Hierarchical Navigable Small World graph index.  "
                    "Built-in to pgvector >= 0.5.0.  Good general-purpose "
                    "choice balancing recall and speed."
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
                        name="ef_search",
                        description="Search width at query time (higher = better recall).",
                        type="int",
                        default=40,
                    ),
                ],
            ),
            IndexDescriptor(
                name="IVFFLAT",
                description=(
                    "Inverted-file flat index.  Partitions vectors into "
                    "lists and searches a subset.  Lower build time than "
                    "HNSW but typically lower recall at the same speed."
                ),
                build_params=[
                    ParamDescriptor(
                        name="lists",
                        description="Number of inverted-file lists (clusters).",
                        type="int",
                        default=100,
                    ),
                ],
                search_params=[
                    ParamDescriptor(
                        name="probes",
                        description="Number of lists to probe at query time.",
                        type="int",
                        default=10,
                    ),
                ],
            ),
            IndexDescriptor(
                name="FLAT",
                description=(
                    "No index -- exact brute-force sequential scan.  "
                    "Perfect recall but O(n) per query."
                ),
                build_params=[],
                search_params=[],
            ),
        ],
        connection_params=[
            ParamDescriptor(
                name="host",
                description="PostgreSQL server hostname or IP.",
                type="str",
                default="127.0.0.1",
            ),
            ParamDescriptor(
                name="port",
                description="PostgreSQL server port.",
                type="str",
                default="5432",
            ),
            ParamDescriptor(
                name="dbname",
                description="Database name to connect to.",
                type="str",
                default="postgres",
            ),
            ParamDescriptor(
                name="user",
                description="Database user.",
                type="str",
                default="postgres",
            ),
            ParamDescriptor(
                name="password",
                description="Database password.",
                type="str",
                default="",
            ),
        ],
    )
