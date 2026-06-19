"""Producer-consumer vector-DB benchmark framework.

Key entry points:

*  :class:`BenchmarkOrchestrator` -- runs the full pipeline.
*  :class:`BenchmarkConfig` -- all tunables.
*  :mod:`backends` -- pluggable, auto-discovered database adapters.
"""

from .backends import (
    BackendDescriptor,
    BackendRegistry,
    CollectionInfo,
    IndexDescriptor,
    ParamDescriptor,
    VectorDBBackend,
    get_backend,
    registry,
)
from .generator import VectorBlock, VectorGenerator, generate_query_vectors
from .ground_truth import GroundTruthBuilder
from .orchestrator import BenchmarkConfig, BenchmarkOrchestrator
from .search_runner import SearchResult, SearchRunner

__all__ = [
    # Config & orchestration
    "BenchmarkConfig",
    "BenchmarkOrchestrator",
    # Backend framework
    "BackendDescriptor",
    "BackendRegistry",
    "CollectionInfo",
    "IndexDescriptor",
    "ParamDescriptor",
    "VectorDBBackend",
    "get_backend",
    "registry",
    # Data pipeline
    "GroundTruthBuilder",
    "VectorBlock",
    "VectorGenerator",
    "generate_query_vectors",
    # Search benchmark
    "SearchResult",
    "SearchRunner",
]
