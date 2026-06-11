"""Abstract base class for vector database backends.

Every concrete backend (Milvus, Qdrant, Weaviate, ...) must subclass
``VectorDBBackend`` and implement the abstract methods below.  The
benchmark orchestrator only talks through this interface, so swapping
databases requires zero changes to the generation / ground-truth pipeline.

Each backend lives in its own sub-package (e.g. ``backends/milvus/``)
and exposes a :func:`backend_descriptor` function that returns a
:class:`BackendDescriptor`.  The registry discovers these packages
automatically at import time.
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Type

import numpy as np

logger = logging.getLogger(__name__)


# =====================================================================
# Capability / descriptor data model
# =====================================================================

@dataclass
class ParamDescriptor:
    """One tunable parameter for an index or a connection."""
    name: str
    description: str
    type: str = "int"          # "int", "float", "str", "bool"
    default: Any = None
    required: bool = False


@dataclass
class IndexDescriptor:
    """Everything the benchmark needs to know about one index algorithm."""
    name: str                          # e.g. "HNSW"
    description: str
    build_params: List[ParamDescriptor] = field(default_factory=list)
    search_params: List[ParamDescriptor] = field(default_factory=list)


@dataclass
class BackendDescriptor:
    """Self-description returned by every backend package.

    The registry collects these and uses them for CLI help, validation,
    and dynamic argument generation.

    Set *active* to ``False`` to keep a backend in the tree without
    exposing it to users (it will be hidden from ``--help``, CLI
    validation, and ``registry.names()``).
    """
    name: str                          # short, lower-case key ("milvus")
    display_name: str                  # human-readable ("Milvus")
    description: str                   # one-paragraph overview
    backend_class: Type["VectorDBBackend"]
    supported_metrics: List[str] = field(default_factory=list)
    supported_indexes: List[IndexDescriptor] = field(default_factory=list)
    connection_params: List[ParamDescriptor] = field(default_factory=list)
    active: bool = True

    # ------------------------------------------------------------------
    # Convenience look-ups
    # ------------------------------------------------------------------
    def index_names(self) -> List[str]:
        """Return the list of supported index algorithm names."""
        return [idx.name for idx in self.supported_indexes]

    def get_index(self, name: str) -> Optional[IndexDescriptor]:
        """Return the :class:`IndexDescriptor` for *name*, or ``None``."""
        for idx in self.supported_indexes:
            if idx.name.upper() == name.upper():
                return idx
        return None


# =====================================================================
# Collection metadata (unchanged)
# =====================================================================

@dataclass
class CollectionInfo:
    """Metadata returned after a collection is created or connected to."""
    name: str
    dimension: int
    metric_type: str
    index_type: str
    row_count: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexProgress:
    """Snapshot of index-build progress returned by backends.

    Backends fill in as much as they know:

    * **Milvus** – has ``total_rows``, ``indexed_rows``, and ``pending_rows``.
    * **pgvector** – ``CREATE INDEX`` is synchronous; simply sets ``is_ready``.
    * **Elasticsearch** – sets ``status`` (red/yellow/green) and ``is_ready``.

    The base-class ``wait_for_index`` handles all logging, adapting
    the detail level to whatever fields the backend provides.
    """
    is_ready: bool = False
    total_rows: int = 0
    indexed_rows: int = 0
    pending_rows: int = 0
    status: str = ""           # free-form backend status (e.g. "yellow")


class VectorDBBackend(abc.ABC):
    """Thin, storage-only contract that every vector DB must satisfy."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def connect(self, **kwargs) -> None:
        """Establish a connection to the database server."""

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Cleanly disconnect from the server."""

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def create_collection(
        self,
        name: str,
        dimension: int,
        metric_type: str = "COSINE",
        index_type: str = "HNSW",
        index_params: Optional[Dict[str, Any]] = None,
        num_shards: int = 1,
        force: bool = False,
    ) -> CollectionInfo:
        """Create (or re-create if *force*) a collection and its index.

        Parameters
        ----------
        name : str
            Collection / table / index name.
        dimension : int
            Dimensionality of the vectors.
        metric_type : str
            Distance metric (``COSINE``, ``L2``, ``IP``).
        index_type : str
            Index algorithm (``HNSW``, ``DISKANN``, ``FLAT``, ...).
        index_params : dict, optional
            Backend-specific index build parameters (e.g. ``M``,
            ``efConstruction`` for HNSW).
        num_shards : int
            Number of shards / partitions.
        force : bool
            If *True*, drop any existing collection with the same name first.

        Returns
        -------
        CollectionInfo
        """

    @abc.abstractmethod
    def collection_exists(self, name: str) -> bool:
        """Return *True* if the collection already exists."""

    @abc.abstractmethod
    def drop_collection(self, name: str) -> None:
        """Drop a collection if it exists."""

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def insert_batch(
        self,
        name: str,
        ids: np.ndarray,
        vectors: np.ndarray,
    ) -> int:
        """Insert a batch of vectors.

        Parameters
        ----------
        name : str
            Target collection name.
        ids : np.ndarray
            1-D array of integer primary keys (int64).
        vectors : np.ndarray
            2-D float32 array of shape ``(n, dim)``.

        Returns
        -------
        int
            Number of vectors successfully inserted.
        """

    @abc.abstractmethod
    def flush(self, name: str) -> None:
        """Flush / commit pending writes for the collection."""

    def compact(self, name: str) -> None:
        """Trigger segment compaction and wait for it to finish.

        Compaction merges many small segments into fewer large ones so
        the index builder can process them efficiently.  The default
        implementation is a no-op (not every backend needs compaction).
        """

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def search(
        self,
        name: str,
        query_vectors: np.ndarray,
        top_k: int,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> List[List[int]]:
        """Run an ANN (or exact) search.

        Parameters
        ----------
        name : str
            Collection to search.
        query_vectors : np.ndarray
            2-D float32 array of shape ``(nq, dim)``.
        top_k : int
            Number of nearest neighbors to return per query.
        search_params : dict, optional
            Backend-specific search parameters (e.g. ``ef`` for HNSW).

        Returns
        -------
        list[list[int]]
            For each query vector, a list of ``top_k`` primary-key IDs
            ordered by distance (closest first).
        """

    # ------------------------------------------------------------------
    # Status / info
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def row_count(self, name: str) -> int:
        """Return the current number of vectors in the collection."""

    @abc.abstractmethod
    def get_index_progress(self, name: str) -> IndexProgress:
        """Return a point-in-time snapshot of the index build.

        Each backend fills in whatever it can.  Milvus can report row
        counts; pgvector simply returns ``is_ready=True`` once the
        synchronous ``CREATE INDEX`` finishes; Elasticsearch checks
        cluster health status.

        The base class ``wait_for_index`` calls this in a loop and
        handles all progress logging.
        """

    # ------------------------------------------------------------------
    # Administration / introspection
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def list_collections(self) -> List[str]:
        """Return names of all collections (tables / indexes) on the server."""

    @abc.abstractmethod
    def get_collection_info(self, name: str) -> Dict[str, Any]:
        """Return detailed metadata about a single collection.

        The returned dict should include at least:

        * ``name`` (str)
        * ``row_count`` (int)
        * ``dimension`` (int or None)
        * ``metric_type`` (str or None)
        * ``index_type`` (str or None)
        * ``schema`` (list[dict] -- one entry per field/column)

        Backends may add extra keys.
        """

    @abc.abstractmethod
    def list_indexes(self, name: str) -> List[Dict[str, Any]]:
        """Return info about every index on *name*.

        Each dict should include at least ``index_name``,
        ``index_type``, and ``params``.
        """

    def drop_index(self, name: str, index_name: Optional[str] = None) -> None:
        """Drop an index from the collection.

        Parameters
        ----------
        name : str
            Collection name.
        index_name : str, optional
            Specific index to drop.  When *None* the backend drops the
            primary / only vector index.

        The default implementation raises :class:`NotImplementedError`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement drop_index"
        )

    def get_collection_stats(self, name: str) -> Dict[str, Any]:
        """Return operational statistics for a collection.

        The default implementation returns the row count and index
        progress; backends may override to add richer metrics.
        """
        prog = self.get_index_progress(name)
        return {
            "name": name,
            "row_count": self.row_count(name),
            "index_ready": prog.is_ready,
            "index_status": prog.status,
            "indexed_rows": prog.indexed_rows,
            "total_rows": prog.total_rows,
            "pending_rows": prog.pending_rows,
        }

    # ------------------------------------------------------------------
    # Unified index-wait with progress logging
    # ------------------------------------------------------------------
    _STALL_LOG_EVERY: int = 6  # stall reminder every N unchanged polls

    def wait_for_index(
        self,
        name: str,
        interval: float = 5.0,
        timeout: float = 0,
        compacted: bool = False,
    ) -> None:
        """Block until the index build finishes.

        Polls :meth:`get_index_progress` every *interval* seconds and
        emits unified progress logs.  When the backend provides row
        counts the output includes overall/recent rates and an ETA;
        otherwise a simpler status line is shown.

        Parameters
        ----------
        interval : float
            Polling interval in seconds.
        timeout : float
            Maximum seconds to wait (0 = forever).
        compacted : bool
            Hint from the orchestrator — used only in stall warnings.
        """
        start = time.time()
        prev_indexed = -1
        prev_time = start
        stall_polls = 0
        eta_deadline = float("inf")
        warned = False

        while True:
            try:
                prog = self.get_index_progress(name)
                now = time.time()
                elapsed = now - start

                # ---------- done? ----------
                if prog.is_ready:
                    if prog.total_rows:
                        logger.info(
                            "Index build complete for '%s' "
                            "(%s rows in %.1fs)",
                            name, f"{prog.total_rows:,}", elapsed,
                        )
                    else:
                        msg = f"Index ready for '{name}'"
                        if prog.status:
                            msg += f" (status: {prog.status})"
                        msg += f"  [{elapsed:.1f}s]"
                        logger.info(msg)
                    return

                # ---------- row-level progress (Milvus-style) ----------
                if prog.total_rows > 0:
                    pct = prog.indexed_rows / prog.total_rows * 100

                    if prog.indexed_rows != prev_indexed:
                        delta = prog.indexed_rows - max(prev_indexed, 0)
                        dt = now - prev_time
                        recent_rate = delta / dt if dt > 0 else 0
                        overall_rate = (
                            prog.indexed_rows / elapsed if elapsed > 0 else 0
                        )
                        remaining = prog.total_rows - prog.indexed_rows
                        eta_secs = (
                            remaining / recent_rate if recent_rate > 0 else 0
                        )
                        eta_deadline = now + eta_secs
                        eta_dt = datetime.now() + timedelta(seconds=eta_secs)
                        remaining_td = str(timedelta(seconds=int(eta_secs)))
                        logger.info(
                            "Building index: %.2f%% complete... "
                            "(%s/%s rows) | Pending rows: %s | "
                            "Overall rate: %.2f rows/sec | "
                            "Recent rate: %.2f rows/sec | "
                            "ETA: %s | Est. remaining: %s",
                            pct,
                            f"{prog.indexed_rows:,}",
                            f"{prog.total_rows:,}",
                            f"{prog.pending_rows:,}",
                            overall_rate,
                            recent_rate,
                            eta_dt.strftime("%Y-%m-%d %H:%M:%S"),
                            remaining_td,
                        )
                        stall_polls = 0
                        warned = False
                        prev_indexed = prog.indexed_rows
                        prev_time = now
                    else:
                        stall_polls += 1
                        if not warned and now > eta_deadline:
                            warned = True
                            if compacted:
                                logger.warning(
                                    "Index build has exceeded ETA by "
                                    "%.0fs (compaction was already "
                                    "performed).  This may be normal "
                                    "for large indexes -- waiting.  "
                                    "[%.0fs elapsed]",
                                    now - eta_deadline, elapsed,
                                )
                            else:
                                logger.warning(
                                    "Index build has exceeded ETA by "
                                    "%.0fs.  Set 'compact: true' in "
                                    "your config so small segments "
                                    "are merged before index build.  "
                                    "[%.0fs elapsed]",
                                    now - eta_deadline, elapsed,
                                )
                        elif stall_polls % self._STALL_LOG_EVERY == 0:
                            overall_rate = (
                                prog.indexed_rows / elapsed
                                if elapsed > 0 else 0
                            )
                            logger.info(
                                "Building index: %.2f%% complete... "
                                "(%s/%s rows) | Pending rows: %s | "
                                "Overall rate: %.2f rows/sec | "
                                "No progress for %.0fs  "
                                "[%.0fs elapsed]",
                                pct,
                                f"{prog.indexed_rows:,}",
                                f"{prog.total_rows:,}",
                                f"{prog.pending_rows:,}",
                                overall_rate,
                                stall_polls * interval,
                                elapsed,
                            )
                # ---------- status-only (ES / pgvector-style) ----------
                else:
                    status_str = prog.status or "waiting"
                    logger.info(
                        "Waiting for index on '%s' … (status: %s)  "
                        "[%.0fs elapsed]",
                        name, status_str, elapsed,
                    )
            except Exception as exc:
                logger.warning("Index progress check failed: %s", exc)

            if timeout > 0 and (time.time() - start) > timeout:
                raise TimeoutError(
                    f"Index build did not finish within {timeout}s"
                )
            time.sleep(interval)
