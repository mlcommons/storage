"""Milvus implementation of :class:`VectorDBBackend`.

This wraps ``pymilvus`` behind the abstract backend interface so the
benchmark pipeline is completely database-agnostic.  The implementation
mirrors the conventions used by the existing ``load_vdb.py`` script
(schema, index params, connection options).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from ..base import CollectionInfo, IndexProgress, VectorDBBackend

logger = logging.getLogger(__name__)


class MilvusBackend(VectorDBBackend):
    """Concrete backend for Milvus / Zilliz Cloud."""

    def __init__(self) -> None:
        self._collections: Dict[str, Collection] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def connect(
        self,
        host: str = "127.0.0.1",
        port: str = "19530",
        **kwargs,
    ) -> None:
        max_msg = kwargs.get("max_message_length", 514_983_574)
        connections.connect(
            "default",
            host=host,
            port=port,
            max_receive_message_length=max_msg,
            max_send_message_length=max_msg,
        )
        logger.info("Connected to Milvus at %s:%s", host, port)

    def disconnect(self) -> None:
        connections.disconnect("default")
        self._collections.clear()
        logger.info("Disconnected from Milvus")

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------
    def _get_collection(self, name: str) -> Collection:
        if name not in self._collections:
            self._collections[name] = Collection(name=name)
        return self._collections[name]

    @staticmethod
    def _build_index_params(
        index_type: str,
        metric_type: str,
        params: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        params = params or {}
        ip: Dict[str, Any] = {
            "index_type": index_type,
            "metric_type": metric_type,
            "params": {},
        }
        if index_type == "HNSW":
            ip["params"] = {
                "M": params.get("M", 16),
                "efConstruction": params.get("efConstruction", 200),
            }
        elif index_type == "DISKANN":
            ip["params"] = {
                "MaxDegree": params.get("MaxDegree", 64),
                "SearchListSize": params.get("SearchListSize", 200),
            }
        elif index_type == "AISAQ":
            ip["params"] = {
                "inline_pq": params.get("inline_pq", 16),
                "max_degree": params.get("max_degree", 32),
                "search_list_size": params.get("search_list_size", 100),
            }
        elif index_type == "FLAT":
            pass  # no extra params
        else:
            ip["params"] = params
        return ip

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------
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
        if utility.has_collection(name):
            if force:
                Collection(name=name).drop()
                logger.info("Dropped existing collection: %s", name)
            else:
                raise ValueError(
                    f"Collection '{name}' already exists. Use force=True to drop it."
                )

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64,
                        is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
        ]
        schema = CollectionSchema(fields, description="Benchmark Collection")
        col = Collection(name=name, schema=schema, num_shards=num_shards)
        logger.info("Created collection '%s' (%s-d, %s shards)", name, f"{dimension:,}", num_shards)

        ip = self._build_index_params(index_type, metric_type, index_params)
        col.create_index("vector", ip)
        logger.info("Index created: %s / %s", index_type, metric_type)

        self._collections[name] = col
        return CollectionInfo(
            name=name,
            dimension=dimension,
            metric_type=metric_type,
            index_type=index_type,
            row_count=0,
            extra={"index_params": ip},
        )

    def collection_exists(self, name: str) -> bool:
        return utility.has_collection(name)

    def drop_collection(self, name: str) -> None:
        if utility.has_collection(name):
            Collection(name=name).drop()
            self._collections.pop(name, None)
            logger.info("Dropped collection: %s", name)

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------
    def insert_batch(
        self,
        name: str,
        ids: np.ndarray,
        vectors: np.ndarray,
    ) -> int:
        col = self._get_collection(name)
        col.insert([ids.tolist(), vectors])
        return len(ids)

    def flush(self, name: str) -> None:
        col = self._get_collection(name)
        t0 = time.time()
        col.flush()
        logger.info("Flush completed in %.2f s", time.time() - t0)

    def compact(self, name: str) -> None:
        """Trigger Milvus segment compaction and block until done."""
        col = self._get_collection(name)
        logger.info("Triggering compaction for '%s' ...", name)
        t0 = time.time()
        col.compact()
        col.wait_for_compaction_completed()
        elapsed = time.time() - t0
        logger.info("Compaction completed in %.2f s", elapsed)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        name: str,
        query_vectors: np.ndarray,
        top_k: int,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> List[List[int]]:
        col = self._get_collection(name)
        col.load()
        raw = search_params or {}
        if "params" in raw:
            # Already in pymilvus format (has metric_type + params wrapper)
            sp = raw
        else:
            # Wrap raw keys into the structure pymilvus expects
            sp = {
                "metric_type": raw.get("metric_type", "COSINE"),
                "params": {k: v for k, v in raw.items()
                           if k != "metric_type"},
            }
        results = col.search(
            data=query_vectors.tolist(),
            anns_field="vector",
            param=sp,
            limit=top_k,
        )
        return [[hit.id for hit in hits] for hits in results]

    # ------------------------------------------------------------------
    # Status / info
    # ------------------------------------------------------------------
    def row_count(self, name: str) -> int:
        col = self._get_collection(name)
        return col.num_entities

    def flush_collection(self, name: str) -> None:
        col = self._get_collection(name)
        col.flush()

    def get_index_progress(self, name: str) -> IndexProgress:
        """Query Milvus ``index_building_progress`` and return a snapshot."""
        progress = utility.index_building_progress(name)
        total = progress.get("total_rows", 0)
        indexed = progress.get("indexed_rows", 0)
        pending = progress.get("pending_index_rows", 0)
        is_ready = total > 0 and indexed >= total and pending == 0
        return IndexProgress(
            is_ready=is_ready,
            total_rows=total,
            indexed_rows=indexed,
            pending_rows=pending,
        )

    # ------------------------------------------------------------------
    # Administration / introspection
    # ------------------------------------------------------------------
    def list_collections(self) -> List[str]:
        return utility.list_collections()

    def get_collection_info(self, name: str) -> Dict[str, Any]:
        # NOTE (VDB-3): introspection must not mutate storage state. row_count
        # here reads col.num_entities directly; call flush_collection(name)
        # explicitly beforehand if you need the count to reflect unflushed
        # in-memory segments.
        col = self._get_collection(name)

        # Extract schema fields
        schema = []
        dimension = None
        for field in col.schema.fields:
            entry: Dict[str, Any] = {
                "name": field.name,
                "dtype": field.dtype.name if hasattr(field.dtype, "name") else str(field.dtype),
                "is_primary": field.is_primary,
            }
            if field.params.get("dim"):
                entry["dim"] = field.params["dim"]
                dimension = field.params["dim"]
            schema.append(entry)

        # Extract index info
        index_type = None
        metric_type = None
        if col.indexes:
            idx = col.indexes[0]
            index_type = idx.params.get("index_type")
            metric_type = idx.params.get("metric_type")

        return {
            "name": name,
            "row_count": col.num_entities,
            "dimension": dimension,
            "metric_type": metric_type,
            "index_type": index_type,
            "schema": schema,
            "num_partitions": len(col.partitions),
            "partitions": [p.name for p in col.partitions],
        }

    def list_indexes(self, name: str) -> List[Dict[str, Any]]:
        col = self._get_collection(name)
        results: List[Dict[str, Any]] = []
        for idx in col.indexes:
            results.append({
                "index_name": idx.field_name,
                "field_name": idx.field_name,
                "index_type": idx.params.get("index_type", "UNKNOWN"),
                "metric_type": idx.params.get("metric_type", "UNKNOWN"),
                "params": idx.params.get("params", {}),
            })
        return results

    def drop_index(self, name: str, index_name: Optional[str] = None) -> None:
        col = self._get_collection(name)
        field = index_name or "vector"
        col.drop_index(field_name=field)
        logger.info("Dropped index on field '%s' from '%s'", field, name)

    def get_collection_stats(self, name: str) -> Dict[str, Any]:
        # NOTE (VDB-3): monitoring/stats must not mutate storage state. See
        # flush_collection(name) for intentional pre-benchmark flushing.
        col = self._get_collection(name)
        prog = self.get_index_progress(name)
        stats: Dict[str, Any] = {
            "name": name,
            "row_count": col.num_entities,
            "index_ready": prog.is_ready,
            "index_status": prog.status,
            "indexed_rows": prog.indexed_rows,
            "total_rows": prog.total_rows,
            "pending_rows": prog.pending_rows,
            "num_partitions": len(col.partitions),
        }
        return stats
