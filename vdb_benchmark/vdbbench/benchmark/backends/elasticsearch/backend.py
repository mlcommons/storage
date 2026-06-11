"""Elasticsearch implementation of :class:`VectorDBBackend`.

This wraps the ``elasticsearch`` Python client behind the abstract backend
interface.  The implementation targets Elasticsearch 8.x dense-vector
fields with native kNN search.

Requirements::

    pip install elasticsearch
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..base import CollectionInfo, IndexProgress, VectorDBBackend

logger = logging.getLogger(__name__)

# Elasticsearch similarity names mapped from our canonical metric names.
_METRIC_TO_ES_SIMILARITY: Dict[str, str] = {
    "COSINE": "cosine",
    "L2": "l2_norm",
    "IP": "dot_product",
}


class ElasticsearchBackend(VectorDBBackend):
    """Concrete backend for Elasticsearch (8.x+ with dense vectors)."""

    def __init__(self) -> None:
        self._client = None  # type: Any   # elasticsearch.Elasticsearch
        self._index_meta: Dict[str, Dict[str, Any]] = {}  # name -> {metric, dim, …}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def connect(
        self,
        host: str = "http://localhost:9200",
        **kwargs,
    ) -> None:
        from elasticsearch import Elasticsearch

        api_key = kwargs.get("api_key")
        cloud_id = kwargs.get("cloud_id")

        if cloud_id:
            self._client = Elasticsearch(cloud_id=cloud_id, api_key=api_key)
        elif api_key:
            self._client = Elasticsearch(host, api_key=api_key)
        else:
            self._client = Elasticsearch(host)

        info = self._client.info()
        logger.info(
            "Connected to Elasticsearch %s at %s",
            info["version"]["number"],
            host,
        )

    def disconnect(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self._index_meta.clear()
        logger.info("Disconnected from Elasticsearch")

    # ------------------------------------------------------------------
    # Collection (index) management
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
        if self.collection_exists(name):
            if force:
                self.drop_collection(name)
            else:
                raise ValueError(
                    f"Index '{name}' already exists.  Use force=True to drop it."
                )

        params = index_params or {}
        similarity = _METRIC_TO_ES_SIMILARITY.get(metric_type.upper(), "cosine")

        # Build the dense_vector mapping
        vector_field: Dict[str, Any] = {
            "type": "dense_vector",
            "dims": dimension,
            "similarity": similarity,
        }

        if index_type.upper() == "HNSW":
            vector_field["index"] = True
            vector_field["index_options"] = {
                "type": "hnsw",
                "m": params.get("m", 16),
                "ef_construction": params.get("ef_construction", 100),
            }
        elif index_type.upper() == "FLAT":
            vector_field["index"] = True
            vector_field["index_options"] = {
                "type": "flat",
            }
        else:
            # Default to HNSW for unknown types
            logger.warning(
                "Unknown index type '%s'; falling back to HNSW", index_type
            )
            vector_field["index"] = True
            vector_field["index_options"] = {"type": "hnsw"}

        mappings = {
            "properties": {
                "vector": vector_field,
            }
        }
        settings = {
            "number_of_shards": num_shards,
            "number_of_replicas": 0,
        }

        self._client.indices.create(
            index=name,
            mappings=mappings,
            settings=settings,
        )
        logger.info(
            "Created index '%s' (%d-d, %s, %s, %d shards)",
            name, dimension, similarity, index_type, num_shards,
        )

        self._index_meta[name] = {
            "dimension": dimension,
            "metric_type": metric_type,
            "index_type": index_type,
            "similarity": similarity,
        }

        return CollectionInfo(
            name=name,
            dimension=dimension,
            metric_type=metric_type,
            index_type=index_type,
            row_count=0,
            extra={"index_params": params, "similarity": similarity},
        )

    def collection_exists(self, name: str) -> bool:
        return self._client.indices.exists(index=name).body

    def drop_collection(self, name: str) -> None:
        if self.collection_exists(name):
            self._client.indices.delete(index=name)
            self._index_meta.pop(name, None)
            logger.info("Deleted index: %s", name)

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------
    def insert_batch(
        self,
        name: str,
        ids: np.ndarray,
        vectors: np.ndarray,
    ) -> int:
        actions = []
        for i in range(len(ids)):
            actions.append({"index": {"_index": name, "_id": str(int(ids[i]))}})
            actions.append({"vector": vectors[i].tolist()})

        resp = self._client.bulk(operations=actions, refresh=False)
        if resp.get("errors"):
            failed = sum(
                1 for item in resp["items"]
                if item.get("index", {}).get("error")
            )
            logger.warning("Bulk insert had %s errors", f"{failed:,}")
            return len(ids) - failed
        return len(ids)

    def flush(self, name: str) -> None:
        t0 = time.time()
        self._client.indices.refresh(index=name)
        logger.info("Refresh completed in %.2f s", time.time() - t0)

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
        """k-NN search over one or more query vectors.

        A single vector uses the regular ``_search`` fast path; multiple
        vectors are sent as one ``_msearch`` (multi-search) request rather
        than a serial loop of HTTP round-trips.

        Note (VDB-4): batching here makes per-query HTTP overhead *more
        comparable* to Milvus's single-RPC batch search, but it does not make
        cross-backend QPS strictly equivalent -- ``_msearch`` and Milvus's RPC
        still differ in execution model, so cross-backend QPS should be read as
        indicative, not as an apples-to-apples measurement.
        """
        params = search_params or {}
        num_candidates = params.get("num_candidates", 100)

        if len(query_vectors) == 1:
            resp = self._client.search(
                index=name,
                knn={
                    "field": "vector",
                    "query_vector": query_vectors[0].tolist(),
                    "k": top_k,
                    "num_candidates": num_candidates,
                },
                size=top_k,
                _source=False,
            )
            return [[int(hit["_id"]) for hit in resp["hits"]["hits"]]]

        msearch_body = []
        for qvec in query_vectors:
            msearch_body.append({"index": name})
            msearch_body.append({
                "knn": {
                    "field": "vector",
                    "query_vector": qvec.tolist(),
                    "k": top_k,
                    "num_candidates": num_candidates,
                },
                "size": top_k,
                "_source": False,
            })
        response = self._client.msearch(body=msearch_body)
        return [[int(h["_id"]) for h in r["hits"]["hits"]] for r in response["responses"]]

    # ------------------------------------------------------------------
    # Status / info
    # ------------------------------------------------------------------
    def row_count(self, name: str) -> int:
        self._client.indices.refresh(index=name)
        resp = self._client.count(index=name)
        return resp["count"]

    def get_index_progress(self, name: str) -> IndexProgress:
        """Check Elasticsearch cluster health for this index.

        Elasticsearch builds HNSW segments during refresh/merge, so
        after a bulk ingest + refresh the index is queryable.  Health
        status of *yellow* or *green* means the index is ready.
        """
        health = self._client.cluster.health(
            index=name, wait_for_status="yellow", timeout="5s"
        )
        status = health["status"]
        is_ready = status in ("yellow", "green")
        return IndexProgress(is_ready=is_ready, status=status)

    # ------------------------------------------------------------------
    # Optional: load_collection (no-op for Elasticsearch)
    # ------------------------------------------------------------------
    def load_collection(self, name: str) -> None:
        """No-op -- Elasticsearch indexes are always queryable once refreshed."""
        logger.debug("load_collection is a no-op for Elasticsearch")

    # ------------------------------------------------------------------
    # Administration / introspection
    # ------------------------------------------------------------------
    def list_collections(self) -> List[str]:
        resp = self._client.cat.indices(format="json")
        return sorted(
            entry["index"]
            for entry in resp
            if not entry["index"].startswith(".")
        )

    def get_collection_info(self, name: str) -> Dict[str, Any]:
        mapping = self._client.indices.get_mapping(index=name)
        props = mapping[name]["mappings"].get("properties", {})

        # Parse vector field
        dimension = None
        metric_type = None
        index_type = None
        schema: List[Dict[str, Any]] = []
        for field_name, field_def in props.items():
            entry: Dict[str, Any] = {
                "name": field_name,
                "dtype": field_def.get("type", "unknown"),
            }
            if field_def.get("type") == "dense_vector":
                dimension = field_def.get("dims")
                entry["dim"] = dimension
                # Reverse-map similarity back to our canonical metric
                sim = field_def.get("similarity", "")
                for canonical, es_sim in _METRIC_TO_ES_SIMILARITY.items():
                    if es_sim == sim:
                        metric_type = canonical
                        break
                idx_opts = field_def.get("index_options", {})
                index_type = idx_opts.get("type", "hnsw").upper()
            schema.append(entry)

        row_count = self.row_count(name)

        return {
            "name": name,
            "row_count": row_count,
            "dimension": dimension,
            "metric_type": metric_type,
            "index_type": index_type,
            "schema": schema,
        }

    def list_indexes(self, name: str) -> List[Dict[str, Any]]:
        mapping = self._client.indices.get_mapping(index=name)
        props = mapping[name]["mappings"].get("properties", {})

        results: List[Dict[str, Any]] = []
        for field_name, field_def in props.items():
            if field_def.get("type") != "dense_vector":
                continue
            idx_opts = field_def.get("index_options", {})
            results.append({
                "index_name": field_name,
                "field_name": field_name,
                "index_type": idx_opts.get("type", "hnsw").upper(),
                "similarity": field_def.get("similarity", ""),
                "params": {
                    k: v for k, v in idx_opts.items() if k != "type"
                },
            })
        return results

    def get_collection_stats(self, name: str) -> Dict[str, Any]:
        stats = self._client.indices.stats(index=name)
        idx_stats = stats["indices"].get(name, {}).get("primaries", {})
        docs = idx_stats.get("docs", {})
        store = idx_stats.get("store", {})
        health = self._client.cluster.health(index=name)
        return {
            "name": name,
            "row_count": docs.get("count", 0),
            "deleted_docs": docs.get("deleted", 0),
            "store_size_bytes": store.get("size_in_bytes", 0),
            "index_ready": health["status"] in ("yellow", "green"),
            "index_status": health["status"],
            "indexed_rows": 0,
            "total_rows": 0,
            "pending_rows": 0,
        }
