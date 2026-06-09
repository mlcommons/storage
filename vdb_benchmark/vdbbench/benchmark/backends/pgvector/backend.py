"""pgvector (PostgreSQL) implementation of :class:`VectorDBBackend`.

This wraps ``psycopg2`` and the ``pgvector`` extension behind the abstract
backend interface so the benchmark pipeline is database-agnostic.

Requirements::

    pip install psycopg2-binary pgvector

The target PostgreSQL server must have the ``vector`` extension installed::

    CREATE EXTENSION IF NOT EXISTS vector;
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np
from psycopg2 import sql

from ..base import CollectionInfo, IndexProgress, VectorDBBackend

logger = logging.getLogger(__name__)


# Mapping from the generic metric names used by the benchmark framework
# to the pgvector operator classes required by each index type.
_METRIC_TO_HNSW_OPS: Dict[str, str] = {
    "L2": "vector_l2_ops",
    "COSINE": "vector_cosine_ops",
    "IP": "vector_ip_ops",
}

_METRIC_TO_IVFFLAT_OPS: Dict[str, str] = {
    "L2": "vector_l2_ops",
    "COSINE": "vector_cosine_ops",
    "IP": "vector_ip_ops",
}

# SQL distance operator used at query time for each metric.
_METRIC_TO_OPERATOR: Dict[str, str] = {
    "L2": "<->",
    "COSINE": "<=>",
    "IP": "<#>",
}

_VECTOR_TYPE_RE = re.compile(r"vector\((\d+)\)", re.IGNORECASE)


class PGVectorBackend(VectorDBBackend):
    """Concrete backend for PostgreSQL + pgvector."""

    def __init__(self) -> None:
        self._conn = None  # type: Any  # psycopg2 connection

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(
        self,
        host: str = "127.0.0.1",
        port: str = "5432",
        dbname: str = "postgres",
        user: str = "postgres",
        password: str = "",
        **kwargs: Any,
    ) -> None:
        """Connect to PostgreSQL and ensure pgvector is available."""
        import psycopg2
        from pgvector.psycopg2 import register_vector

        connect_timeout = kwargs.pop("connect_timeout", kwargs.pop("timeout", 10))

        conn_params: Dict[str, Any] = {
            "host": host,
            "port": port,
            "dbname": dbname,
            "user": user,
            "password": password,
        }

        if connect_timeout is not None:
            conn_params["connect_timeout"] = int(connect_timeout)

        # Support common optional psycopg2 connection parameters without
        # passing arbitrary benchmark config keys into psycopg2.connect().
        for optional_key in (
            "sslmode",
            "sslrootcert",
            "sslcert",
            "sslkey",
            "application_name",
        ):
            value = kwargs.get(optional_key)
            if value is not None:
                conn_params[optional_key] = value

        self._conn = psycopg2.connect(**conn_params)
        self._conn.autocommit = True

        register_vector(self._conn)

        # Ensure the vector extension exists.
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

        logger.info("Connected to PostgreSQL at %s:%s (db=%s)", host, port, dbname)

    def disconnect(self) -> None:
        """Disconnect from PostgreSQL."""
        if self._conn is not None and not self._conn.closed:
            self._conn.close()

        self._conn = None
        logger.info("Disconnected from PostgreSQL")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cur(self):
        """Return a new cursor, raising if not connected."""
        if self._conn is None or self._conn.closed:
            raise RuntimeError("Not connected to PostgreSQL")
        return self._conn.cursor()

    @staticmethod
    def _ident(name: str) -> sql.Identifier:
        """Return a safely quoted SQL identifier."""
        return sql.Identifier(name)

    @staticmethod
    def _index_name(table: str, suffix: str = "vec_idx") -> str:
        """Return the default vector index name for a table."""
        return f"{table}_{suffix}"

    @staticmethod
    def _vector_literal(vector: np.ndarray) -> str:
        """Convert one vector into pgvector's text input format.

        The result is always passed as a bound parameter and cast to ``vector``.
        """
        return "[" + ",".join(str(float(v)) for v in vector) + "]"

    @staticmethod
    def _metric(metric_type: str) -> str:
        """Normalize metric names to the benchmark's canonical spelling."""
        metric = (metric_type or "COSINE").upper()
        if metric not in _METRIC_TO_OPERATOR:
            logger.warning(
                "Unknown pgvector metric '%s'; defaulting to COSINE.",
                metric_type,
            )
            metric = "COSINE"
        return metric

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
        """Create a PostgreSQL table with an ``id`` primary key and vector column."""
        if self.collection_exists(name):
            if force:
                self.drop_collection(name)
            else:
                raise ValueError(f"Table '{name}' already exists. Use force=True to drop it.")

        dimension = int(dimension)
        metric = self._metric(metric_type)
        index_params = index_params or {}

        with self._cur() as cur:
            cur.execute(
                sql.SQL(
                    "CREATE TABLE {} ("
                    " id BIGINT PRIMARY KEY,"
                    " vector vector({})"
                    ")"
                ).format(
                    self._ident(name),
                    sql.SQL(str(dimension)),
                )
            )

        logger.info("Created table '%s' (%s-d)", name, f"{dimension:,}")

        # Build the index unless FLAT / NONE was requested.
        if index_type.upper() not in ("FLAT", "NONE"):
            self._create_index(
                name=name,
                dimension=dimension,
                metric_type=metric,
                index_type=index_type,
                index_params=index_params,
            )

        return CollectionInfo(
            name=name,
            dimension=dimension,
            metric_type=metric,
            index_type=index_type,
            row_count=0,
            extra={
                "index_params": index_params,
                "num_shards": num_shards,
            },
        )

    def _create_index(
        self,
        name: str,
        dimension: int,
        metric_type: str,
        index_type: str,
        index_params: Dict[str, Any],
    ) -> None:
        """Create a pgvector HNSW or IVFFLAT index."""
        del dimension  # Kept in signature for interface symmetry / future use.

        idx_name = self._index_name(name)
        upper = index_type.upper()
        metric = self._metric(metric_type)

        if upper == "HNSW":
            ops = _METRIC_TO_HNSW_OPS.get(metric, "vector_cosine_ops")
            m = int(index_params.get("M", index_params.get("m", 16)))
            ef_construction = int(
                index_params.get(
                    "efConstruction",
                    index_params.get("ef_construction", 200),
                )
            )

            stmt = sql.SQL(
                "CREATE INDEX {} ON {} "
                "USING hnsw (vector {}) "
                "WITH (m = {}, ef_construction = {})"
            ).format(
                self._ident(idx_name),
                self._ident(name),
                sql.SQL(ops),  # safe: selected from whitelist above
                sql.Literal(m),
                sql.Literal(ef_construction),
            )

        elif upper == "IVFFLAT":
            ops = _METRIC_TO_IVFFLAT_OPS.get(metric, "vector_cosine_ops")
            lists = int(index_params.get("nlist", index_params.get("lists", 100)))

            stmt = sql.SQL(
                "CREATE INDEX {} ON {} "
                "USING ivfflat (vector {}) "
                "WITH (lists = {})"
            ).format(
                self._ident(idx_name),
                self._ident(name),
                sql.SQL(ops),  # safe: selected from whitelist above
                sql.Literal(lists),
            )

        else:
            logger.warning(
                "Unknown index type '%s' for pgvector; skipping index creation.",
                index_type,
            )
            return

        logger.info("Creating index: %s", stmt.as_string(self._conn))

        with self._cur() as cur:
            cur.execute(stmt)

        logger.info("Index '%s' created (%s / %s)", idx_name, index_type, metric)

    def collection_exists(self, name: str) -> bool:
        """Return True if a public table with this name exists."""
        with self._cur() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                      AND table_type = 'BASE TABLE'
                      AND table_name = %s
                )
                """,
                (name,),
            )
            return bool(cur.fetchone()[0])

    def drop_collection(self, name: str) -> None:
        """Drop a PostgreSQL table if it exists."""
        with self._cur() as cur:
            cur.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                    self._ident(name)
                )
            )

        logger.info("Dropped table: %s", name)

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def insert_batch(
        self,
        name: str,
        ids: np.ndarray,
        vectors: np.ndarray,
    ) -> int:
        """Insert a batch of vectors into the target table."""
        import psycopg2.extras

        ids = np.asarray(ids, dtype=np.int64)
        vectors = np.asarray(vectors, dtype=np.float32)

        n = int(len(ids))
        rows = [
            (
                int(ids[i]),
                self._vector_literal(vectors[i]),
            )
            for i in range(n)
        ]

        stmt = sql.SQL(
            "INSERT INTO {} (id, vector) VALUES %s "
            "ON CONFLICT (id) DO NOTHING"
        ).format(self._ident(name))

        with self._cur() as cur:
            psycopg2.extras.execute_values(
                cur,
                stmt.as_string(self._conn),
                rows,
                template="(%s, %s::vector)",
                page_size=1000,
            )

        return n

    def flush(self, name: str) -> None:
        """No-op because this backend uses autocommit."""
        logger.info("Flush no-op with autocommit for table '%s'", name)

    def compact(self, name: str) -> None:
        """No-op for PostgreSQL / pgvector."""
        logger.info("Compaction no-op for pgvector table '%s'", name)

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
        """Run pgvector nearest-neighbor search."""
        search_params = search_params or {}

        metric = self._metric(search_params.get("metric_type", "COSINE"))
        op = _METRIC_TO_OPERATOR.get(metric, "<=>")  # safe whitelist lookup

        ef_search = search_params.get("ef_search", search_params.get("ef"))
        probes = search_params.get("probes")

        query_vectors = np.atleast_2d(np.asarray(query_vectors, dtype=np.float32))
        top_k = int(top_k)

        results: List[List[int]] = []

        # SET LOCAL requires a transaction block. The connection normally runs
        # in autocommit mode, so temporarily disable it when GUCs are requested.
        need_txn = ef_search is not None or probes is not None
        original_autocommit = self._conn.autocommit

        if need_txn:
            self._conn.autocommit = False

        try:
            with self._cur() as cur:
                if ef_search is not None:
                    cur.execute(
                        sql.SQL("SET LOCAL hnsw.ef_search = {}").format(
                            sql.Literal(int(ef_search))
                        )
                    )

                if probes is not None:
                    cur.execute(
                        sql.SQL("SET LOCAL ivfflat.probes = {}").format(
                            sql.Literal(int(probes))
                        )
                    )

                stmt = sql.SQL(
                    "SELECT id FROM {} "
                    "ORDER BY vector {} %s::vector "
                    "LIMIT %s"
                ).format(
                    self._ident(name),
                    sql.SQL(op),  # safe: selected from whitelist above
                )

                for qvec in query_vectors:
                    vec_literal = self._vector_literal(qvec)
                    cur.execute(stmt, (vec_literal, top_k))
                    results.append([int(row[0]) for row in cur.fetchall()])

            if need_txn:
                self._conn.commit()

        except Exception:
            if need_txn:
                self._conn.rollback()
            raise

        finally:
            if need_txn:
                self._conn.autocommit = original_autocommit

        return results

    # ------------------------------------------------------------------
    # Status / info
    # ------------------------------------------------------------------

    def row_count(self, name: str) -> int:
        """Return the number of rows in the table."""
        with self._cur() as cur:
            cur.execute(
                sql.SQL("SELECT COUNT(*) FROM {}").format(self._ident(name))
            )
            return int(cur.fetchone()[0])

    def get_index_progress(self, name: str) -> IndexProgress:
        """Return index readiness.

        PostgreSQL ``CREATE INDEX`` is synchronous in this backend, so once
        control returns from index creation the index is ready. For FLAT/NONE,
        no vector index is required, so the collection is also ready.
        """
        if not self.collection_exists(name):
            return IndexProgress(is_ready=False, status="table does not exist")

        indexes = self.list_indexes(name)

        if indexes:
            return IndexProgress(
                is_ready=True,
                total_rows=self.row_count(name),
                indexed_rows=self.row_count(name),
                pending_rows=0,
                status=", ".join(idx["index_name"] for idx in indexes),
            )

        return IndexProgress(
            is_ready=True,
            total_rows=self.row_count(name),
            indexed_rows=self.row_count(name),
            pending_rows=0,
            status="flat/no vector index",
        )

    # ------------------------------------------------------------------
    # Administration / introspection
    # ------------------------------------------------------------------

    def list_collections(self) -> List[str]:
        """List all public base tables."""
        with self._cur() as cur:
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """
            )
            return [row[0] for row in cur.fetchall()]

    def get_collection_info(self, name: str) -> Dict[str, Any]:
        """Return schema, row count, index type, and metric metadata."""
        schema: List[Dict[str, Any]] = []
        dimension: Optional[int] = None

        with self._cur() as cur:
            cur.execute(
                """
                SELECT
                    a.attname,
                    format_type(a.atttypid, a.atttypmod) AS formatted_type,
                    a.attnotnull,
                    EXISTS (
                        SELECT 1
                        FROM pg_index i
                        WHERE i.indrelid = a.attrelid
                          AND i.indisprimary
                          AND a.attnum = ANY(i.indkey)
                    ) AS is_primary_key
                FROM pg_attribute a
                JOIN pg_class c
                  ON c.oid = a.attrelid
                JOIN pg_namespace n
                  ON n.oid = c.relnamespace
                WHERE n.nspname = 'public'
                  AND c.relname = %s
                  AND a.attnum > 0
                  AND NOT a.attisdropped
                ORDER BY a.attnum
                """,
                (name,),
            )

            for col_name, formatted_type, not_null, is_primary_key in cur.fetchall():
                entry: Dict[str, Any] = {
                    "name": col_name,
                    "dtype": formatted_type,
                    "nullable": not bool(not_null),
                    "primary_key": bool(is_primary_key),
                }

                match = _VECTOR_TYPE_RE.search(formatted_type or "")
                if match:
                    dimension = int(match.group(1))
                    entry["dim"] = dimension

                schema.append(entry)

        indexes = self.list_indexes(name)

        index_type: Optional[str]
        metric_type: Optional[str]

        if indexes:
            index_type = indexes[0].get("index_type")
            opclass = indexes[0].get("params", {}).get("opclass", "")

            metric_type = None
            for metric, hnsw_opclass in _METRIC_TO_HNSW_OPS.items():
                ivfflat_opclass = _METRIC_TO_IVFFLAT_OPS.get(metric)
                if opclass in (hnsw_opclass, ivfflat_opclass):
                    metric_type = metric
                    break
        else:
            index_type = "FLAT"
            metric_type = None

        return {
            "name": name,
            "row_count": self.row_count(name),
            "dimension": dimension,
            "metric_type": metric_type,
            "index_type": index_type,
            "schema": schema,
        }

    def list_indexes(self, name: str) -> List[Dict[str, Any]]:
        """Return non-primary-key indexes on the table."""
        results: List[Dict[str, Any]] = []

        with self._cur() as cur:
            cur.execute(
                """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = 'public'
                  AND tablename = %s
                ORDER BY indexname
                """,
                (name,),
            )

            for idx_name, idx_def in cur.fetchall():
                # Skip the primary-key btree index.
                if idx_name.endswith("_pkey"):
                    continue

                idx_def_upper = idx_def.upper()

                if "USING HNSW" in idx_def_upper:
                    idx_type = "HNSW"
                elif "USING IVFFLAT" in idx_def_upper:
                    idx_type = "IVFFLAT"
                else:
                    idx_type = "UNKNOWN"

                opclass = None
                for candidate in set(
                    list(_METRIC_TO_HNSW_OPS.values())
                    + list(_METRIC_TO_IVFFLAT_OPS.values())
                ):
                    if candidate in idx_def:
                        opclass = candidate
                        break

                results.append(
                    {
                        "index_name": idx_name,
                        "index_type": idx_type,
                        "definition": idx_def,
                        "params": {
                            "opclass": opclass,
                        },
                    }
                )

        return results

    def drop_index(
        self,
        name: str,
        index_name: Optional[str] = None,
    ) -> None:
        """Drop a pgvector index from the table."""
        if index_name is None:
            index_name = self._index_name(name)

        with self._cur() as cur:
            cur.execute(
                sql.SQL("DROP INDEX IF EXISTS {}").format(
                    self._ident(index_name)
                )
            )

        logger.info("Dropped index '%s' from table '%s'", index_name, name)
