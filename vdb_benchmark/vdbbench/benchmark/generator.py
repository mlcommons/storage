"""Vector generator -- the *producer* side of the pipeline.

Generates random vectors in configurable blocks and pushes them onto a
:class:`queue.Queue`.  Each block is a :class:`VectorBlock` containing:

* ``ids``      -- int64 primary keys (globally unique, monotonically increasing)
* ``vectors``  -- float32 array of shape ``(block_size, dimension)``

The generator also produces a separate set of **query vectors** that are
held aside for benchmarking and ground-truth computation.

Supported distributions: ``uniform``, ``normal``.
All vectors are L2-normalized so that COSINE distance is meaningful.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Sentinel pushed onto the queue after the last block.
_DONE = None


@dataclass
class VectorBlock:
    """A batch of vectors ready for consumption."""
    ids: np.ndarray       # shape (n,), dtype int64
    vectors: np.ndarray   # shape (n, dim), dtype float32
    block_index: int      # ordinal of this block (0-based)


def _generate_block(
    num_vectors: int,
    dimension: int,
    distribution: str,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Return a normalized float32 array of shape ``(num_vectors, dimension)``."""
    if distribution == "normal":
        vectors = rng.normal(0, 1, (num_vectors, dimension)).astype(np.float32)
    else:  # uniform (default)
        vectors = rng.random((num_vectors, dimension)).astype(np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid division by zero
    vectors /= norms
    return vectors


def generate_query_vectors(
    num_queries: int,
    dimension: int,
    distribution: str = "uniform",
    seed: int = 99,
) -> np.ndarray:
    """Deterministically generate a set of query vectors.

    Uses a *separate* seed from the database vectors so that the query
    set is independent of the dataset.

    Returns
    -------
    np.ndarray
        Shape ``(num_queries, dimension)``, dtype float32, L2-normalized.
    """
    rng = np.random.RandomState(seed)
    return _generate_block(num_queries, dimension, distribution, rng)


class VectorGenerator:
    """Producer that feeds vector blocks into a queue.

    Parameters
    ----------
    total_vectors : int
        How many database vectors to produce in total.
    dimension : int
        Dimensionality of each vector.
    block_size : int
        Vectors per block (the last block may be smaller).
    distribution : str
        ``"uniform"`` or ``"normal"``.
    seed : int
        Random seed for reproducibility.
    max_queue_depth : int
        Backpressure limit -- producer blocks when queue is this full.
    """

    def __init__(
        self,
        total_vectors: int,
        dimension: int,
        block_size: int = 100_000,
        distribution: str = "uniform",
        seed: int = 42,
        max_queue_depth: int = 4,
    ) -> None:
        self.total_vectors = total_vectors
        self.dimension = dimension
        self.block_size = block_size
        self.distribution = distribution
        self.seed = seed
        self.queue: queue.Queue[Optional[VectorBlock]] = queue.Queue(
            maxsize=max_queue_depth
        )
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[Exception] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Spawn the producer thread.  Non-blocking."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def join(self) -> None:
        """Wait for the producer to finish.  Raises if it errored."""
        if self._thread is not None:
            self._thread.join()
        if self._error is not None:
            raise self._error

    @property
    def num_blocks(self) -> int:
        return (self.total_vectors + self.block_size - 1) // self.block_size

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            rng = np.random.RandomState(self.seed)
            remaining = self.total_vectors
            block_idx = 0
            next_id = 0

            while remaining > 0:
                n = min(self.block_size, remaining)
                vectors = _generate_block(n, self.dimension, self.distribution, rng)
                ids = np.arange(next_id, next_id + n, dtype=np.int64)

                block = VectorBlock(
                    ids=ids, vectors=vectors, block_index=block_idx
                )
                self.queue.put(block)
                logger.info(
                    "Producer: block %d  (%s vectors, ids %s..%s)",
                    block_idx, f"{n:,}", f"{next_id:,}", f"{next_id + n - 1:,}",
                )

                next_id += n
                remaining -= n
                block_idx += 1

            # Sentinel signals consumers that production is done.
            self.queue.put(_DONE)
        except Exception as exc:
            logger.exception("Producer thread failed")
            self._error = exc
            self.queue.put(_DONE)
