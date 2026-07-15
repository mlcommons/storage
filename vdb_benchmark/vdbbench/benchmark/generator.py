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


# Valid query_mode values (issue #625)
QUERY_MODES = ("independent", "planted")

# Default L2 displacement of a planted query from its base database vector.
DEFAULT_QUERY_NOISE = 0.05


def plant_queries(
    base_vectors: np.ndarray,
    seed: int,
    query_noise: float = DEFAULT_QUERY_NOISE,
) -> np.ndarray:
    """Derive query vectors from database vectors by small perturbation.

    Each query is ``normalize(base + query_noise * u)`` where *u* is a
    deterministic unit-norm Gaussian direction drawn from *seed*.  This
    "plants" a genuine near neighbor for every query, so recall@k is
    well-conditioned even when the database vectors themselves are
    i.i.d. random (issue #625): with independent queries over uniform
    1536-d vectors, the query-to-corpus similarity distribution
    concentrates (relative contrast ~1.1) and the true top-K boundary
    falls within float32 noise, making recall non-discriminative.

    Parameters
    ----------
    base_vectors : np.ndarray
        Shape ``(nq, dim)``, L2-normalized database vectors to perturb.
    seed : int
        Seed for the perturbation directions (independent of the
        dataset seed).
    query_noise : float
        Approximate L2 displacement of each query from its base vector.
        0 reproduces the base vectors exactly; ~0.05 keeps the base
        vector as the clear nearest neighbor while still exercising
        the index.

    Returns
    -------
    np.ndarray
        Shape ``(nq, dim)``, dtype float32, L2-normalized.
    """
    base = np.ascontiguousarray(base_vectors, dtype=np.float32)
    nq, dim = base.shape
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, 1, (nq, dim)).astype(np.float32)
    noise_norms = np.linalg.norm(noise, axis=1, keepdims=True)
    noise_norms[noise_norms == 0] = 1.0
    vectors = base + np.float32(query_noise) * (noise / noise_norms)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def generate_query_vectors(
    num_queries: int,
    dimension: int,
    distribution: str = "uniform",
    seed: int = 99,
    query_mode: str = "independent",
    dataset_seed: int = 42,
    query_noise: float = DEFAULT_QUERY_NOISE,
) -> np.ndarray:
    """Deterministically generate a set of query vectors.

    Two modes are supported (issue #625):

    * ``independent`` (default, preserves historical behavior) -- an
      i.i.d. draw from *seed*, fully independent of the dataset.
    * ``planted`` -- queries are perturbations of the first
      ``num_queries`` **database** vectors.  Because the producer's
      RNG stream is deterministic, regenerating the first rows with
      ``dataset_seed`` reproduces the stored vectors bit-exactly
      (requires ``num_queries <= block_size``; the orchestrator
      validates this).  Perturbation directions come from *seed* so
      the query set is still distinct from the stored data.

    Returns
    -------
    np.ndarray
        Shape ``(num_queries, dimension)``, dtype float32, L2-normalized.
    """
    if query_mode not in QUERY_MODES:
        raise ValueError(
            f"Invalid query_mode '{query_mode}'.  Must be one of {QUERY_MODES}"
        )

    if query_mode == "planted":
        # Bit-exact reproduction of the first `num_queries` database
        # vectors: _generate_block consumes the RNG stream in row-major
        # order, so a fresh RandomState(dataset_seed) draw of shape
        # (num_queries, dim) equals the first rows of block 0.
        base_rng = np.random.RandomState(dataset_seed)
        base = _generate_block(num_queries, dimension, distribution, base_rng)
        return plant_queries(base, seed=seed, query_noise=query_noise)

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
