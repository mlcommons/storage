from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


RANK_ENV = (
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "PMI_ID",
    "PMIX_RANK",
    "SLURM_PROCID",
)

WORLD_ENV = (
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PMIX_SIZE",
    "SLURM_NTASKS",
)

LOCAL_RANK_ENV = (
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "MPI_LOCALRANKID",
    "PMI_LOCAL_RANK",
    "SLURM_LOCALID",
)


@dataclass(frozen=True)
class MpiContext:
    rank: int
    world_size: int
    local_rank: int
    hostname: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _env_int(names: Iterable[str], default: int) -> int:
    for name in names:
        value = os.environ.get(name)
        if value is None or value == "":
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return default


def get_mpi_context() -> MpiContext:
    return MpiContext(
        rank=_env_int(RANK_ENV, 0),
        world_size=_env_int(WORLD_ENV, 1),
        local_rank=_env_int(LOCAL_RANK_ENV, 0),
        hostname=socket.gethostname(),
    )


def compute_rank_slice(total: int, rank: int, world_size: int) -> tuple[int, int]:
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    if rank < 0:
        raise ValueError(f"rank must be >= 0, got {rank}")
    if world_size <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}")
    if rank >= world_size:
        raise ValueError(f"rank {rank} must be < world_size {world_size}")

    base = total // world_size
    remainder = total % world_size
    count = base + (1 if rank < remainder else 0)
    start = rank * base + min(rank, remainder)
    return start, count


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def wait_for_file(
    path: Path,
    *,
    timeout_seconds: int,
    poll_seconds: float = 1.0,
    fail_file: Path | None = None,
) -> None:
    start = time.time()
    while not path.exists():
        if fail_file is not None and fail_file.exists():
            raise RuntimeError(f"Failure marker found: {fail_file}")
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for marker: {path}")
        time.sleep(poll_seconds)


def wait_for_rank_markers(
    base_dir: Path,
    *,
    marker_suffix: str,
    expected_ranks: int,
    timeout_seconds: int,
) -> None:
    start = time.time()
    while True:
        missing = [
            r for r in range(expected_ranks)
            if not (base_dir / f"rank_{r}.{marker_suffix}").exists()
        ]
        errors = sorted(base_dir.glob("rank_*.error.json"))
        if errors:
            raise RuntimeError(
                "One or more ranks failed: "
                + ", ".join(str(p) for p in errors)
            )
        if not missing:
            return
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"Missing rank markers: {missing}")
        time.sleep(1.0)
