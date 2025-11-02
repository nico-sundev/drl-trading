"""Dask configuration for parallel processing across services."""
from pydantic import Field, field_validator

from drl_trading_common.base.base_schema import BaseSchema


class DaskConfig(BaseSchema):
    """Configuration for Dask parallel processing.

    Controls how Dask executes parallel tasks to prevent CPU bottlenecks and OOM issues.
    Dask internally handles task queuing and worker management - you don't need to batch tasks manually.

    Scheduler options:
    - 'synchronous': Single-threaded, no parallelism (useful for debugging)
    - 'threads': Multi-threaded parallelism (limited by GIL, good for I/O-bound tasks)
    - 'processes': Multi-process parallelism (true parallelism, higher overhead)

    Best practices:
    - For I/O-bound tasks (DB/Feast queries): Use 'threads' scheduler
    - For CPU-bound tasks (feature computation): Use 'processes' scheduler
    - Start with num_workers = CPU cores / 2 to avoid oversubscription
    - Dask will automatically queue tasks beyond num_workers and execute as workers free up

    Attributes:
        scheduler: Execution scheduler type
        num_workers: Number of parallel workers (None = auto-detect, controls max concurrency)
        threads_per_worker: Threads per worker for 'threads'/'processes' scheduler
        memory_limit_per_worker_mb: Memory limit per worker to prevent OOM
    """
    scheduler: str = Field(
        default="threads",
        description="Dask scheduler: 'synchronous', 'threads', or 'processes'"
    )
    num_workers: int | None = Field(
        default=None,
        description="Number of workers (None = auto-detect, controls max concurrent tasks)"
    )
    threads_per_worker: int = Field(
        default=1,
        description="Threads per worker (only for 'threads' and 'processes' schedulers)"
    )

    @field_validator("scheduler")
    @classmethod
    def validate_scheduler(cls, v: str) -> str:
        """Validate scheduler type."""
        valid_schedulers = ["synchronous", "threads", "processes"]
        if v not in valid_schedulers:
            raise ValueError(f"scheduler must be one of {valid_schedulers}, got: {v}")
        return v

    @field_validator("num_workers")
    @classmethod
    def validate_num_workers(cls, v: int | None) -> int | None:
        """Validate num_workers."""
        if v is not None and v < 1:
            raise ValueError("num_workers must be >= 1")
        return v
