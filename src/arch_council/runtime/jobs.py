from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from .persistence import SQLiteStore


@dataclass(frozen=True)
class Job:
    job_id: str
    status: str


class ReviewJobManager:
    """Small process-local background runner with durable status events."""

    def __init__(self, store: SQLiteStore, workers: int = 1) -> None:
        self.store = store
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="arc-review")
        self.futures: dict[str, Future[object]] = {}

    def submit(self, job_id: str, function: Callable[[], object]) -> Job:
        if job_id in self.futures:
            return self.status(job_id)
        self.store.event(job_id, "job", "queued")

        def run() -> object:
            self.store.event(job_id, "job", "started")
            try:
                result = function()
            except Exception as exc:
                self.store.event(job_id, "job", "failed", {"error": str(exc)})
                raise
            self.store.event(job_id, "job", "completed")
            return result

        self.futures[job_id] = self.executor.submit(run)
        return Job(job_id, "running")

    def status(self, job_id: str) -> Job:
        future = self.futures.get(job_id)
        if future is None:
            return Job(job_id, "unknown")
        if future.cancelled():
            return Job(job_id, "cancelled")
        if not future.done():
            return Job(job_id, "running")
        return Job(job_id, "failed" if future.exception() else "completed")

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)
