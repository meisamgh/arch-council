from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class CallLimiter:
    def __init__(self, limit: int = 3) -> None:
        if limit < 1:
            raise ValueError("limit must be positive")
        self.semaphore = asyncio.Semaphore(limit)

    async def run(self, operation: Callable[[], Awaitable[T]]) -> T:
        async with self.semaphore:
            return await operation()
