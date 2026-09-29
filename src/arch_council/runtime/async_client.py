from __future__ import annotations

import asyncio
from typing import Any

from ..client import ProviderRouterClient
from .concurrency import CallLimiter


class AsyncProviderRouterClient:
    """Async facade that bounds synchronous provider calls with a semaphore."""

    def __init__(self, router: ProviderRouterClient, max_concurrency: int = 3) -> None:
        self.router = router
        self.limiter = CallLimiter(max_concurrency)

    async def complete(self, *, model: str, system: str, user: str, **kwargs: Any) -> str:
        async def call() -> str:
            return await asyncio.to_thread(
                self.router.complete, model=model, system=system, user=user, **kwargs
            )

        return await self.limiter.run(call)
