from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .persistence import SQLiteStore


@dataclass(frozen=True)
class OperationResult:
    key: str
    reused: bool
    payload: Any


class OperationRunner:
    """Idempotent operation boundary used by review stages and resume."""

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def run(self, key: str, function: Callable[[], Any]) -> OperationResult:
        if self.store.operation_done(key):
            return OperationResult(key, True, None)
        payload = function()
        self.store.complete_operation(key, payload)
        return OperationResult(key, False, payload)
