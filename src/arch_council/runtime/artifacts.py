from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


class ArtifactStore:
    """Immutable-ish, atomic artifact storage for review audit and recovery."""

    def __init__(self, root: str | Path = ".arch-council/reviews") -> None:
        self.root = Path(root)

    def write_text(self, review_id: str, name: str, value: str) -> Path:
        return self._write(review_id, name, value.encode("utf-8"))

    def write_json(self, review_id: str, name: str, value: Any) -> Path:
        return self._write(review_id, name, json.dumps(value, indent=2, sort_keys=True).encode())

    def read_json(self, review_id: str, name: str) -> Any:
        return json.loads(self.path(review_id, name).read_text(encoding="utf-8"))

    def path(self, review_id: str, name: str) -> Path:
        return self.root / review_id / name

    def _write(self, review_id: str, name: str, data: bytes) -> Path:
        target = self.path(review_id, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, target)
        digest = hashlib.sha256(data).hexdigest()
        if not name.endswith(".sha256"):
            self.write_text(review_id, f"{name}.sha256", digest)
        return target
