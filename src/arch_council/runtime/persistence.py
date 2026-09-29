from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from ..models import ProjectProfile


class SQLiteStore:
    """Small durable store for profiles, idempotent operations, and checkpoints."""

    def __init__(self, path: str | Path = ".arch-council/state.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Background review workers share this store; SQLite serializes the small writes.
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.executescript(
            """CREATE TABLE IF NOT EXISTS projects (project_id TEXT PRIMARY KEY, repository TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS profiles (profile_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, profile_hash TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS operations (operation_key TEXT PRIMARY KEY, status TEXT NOT NULL, payload TEXT, updated_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS checkpoints (review_id TEXT NOT NULL, stage TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(review_id, stage));
            CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY AUTOINCREMENT, operation_key TEXT, call_type TEXT, outcome TEXT, error TEXT);
            CREATE TABLE IF NOT EXISTS evidence (review_id TEXT, source_id TEXT, payload TEXT);
            CREATE TABLE IF NOT EXISTS research_cache (cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS reports (review_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, review_id TEXT NOT NULL, stage TEXT NOT NULL, event TEXT NOT NULL, payload TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);"""
        )
        self.connection.commit()

    def save_profile(self, repository: str, readme: str, *, profile_id: str = "default") -> ProjectProfile:
        profile_hash = hashlib.sha256(readme.encode()).hexdigest()
        project_id = hashlib.sha256(repository.encode()).hexdigest()[:16]
        version_id = f"{project_id}-{profile_hash[:16]}" if profile_id == "default" else profile_id
        existing = self.connection.execute(
            "SELECT profile_hash FROM profiles WHERE profile_id=?", (version_id,)
        ).fetchone()
        if existing and existing[0] != profile_hash:
            raise ValueError(f"profile_id {version_id!r} is immutable and has a different hash")
        profile = ProjectProfile(profile_id=version_id, profile_hash=profile_hash, repository=repository, readme=readme)
        self.connection.execute("INSERT OR IGNORE INTO projects VALUES (?, ?)", (project_id, repository))
        self.connection.execute(
            "INSERT OR IGNORE INTO profiles VALUES (?, ?, ?, ?)",
            (version_id, project_id, profile_hash, profile.model_dump_json()),
        )
        self.connection.commit()
        return profile

    def get_profile(self, profile_id: str) -> ProjectProfile | None:
        row = self.connection.execute("SELECT payload FROM profiles WHERE profile_id=?", (profile_id,)).fetchone()
        return ProjectProfile.model_validate_json(row[0]) if row else None

    def checkpoint(self, review_id: str, stage: str, payload: object) -> None:
        self.connection.execute("INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?)", (review_id, stage, json.dumps(payload)))
        self.event(review_id, stage, "checkpoint_completed", payload)
        self.connection.commit()

    def event(self, review_id: str, stage: str, event: str, payload: object = None) -> None:
        self.connection.execute(
            "INSERT INTO events(review_id,stage,event,payload) VALUES(?,?,?,?)",
            (review_id, stage, event, json.dumps(payload) if payload is not None else None),
        )
        self.connection.commit()

    def events(self, review_id: str) -> list[dict[str, object]]:
        rows = self.connection.execute(
            "SELECT stage,event,payload,created_at FROM events WHERE review_id=? ORDER BY id",
            (review_id,),
        ).fetchall()
        return [
            {"stage": stage, "event": event, "payload": json.loads(payload) if payload else None, "created_at": created_at}
            for stage, event, payload, created_at in rows
        ]

    def checkpoint_payload(self, review_id: str, stage: str) -> object | None:
        row = self.connection.execute("SELECT payload FROM checkpoints WHERE review_id=? AND stage=?", (review_id, stage)).fetchone()
        return json.loads(row[0]) if row else None

    def operation_done(self, key: str) -> bool:
        row = self.connection.execute("SELECT status FROM operations WHERE operation_key=?", (key,)).fetchone()
        return bool(row and row[0] == "success")

    def complete_operation(self, key: str, payload: object = None) -> None:
        self.connection.execute("INSERT OR REPLACE INTO operations(operation_key,status,payload) VALUES(?,?,?)", (key, "success", json.dumps(payload)))
        self.connection.commit()

    def save_report(self, review_id: str, payload: object) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO reports(review_id,payload) VALUES(?,?)",
            (review_id, json.dumps(payload)),
        )
        self.connection.commit()

    def report(self, review_id: str) -> object | None:
        row = self.connection.execute("SELECT payload FROM reports WHERE review_id=?", (review_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def report_history(self) -> list[dict[str, object]]:
        rows = self.connection.execute(
            "SELECT review_id,payload FROM reports ORDER BY rowid DESC"
        ).fetchall()
        return [{"review_id": review_id, **json.loads(payload)} for review_id, payload in rows]

    def save_research_cache(self, key: str, payload: object) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO research_cache(cache_key,payload) VALUES(?,?)",
            (key, json.dumps(payload)),
        )
        self.connection.commit()

    def get_research_cache(self, key: str) -> object | None:
        row = self.connection.execute(
            "SELECT payload FROM research_cache WHERE cache_key=?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def record_attempt(self, operation_key: str, call_type: str, outcome: str, error: str | None = None) -> None:
        self.connection.execute(
            "INSERT INTO attempts(operation_key,call_type,outcome,error) VALUES(?,?,?,?)",
            (operation_key, call_type, outcome, error),
        )
        self.connection.commit()
