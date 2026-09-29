import asyncio

from arch_council.client import ProviderRouterClient
from arch_council.runtime.artifacts import ArtifactStore
from arch_council.runtime.async_client import AsyncProviderRouterClient
from arch_council.runtime.concurrency import CallLimiter
from arch_council.runtime.jobs import ReviewJobManager
from arch_council.runtime.operations import OperationRunner
from arch_council.runtime.persistence import SQLiteStore


def test_artifact_store_writes_atomic_content_and_hash(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "reviews")
    path = store.write_json("r1", "rounds/round-001/summary.json", {"ok": True})
    assert path.exists()
    assert store.read_json("r1", "rounds/round-001/summary.json") == {"ok": True}
    assert len(store.path("r1", "rounds/round-001/summary.json.sha256").read_text()) == 64


def test_operation_runner_reuses_completed_operation(tmp_path) -> None:
    runner = OperationRunner(SQLiteStore(tmp_path / "state.sqlite3"))
    calls = []
    first = runner.run("proposal:A", lambda: calls.append(1) or {"done": True})
    second = runner.run("proposal:A", lambda: calls.append(1) or {"done": True})
    assert first.reused is False
    assert second.reused is True
    assert calls == [1]


def test_checkpoint_events_are_queryable(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "state.sqlite3")
    store.checkpoint("review-1", "proposals", {"count": 3})
    store.event("review-1", "debate-1", "started")

    events = store.events("review-1")

    assert [event["event"] for event in events] == ["checkpoint_completed", "started"]
    assert events[0]["payload"] == {"count": 3}


def test_background_job_persists_lifecycle_events(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "state.sqlite3")
    manager = ReviewJobManager(store)
    job = manager.submit("job-1", lambda: "ok")
    manager.futures["job-1"].result(timeout=2)

    assert job.status == "running"
    assert manager.status("job-1").status == "completed"
    assert [event["event"] for event in store.events("job-1")] == ["queued", "started", "completed"]
    manager.shutdown()


def test_profiles_are_immutable_hash_versions(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "state.sqlite3")
    first = store.save_profile("/repo", "README version one")
    second = store.save_profile("/repo", "README version two")

    assert first.profile_id != second.profile_id
    assert store.get_profile(first.profile_id).readme == "README version one"
    assert store.get_profile(second.profile_id).readme == "README version two"


def test_report_history_keeps_content(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "state.sqlite3")
    store.save_report("review-1", {"path": "report.md", "content": "# Decision"})

    assert store.report("review-1")["content"] == "# Decision"
    assert store.report_history()[0]["review_id"] == "review-1"


def test_attempt_records_are_persisted(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "state.sqlite3")
    store.record_attempt("review-1", "primary", "rate_limited", "HTTP 429")

    row = store.connection.execute(
        "SELECT operation_key,call_type,outcome,error FROM attempts"
    ).fetchone()
    assert row == ("review-1", "primary", "rate_limited", "HTTP 429")


def test_call_limiter_bounds_concurrency() -> None:
    active = 0
    maximum = 0

    async def work() -> None:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0)
        active -= 1

    async def run() -> None:
        limiter = CallLimiter(2)
        await asyncio.gather(*(limiter.run(work) for _ in range(6)))

    asyncio.run(run())
    assert maximum <= 2


def test_async_provider_router_bounds_calls() -> None:
    class FakeClient:
        def complete(self, **kwargs: object) -> str:
            return "ok"

    async def run() -> None:
        router = AsyncProviderRouterClient(
            ProviderRouterClient({"m": (FakeClient(), "m")}),  # type: ignore[arg-type]
            max_concurrency=2,
        )
        assert await asyncio.gather(*(router.complete(model="m", system="", user="") for _ in range(4))) == ["ok"] * 4

    asyncio.run(run())
