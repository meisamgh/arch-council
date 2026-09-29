import json

import pytest

from arch_council.readme_summary import ReadmeSummarizer
from arch_council.runtime.persistence import SQLiteStore


class Client:
    def __init__(self):
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps({
            "objectives": ["Improve customer decisions"],
            "design": ["Proposed briefing; not verified as implemented"],
            "constraints": ["Preserve uncertainty"],
            "assumptions": ["Customer access is assumed"],
            "evidence": [], "open_questions": ["Is the data sufficient?"],
            "source_quotes": ["Customer access is assumed"],
        })


def service(tmp_path, client):
    return ReadmeSummarizer(client, SQLiteStore(tmp_path / "db"), "qwen", 1200)


def test_short_auto_and_never_make_no_calls(tmp_path):
    client = Client()
    s = service(tmp_path, client)
    assert s.prepare("short draft", "auto") == "short draft"
    assert s.prepare("x" * 12000, "never") == "x" * 12000
    assert client.calls == []


def test_always_cache_and_source_model_invalidation(tmp_path):
    client = Client()
    s = service(tmp_path, client)
    source = "Customer access is assumed"
    first = s.prepare(source, "always")
    assert s.prepare(source, "always") == first
    assert len(client.calls) == 1
    s.prepare(source + " changed", "always")
    assert len(client.calls) == 2
    s.model = "other"
    s.prepare(source, "always")
    assert len(client.calls) == 3


def test_large_source_is_fully_processed_with_bounded_prompts(tmp_path):
    client = Client()
    source = "Customer access is assumed\n" + "Details. " * 3000 + "END_MARKER"
    result = service(tmp_path, client).prepare(source, "auto")
    assert len(result) < len(source) / 3
    assert "END_MARKER" in client.calls[-1]["user"]
    assert max(len(c["system"]) + len(c["user"]) for c in client.calls) < 8000
    assert "not verified" in result and "Is the data sufficient?" in result


def test_invalid_summary_is_repaired_and_raw_retained(tmp_path):
    class BrokenOnce(Client):
        def complete(self, **kwargs):
            valid = super().complete(**kwargs)
            return "not json" if len(self.calls) == 1 else valid
    client = BrokenOnce()
    s = service(tmp_path, client)
    s.prepare("Customer access is assumed", "always")
    assert len(client.calls) == 2
    assert "Validation" in client.calls[1]["user"]
    assert s.store.connection.execute("SELECT count(*) FROM checkpoints").fetchone()[0] >= 2


def test_fabricated_quote_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="summary"):
        service(tmp_path, Client()).prepare("No matching quote here", "always")


def test_council_summary_budget_resume_and_full_short_context(tmp_path):
    from test_council import FakeClient

    from arch_council.council import Council

    class SummaryCouncilClient(FakeClient):
        def complete(self, **kwargs):
            normal = super().complete(**kwargs)
            return Client().complete(**kwargs) if kwargs["stage"].startswith("readme-summary") else normal

    store = SQLiteStore(tmp_path / "db")
    source = "Customer access is assumed\n" + "details " * 500
    options = {"store": store, "review_id": "review", "topic": "Review", "context": source,
               "models": {"A": "a", "B": "b"}, "readme_summary": "always"}
    first = SummaryCouncilClient()
    result = Council(client=first, max_calls=1, **options).run()
    assert result["stop_reason"] == "budget_exhausted"
    assert result["budget"]["calls"] == 1
    second = SummaryCouncilClient()
    result = Council(client=second, max_calls=12, **options).run()
    assert result["stop_reason"] == "converged"
    assert sum(c["stage"] == "readme-summary-1" for c in second.calls) == 0
    with pytest.raises(ValueError, match="configuration"):
        Council(client=second, **{**options, "context": source + "changed"}).run()
    plain = Council(client=FakeClient(), store=store, review_id="plain", topic="Review",
                    context="a" * 4000 + "END_MARKER", models={"A": "a", "B": "b"})
    assert "END_MARKER" in plain.prompt("A", blind=True)


def test_cli_defaults_and_case_insensitive_readme(tmp_path):
    from arch_council.cli import _build_parser
    from arch_council.context import build_readme_context

    args = _build_parser().parse_args(["review", "--question", "Review"])
    assert args.readme_summary == "auto"
    assert args.summary_max_tokens == 1200
    (tmp_path / "README.MD").write_text("Complete draft", encoding="utf-8")
    assert "Complete draft" in build_readme_context(tmp_path).text


def test_council_cli_does_not_summarize_a_truncated_source(tmp_path, monkeypatch):
    from arch_council.cli import _build_parser, _run_review
    from arch_council.config import Settings

    monkeypatch.setattr("arch_council.council_cli.Settings.from_env",
                        lambda: Settings(api_key="fixture"))
    (tmp_path / "README.md").write_text("a" * 3000, encoding="utf-8")
    args = _build_parser().parse_args([
        "review", "--question", "Review", "--repo", str(tmp_path), "--readme-only",
        "--max-context-chars", "1000", "--state-db", str(tmp_path / "db"),
    ])
    with pytest.raises(ValueError, match="whole source"):
        _run_review(args)


def test_cli_summary_end_to_end_and_resume(tmp_path, monkeypatch):
    from test_council import FakeClient

    from arch_council.cli import _build_parser, _run_resume, _run_review
    from arch_council.config import Settings

    class SummaryClient(FakeClient):
        def complete(self, **kwargs):
            normal = super().complete(**kwargs)
            if kwargs["stage"].startswith("readme-summary"):
                return Client().complete(**kwargs)
            return normal

    client = SummaryClient()
    monkeypatch.setattr("arch_council.council_cli.Settings.from_env",
                        lambda: Settings(api_key="fixture"))
    monkeypatch.setattr("arch_council.council_cli.CouncilRouter", lambda *args: client)
    (tmp_path / "README.md").write_text(
        "Customer access is assumed\n" + "details " * 1300, encoding="utf-8")
    parser = _build_parser()
    db = str(tmp_path / "db")
    args = parser.parse_args([
        "review", "--question", "Review", "--repo", str(tmp_path), "--readme-only",
        "--state-db", db, "--output-dir", str(tmp_path / "reports"), "--review-id", "e2e",
    ])
    assert _run_review(args) == 0
    count = len(client.calls)
    assert any(c["stage"].startswith("readme-summary") for c in client.calls)
    proposal = next(c for c in client.calls if c["stage"] == "proposal-A")
    assert "README brief" in proposal["user"]
    assert len(proposal["user"]) < 5000
    assert _run_resume(parser.parse_args([
        "resume", "--review-id", "e2e", "--state-db", db,
    ])) == 0
    assert len(client.calls) == count
