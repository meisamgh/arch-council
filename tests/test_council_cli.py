from test_council import FakeClient

from arch_council.cli import _build_parser, _run_resume, _run_review
from arch_council.config import Settings
from arch_council.runtime.persistence import SQLiteStore


def test_topic_cli_reports_and_actual_resume_without_new_calls(tmp_path, monkeypatch):
    monkeypatch.setattr("arch_council.council_cli.Settings.from_env", lambda: Settings(api_key="fixture"))
    client = FakeClient()
    monkeypatch.setattr("arch_council.council_cli.CouncilRouter", lambda *a: client)
    parser = _build_parser()
    db = str(tmp_path / "state.db")
    args = parser.parse_args(["review", "--topic", "Improve feature discovery", "--review-id", "offline",
                              "--state-db", db, "--output-dir", str(tmp_path / "reports")])
    assert args.mode == "council" and args.agents == 2 and args.rounds == 2
    assert _run_review(args) == 0
    assert len(client.calls) == 5  # early convergence
    assert _run_resume(parser.parse_args(["resume", "--review-id", "offline", "--state-db", db])) == 0
    assert len(client.calls) == 5
    report = SQLiteStore(db).report("offline")
    assert "Audit: claims, dissent" in report["content"]
    assert report["budget"]["calls"] == 5


def test_model_provider_key_does_not_fall_back_to_other_provider():
    import pytest

    from arch_council.council_cli import CouncilRouter

    router = CouncilRouter(Settings(api_key="vyceai-only", provider="vyceai"))
    with pytest.raises(ValueError, match="Missing GROQ_API_KEY"):
        router.complete(model="groq/qwen", system="hi", user="hi")
