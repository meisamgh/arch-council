import pytest

from arch_council.research import (
    ResearchError,
    ResearchPack,
    ResearchSource,
    SearXNGResearchClient,
    parse_research_queries,
)


def test_parse_research_queries_accepts_bullets_and_prefixes() -> None:
    raw = """\
QUERY: durable execution agent checkpoint recovery
- LangGraph persistence checkpoint retry semantics
* Temporal workflow idempotency activity retries
"""

    queries = parse_research_queries(raw, max_queries=3)

    assert queries == [
        "durable execution agent checkpoint recovery",
        "LangGraph persistence checkpoint retry semantics",
        "Temporal workflow idempotency activity retries",
    ]


def test_research_pack_keeps_source_ids_and_urls() -> None:
    pack = ResearchPack(
        queries=("agent recovery",),
        sources=(
            ResearchSource(
                source_id="S1",
                query="agent recovery",
                title="Example source",
                url="https://example.com/evidence",
                content="Checkpointed workflows can resume after a process failure.",
                score=0.91,
            ),
        ),
    )

    prompt = pack.to_prompt()

    assert "[S1]" in prompt
    assert "https://example.com/evidence" in prompt
    assert "Checkpointed workflows" in prompt


def test_searxng_search_uses_json_api_and_limits_results(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json() -> dict[str, object]:
            return {
                "results": [
                    {
                        "title": "Result one",
                        "url": "https://example.com/1",
                        "content": "First result",
                        "score": 2.0,
                    },
                    {
                        "title": "Result two",
                        "url": "https://example.com/2",
                        "content": "Second result",
                    },
                    {
                        "title": "Result three",
                        "url": "https://example.com/3",
                        "content": "Third result",
                    },
                ]
            }

    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("arch_council.research.requests.get", fake_get)

    client = SearXNGResearchClient("http://localhost:8080")
    results = client.search("agent recovery", max_results=2)

    assert captured["url"] == "http://localhost:8080/search"
    assert captured["params"] == {
        "q": "agent recovery",
        "format": "json",
        "categories": "general",
        "language": "all",
        "safesearch": 1,
    }
    assert len(results) == 2
    assert results[0]["url"] == "https://example.com/1"


def test_searxng_403_explains_json_format_requirement(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status_code = 403
        text = "Forbidden"

    monkeypatch.setattr(
        "arch_council.research.requests.get",
        lambda *args, **kwargs: FakeResponse(),
    )

    client = SearXNGResearchClient("http://localhost:8080")

    with pytest.raises(ResearchError, match="search.formats"):
        client.search("agent recovery")
