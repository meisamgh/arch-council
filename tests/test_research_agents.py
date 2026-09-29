from arch_council.research import (
    GitHubResearchAgent,
    OfficialDocsResearchAgent,
    PaperResearchAgent,
    ResearchSource,
)


class FakeInspector:
    def inspect(self, url: str, *, source_type: str | None = None) -> str:
        return f"inspected {source_type}: {url}"


def source(url: str) -> ResearchSource:
    return ResearchSource("S1", "query", "title", url, "search result")


def test_specialized_research_agents_preserve_source_family() -> None:
    assert PaperResearchAgent(FakeInspector()).inspect(source("https://arxiv.org/abs/1")).source_type == "research_paper"
    assert GitHubResearchAgent(FakeInspector()).inspect(source("https://github.com/a/b")).source_type == "github_repository"
    assert OfficialDocsResearchAgent(FakeInspector()).inspect(source("https://docs.example.com/api")).source_type == "official_docs_candidate"


def test_official_docs_agent_does_not_fetch_arbitrary_pages() -> None:
    result = OfficialDocsResearchAgent(FakeInspector()).inspect(source("https://docs.example.com/api"))
    assert result.inspection is None
    assert result.quality is not None
