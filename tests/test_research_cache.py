from arch_council.research import CachedResearchClient, ResearchPack, ResearchSource
from arch_council.runtime.persistence import SQLiteStore


class CountingResearch:
    def __init__(self) -> None:
        self.calls = 0

    def search_many(self, queries, **kwargs):
        self.calls += 1
        return ResearchPack(tuple(queries), (ResearchSource("S1", "q", "title", "https://x", "e"),))


def test_research_cache_hit_skips_provider(tmp_path) -> None:
    provider = CountingResearch()
    store = SQLiteStore(tmp_path / "state.sqlite3")
    cached = CachedResearchClient(provider, store)

    cached.search_many(["architecture evidence"])
    result = cached.search_many(["architecture evidence"])

    assert provider.calls == 1
    assert cached.last_cache_hit is True
    assert result.sources[0].title == "title"
