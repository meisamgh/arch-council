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


def test_research_cache_isolated_by_provider_and_endpoint(tmp_path) -> None:
    class Provider(CountingResearch):
        def __init__(self, endpoint):
            super().__init__()
            self.base_url = endpoint

    store = SQLiteStore(tmp_path / "state.sqlite3")
    first = Provider("https://search-one.example")
    second = Provider("https://search-two.example")
    CachedResearchClient(first, store).search_many(["same query"])
    CachedResearchClient(second, store).search_many(["same query"])
    assert first.calls == second.calls == 1
    CachedResearchClient(Provider("https://search-one.example"), store).search_many(["same query"])
    assert first.calls == 1
