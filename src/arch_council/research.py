from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from typing import Protocol
from urllib.parse import urlparse

import requests


class ResearchError(RuntimeError):
    """Raised when external research cannot be completed safely."""


@dataclass(frozen=True)
class EvidenceQuality:
    authority: int
    directness: int
    recency: int | None = None
    independent_support: int | None = None

    def to_prompt(self) -> str:
        values = [f"authority={self.authority}/4", f"directness={self.directness}/4"]
        if self.recency is not None:
            values.append(f"recency={self.recency}/4")
        if self.independent_support is not None:
            values.append(f"independent_support={self.independent_support}/4")
        return ", ".join(values)


@dataclass(frozen=True)
class ResearchSource:
    source_id: str
    query: str
    title: str
    url: str
    content: str
    score: float | None = None
    source_type: str = "web"
    inspection: str | None = None
    quality: EvidenceQuality | None = None


@dataclass(frozen=True)
class ResearchPack:
    queries: tuple[str, ...]
    sources: tuple[ResearchSource, ...]

    def to_prompt(self) -> str:
        if not self.sources:
            return "No external research sources were returned."

        blocks: list[str] = []
        for source in self.sources:
            score = "" if source.score is None else f"\nSearch relevance score: {source.score:.3f}"
            quality = "" if source.quality is None else f"\nEvidence quality: {source.quality.to_prompt()}"
            inspection = ""
            if source.inspection:
                inspection = f"\nInspected source content:\n{source.inspection}"
            blocks.append(
                f"""[{source.source_id}]
Query: {source.query}
Type: {source.source_type}
Title: {source.title}
URL: {source.url}{score}{quality}
Search evidence: {source.content}{inspection}"""
            )
        return "\n\n".join(blocks)


class ResearchClient(Protocol):
    """Provider-neutral interface consumed by the architecture debate."""

    def search_many(
        self,
        queries: list[str] | tuple[str, ...],
        *,
        max_results_per_query: int = 3,
        max_sources: int = 12,
        max_content_chars: int = 1_200,
    ) -> ResearchPack: ...


class CachedResearchClient:
    """Persistent cache wrapper shared by all research providers."""

    def __init__(self, client: ResearchClient, store: object) -> None:
        self.client = client
        self.store = store
        self.last_cache_hit = False

    def search_many(self, queries, *, max_results_per_query=3, max_sources=12, max_content_chars=1200):
        provider = type(self.client).__name__
        endpoint = getattr(self.client, "base_url", None)
        if endpoint is None and isinstance(self.client, TavilyResearchClient):
            endpoint = "https://api.tavily.com/search"
        key_material = json.dumps(
            ["research-cache-v2", provider, endpoint,
             tuple(sorted(q.strip() for q in queries if q.strip())),
             max_results_per_query, max_sources, max_content_chars],
            separators=(",", ":"),
        )
        key = hashlib.sha256(key_material.encode()).hexdigest()
        cached = self.store.get_research_cache(key)
        if cached is not None:
            self.last_cache_hit = True
            return ResearchPack(
                tuple(cached["queries"]),
                tuple(
                    ResearchSource(
                        **{**source, "quality": EvidenceQuality(**source["quality"]) if source.get("quality") else None}
                    )
                    for source in cached["sources"]
                ),
            )
        self.last_cache_hit = False
        pack = self.client.search_many(
            queries,
            max_results_per_query=max_results_per_query,
            max_sources=max_sources,
            max_content_chars=max_content_chars,
        )
        self.store.save_research_cache(
            key,
            {
                "queries": list(pack.queries),
                "sources": [
                    {**source.__dict__, "quality": asdict(source.quality) if source.quality else None}
                    for source in pack.sources
                ],
            },
        )
        return pack


def _source_type(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if host in {"github.com", "www.github.com"}:
        return "github_repository"
    if host.endswith("arxiv.org"):
        return "research_paper"
    if "docs." in host or "/docs/" in url.lower() or "documentation" in url.lower():
        return "official_docs_candidate"
    return "web"


def _base_quality(source_type: str, inspected: bool) -> EvidenceQuality:
    authority = {
        "research_paper": 4,
        "official_docs_candidate": 3,
        "github_repository": 2,
        "web": 1,
    }.get(source_type, 1)
    return EvidenceQuality(authority=authority, directness=4 if inspected else 2)


class EvidenceInspector:
    """Bounded reader for high-value GitHub repositories and arXiv papers.

    The inspector deliberately does not fetch arbitrary search-result URLs. It only
    deep-reads known public source families used for architecture evidence.
    """

    def __init__(self, timeout_seconds: int = 30, max_chars: int = 6_000) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_chars = max_chars

    def inspect_pack(self, pack: ResearchPack, *, max_sources: int = 4) -> ResearchPack:
        inspection_attempts = 0
        sources: list[ResearchSource] = []
        for source in pack.sources:
            source_type = source.source_type or _source_type(source.url)
            inspection = source.inspection
            if inspection_attempts < max_sources and source_type in {
                "github_repository",
                "research_paper",
            }:
                inspection_attempts += 1
                inspection = self.inspect(source.url, source_type=source_type)
            sources.append(
                ResearchSource(
                    source_id=source.source_id,
                    query=source.query,
                    title=source.title,
                    url=source.url,
                    content=source.content,
                    score=source.score,
                    source_type=source_type,
                    inspection=inspection,
                    quality=_base_quality(source_type, bool(inspection)),
                )
            )
        return ResearchPack(pack.queries, tuple(sources))

    def inspect(self, url: str, *, source_type: str | None = None) -> str | None:
        kind = source_type or _source_type(url)
        try:
            if kind == "github_repository":
                return self._inspect_github(url)
            if kind == "research_paper":
                return self._inspect_arxiv(url)
        except requests.RequestException:
            return None
        return None

    def _inspect_github(self, url: str) -> str | None:
        parsed = urlparse(url)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            return None
        owner, repo = parts[0], parts[1].removesuffix(".git")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", owner) or not re.fullmatch(
            r"[A-Za-z0-9_.-]+", repo
        ):
            return None
        for branch in ("main", "master"):
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.md"
            response = requests.get(raw_url, timeout=self.timeout_seconds)
            if response.status_code == 200 and response.text.strip():
                return response.text.strip()[: self.max_chars]
        return None

    def _inspect_arxiv(self, url: str) -> str | None:
        parsed = urlparse(url)
        match = re.search(r"/(?:abs|pdf)/([^/?#]+)", parsed.path)
        if not match:
            return None
        arxiv_id = match.group(1).removesuffix(".pdf")
        api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
        response = requests.get(api_url, timeout=self.timeout_seconds)
        if response.status_code != 200 or not response.text.strip():
            return None
        try:
            root = ET.fromstring(response.text)
        except ET.ParseError:
            return None
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", namespace)
        if entry is None:
            return None
        title = " ".join((entry.findtext("atom:title", default="", namespaces=namespace)).split())
        summary = " ".join((entry.findtext("atom:summary", default="", namespaces=namespace)).split())
        published = entry.findtext("atom:published", default="", namespaces=namespace)
        text = f"Paper title: {title}\nPublished: {published}\nAbstract: {summary}".strip()
        return text[: self.max_chars] if text else None


class SourceResearchAgent:
    """Specialized bounded evidence reader for one source family."""

    source_type = "web"

    def __init__(self, inspector: EvidenceInspector | None = None) -> None:
        self.inspector = inspector or EvidenceInspector()

    def inspect(self, source: ResearchSource) -> ResearchSource:
        inspection = self.inspector.inspect(source.url, source_type=self.source_type)
        return ResearchSource(
            source_id=source.source_id,
            query=source.query,
            title=source.title,
            url=source.url,
            content=source.content,
            score=source.score,
            source_type=self.source_type,
            inspection=inspection,
            quality=_base_quality(self.source_type, bool(inspection)),
        )


class PaperResearchAgent(SourceResearchAgent):
    source_type = "research_paper"


class GitHubResearchAgent(SourceResearchAgent):
    source_type = "github_repository"


class OfficialDocsResearchAgent(SourceResearchAgent):
    source_type = "official_docs_candidate"

    def inspect(self, source: ResearchSource) -> ResearchSource:
        # Official docs are retained as search evidence unless a caller supplies a
        # provider-specific fetcher; arbitrary pages are not fetched implicitly.
        return ResearchSource(
            source_id=source.source_id,
            query=source.query,
            title=source.title,
            url=source.url,
            content=source.content,
            score=source.score,
            source_type=self.source_type,
            inspection=source.inspection,
            quality=_base_quality(self.source_type, bool(source.inspection)),
        )


class _BaseSearchClient:
    """Shared result packing for search providers."""

    def search(self, query: str, *, max_results: int = 3) -> list[dict[str, object]]:
        raise NotImplementedError

    def search_many(
        self,
        queries: list[str] | tuple[str, ...],
        *,
        max_results_per_query: int = 3,
        max_sources: int = 12,
        max_content_chars: int = 1_200,
    ) -> ResearchPack:
        unique_queries = tuple(dict.fromkeys(q.strip() for q in queries if q.strip()))
        sources: list[ResearchSource] = []
        seen_urls: set[str] = set()

        for query in unique_queries:
            for item in self.search(query, max_results=max_results_per_query):
                url = str(item.get("url", "")).strip()
                if not url or url in seen_urls:
                    continue
                title = str(item.get("title", url)).strip() or url
                content = str(item.get("content", "")).strip()
                if len(content) > max_content_chars:
                    content = content[:max_content_chars] + " ... [truncated]"
                raw_score = item.get("score")
                score = float(raw_score) if isinstance(raw_score, (int, float)) else None
                source_type = _source_type(url)
                source_id = f"S{len(sources) + 1}"
                sources.append(
                    ResearchSource(
                        source_id=source_id,
                        query=query,
                        title=title,
                        url=url,
                        content=content,
                        score=score,
                        source_type=source_type,
                        quality=_base_quality(source_type, False),
                    )
                )
                seen_urls.add(url)
                if len(sources) >= max_sources:
                    return ResearchPack(queries=unique_queries, sources=tuple(sources))

        return ResearchPack(queries=unique_queries, sources=tuple(sources))


class SearXNGResearchClient(_BaseSearchClient):
    """Research client for a SearXNG instance using its JSON search API."""

    def __init__(self, base_url: str = "http://localhost:8080", timeout_seconds: int = 60) -> None:
        base_url = base_url.strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise ResearchError("SEARXNG_URL must start with http:// or https://")
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, *, max_results: int = 3) -> list[dict[str, object]]:
        try:
            response = requests.get(
                f"{self.base_url}/search",
                params={
                    "q": query,
                    "format": "json",
                    "categories": "general",
                    "language": "all",
                    "safesearch": 1,
                },
                headers={"Accept": "application/json"},
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise ResearchError(
                f"SearXNG request failed for {self.base_url}: {exc}. "
                "Make sure the SearXNG service is running."
            ) from exc

        if response.status_code != 200:
            preview = response.text[:500].replace("\n", " ")
            hint = ""
            if response.status_code == 403:
                hint = " Ensure `json` is enabled under `search.formats` in SearXNG settings.yml."
            raise ResearchError(
                f"SearXNG search returned HTTP {response.status_code}: {preview}.{hint}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise ResearchError(
                "SearXNG returned non-JSON content. Ensure the instance supports format=json."
            ) from exc

        results = payload.get("results", [])
        if not isinstance(results, list):
            raise ResearchError("SearXNG response did not contain a results list")

        normalized: list[dict[str, object]] = []
        for item in results[:max_results]:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score"),
                }
            )
        return normalized


class TavilyResearchClient(_BaseSearchClient):
    """Optional Tavily Search API client using requests only."""

    def __init__(self, api_key: str, timeout_seconds: int = 60) -> None:
        api_key = api_key.strip()
        if not api_key:
            raise ResearchError(
                "TAVILY_API_KEY is missing. Add it to .env or choose --research-provider searxng."
            )
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, *, max_results: int = 3) -> list[dict[str, object]]:
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "query": query,
                    "search_depth": "basic",
                    "max_results": max_results,
                    "include_answer": False,
                    "include_raw_content": False,
                },
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise ResearchError(f"External research request failed: {exc}") from exc

        if response.status_code != 200:
            preview = response.text[:500].replace("\n", " ")
            raise ResearchError(f"Tavily search returned HTTP {response.status_code}: {preview}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise ResearchError("Tavily returned a non-JSON response") from exc

        results = payload.get("results", [])
        if not isinstance(results, list):
            raise ResearchError("Tavily response did not contain a results list")
        return [item for item in results if isinstance(item, dict)]


def merge_research_packs(*packs: ResearchPack | None, max_sources: int = 24) -> ResearchPack:
    queries: list[str] = []
    sources: list[ResearchSource] = []
    seen_urls: set[str] = set()
    for pack in packs:
        if pack is None:
            continue
        for query in pack.queries:
            if query not in queries:
                queries.append(query)
        for source in pack.sources:
            if source.url in seen_urls:
                continue
            seen_urls.add(source.url)
            sources.append(
                ResearchSource(
                    source_id=f"S{len(sources) + 1}",
                    query=source.query,
                    title=source.title,
                    url=source.url,
                    content=source.content,
                    score=source.score,
                    source_type=source.source_type,
                    inspection=source.inspection,
                    quality=source.quality,
                )
            )
            if len(sources) >= max_sources:
                return ResearchPack(tuple(queries), tuple(sources))
    return ResearchPack(tuple(queries), tuple(sources))


def parse_research_queries(text: str, *, max_queries: int = 4) -> list[str]:
    """Parse one-query-per-line LLM output defensively."""
    queries: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for prefix in ("- ", "* ", "• "):
            if line.startswith(prefix):
                line = line[len(prefix) :].strip()
        if line.upper().startswith("QUERY:"):
            line = line.split(":", 1)[1].strip()
        if len(line) < 8 or len(line) > 300:
            continue
        if line.startswith("#"):
            continue
        if line not in queries:
            queries.append(line)
        if len(queries) >= max_queries:
            break
    return queries
