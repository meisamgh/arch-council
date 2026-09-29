"""Read bounded public source excerpts; never execute downloaded code."""
from __future__ import annotations

import hashlib
import re
from html.parser import HTMLParser
from urllib.parse import urlparse

import requests

from .council import Request
from .research import ResearchError


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


def fetch(url: str) -> str:
    # Fixed provider endpoints only; redirects are not followed to unknown hosts.
    with requests.get(url, timeout=(5, 15), stream=True, allow_redirects=False,
                      headers={"User-Agent": "arch-council/0.8"}) as response:
        if response.status_code != 200:
            raise ResearchError(f"Source HTTP {response.status_code}")
        data = bytearray()
        for chunk in response.iter_content(8192):
            data.extend(chunk)
            if len(data) >= 400_000:
                break
    return bytes(data[:400_000]).decode("utf-8", errors="replace")


def inspect_source(url: str, query: str) -> tuple[str, str]:
    parsed = urlparse(url)
    if parsed.hostname == "github.com":
        parts = parsed.path.strip("/").split("/")
        if len(parts) < 2 or not all(re.fullmatch(r"[\w.-]+", s) for s in parts[:2]):
            raise ResearchError("Invalid GitHub URL")
        import json
        root = "/".join(parts[:2])
        commit = json.loads(fetch(f"https://api.github.com/repos/{root}/commits/HEAD"))["sha"]
        if not re.fullmatch(r"[a-f0-9]{40}", commit):
            raise ResearchError("Invalid commit reference")
        tree = json.loads(fetch(f"https://api.github.com/repos/{root}/git/trees/{commit}?recursive=1"))
        files = [v["path"] for v in tree.get("tree", []) if v.get("type") == "blob"]
        terms = set(re.findall(r"[a-z]{4,}", query.lower()))
        def rank(path):
            return (sum(t in path.lower() for t in terms) +
                    (2 if "test" in path.lower() else 0) +
                    (3 if path.lower() == "readme.md" else 0))
        candidates = sorted((p for p in files if p.endswith((".md", ".py", ".ts", ".rs", ".toml"))
                             and re.fullmatch(r"[\w./-]+", p) and ".." not in p.split("/")),
                            key=rank, reverse=True)[:2]
        excerpts = []
        for path in candidates:
            text = fetch(f"https://raw.githubusercontent.com/{root}/{commit}/{path}")
            lines = text.splitlines()
            start = next((i for i, line in enumerate(lines)
                          if any(t in line.lower() for t in terms)), 0)
            excerpts.append(f"{path}:L{start + 1}\n" + "\n".join(lines[start:start+30])[:1800])
        return "\n\n".join(excerpts), f"GitHub selected files at commit {commit}; not executed"
    if parsed.hostname in {"arxiv.org", "www.arxiv.org"}:
        match = re.search(r"/(?:abs|pdf|html)/(\d{4}\.\d{4,5}(?:v\d+)?)", parsed.path)
        if not match:
            raise ResearchError("No supported arXiv paper ID")
        parser = TextParser()
        parser.feed(fetch(f"https://arxiv.org/html/{match[1]}"))
        text = "\n".join(parser.parts)
        excerpts = []
        for heading in ("method", "experiment", "result", "limitation"):
            offset = text.lower().find(heading, 1000)
            if offset >= 0:
                excerpts.append(f"{heading}: " + text[offset:offset+900])
        return "\n".join(excerpts) or text[:3000], "arXiv HTML section excerpts (not full-paper review)"
    if parsed.hostname in {"docs.python.org", "pydantic.dev", "docs.pydantic.dev", "docs.github.com"}:
        parser = TextParser()
        parser.feed(fetch(url))
        return "\n".join(parser.parts)[:3600], "official documentation excerpt"
    raise ResearchError("Deep inspection unavailable for this host; retaining search snippet")


class CouncilResearch:
    def __init__(self, search, *, inspections=4, results=2, store=None, review_id=""):
        self.search, self.inspections, self.results = search, inspections, results
        self.store, self.review_id = store, review_id
        self.used = 0

    def __call__(self, request: Request) -> dict:
        prefix = {"github": "site:github.com ", "paper": "site:arxiv.org ",
                  "docs": "official documentation ", "web": ""}[request.source]
        try:
            pack = self.search.search_many([prefix + request.query],
                                          max_results_per_query=self.results,
                                          max_sources=min(2, self.results))
        except (ResearchError, requests.RequestException):
            return {"status": "unavailable", "sources": [], "detail": "Search unavailable"}
        sources, gaps = [], []
        for source in pack.sources:
            content, provenance = source.content, "search snippet; source not inspected"
            key = "inspection:" + hashlib.sha256((source.url + request.query).encode()).hexdigest()
            cached = self.store.checkpoint_payload(self.review_id, key) if self.store else None
            used = (self.store.checkpoint_payload(self.review_id, "inspection-count") or 0
                    if self.store else self.used)
            if cached is not None:
                content, provenance = cached["text"], cached["provenance"]
                if not cached.get("inspected", False):
                    gaps.append(source.url)
            elif used < self.inspections:
                self.used = used + 1
                if self.store:
                    self.store.checkpoint(self.review_id, "inspection-count", self.used)
                try:
                    content, provenance = inspect_source(source.url, request.query)
                except (ResearchError, requests.RequestException, ValueError, KeyError):
                    gaps.append(source.url)
                if self.store:
                    self.store.checkpoint(self.review_id, key, {"text": content, "provenance": provenance,
                                                               "inspected": source.url not in gaps})
            else:
                gaps.append(source.url)
            sources.append({"url": source.url, "title": source.title,
                            "excerpt": content[:3600], "provenance": provenance})
        return {"status": "partial" if gaps else "available" if sources else "unavailable",
                "sources": sources, "detail": f"{len(gaps)} sources not deeply inspected"}
