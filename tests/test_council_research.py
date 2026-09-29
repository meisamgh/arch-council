import json

from arch_council.council import Claim, Council, CouncilState, Request, ResearchRequest
from arch_council.research import ResearchError, ResearchPack, ResearchSource
from arch_council.research_tools import CouncilResearch, inspect_source
from arch_council.runtime.persistence import SQLiteStore


def test_one_source_unavailable_retains_partial_evidence_and_cached_inspection(tmp_path, monkeypatch):
    class Search:
        def search_many(self, queries, **kwargs):
            return ResearchPack(tuple(queries), (
                ResearchSource("S1", queries[0], "Paper", "https://arxiv.org/abs/1", "snippet"),
                ResearchSource("S2", queries[0], "Repo", "https://github.com/a/b", "fallback")))
    calls = []
    def inspect(url, query):
        calls.append(url)
        if "github" in url:
            raise ResearchError("fixture unavailable")
        return "Methods and limitations", "paper HTML"
    monkeypatch.setattr("arch_council.research_tools.inspect_source", inspect)
    store = SQLiteStore(tmp_path / "db")
    request = Request(claim_id="C001", query="point in time features", purpose="Test the leakage claim")
    tool = CouncilResearch(Search(), store=store, review_id="r", inspections=2)
    result = tool(request)
    assert result["status"] == "partial"
    assert len(result["sources"]) == 2
    assert result["sources"][1]["excerpt"] == "fallback"
    assert result["sources"][1]["inspection_status"] == "snippet"
    assert result["sources"][0]["source_type"] == "paper"
    assert result["sources"][0]["retrieved_at"]
    replay = CouncilResearch(Search(), store=store, review_id="r", inspections=2)(request)
    assert len(calls) == 2
    assert replay["sources"] == result["sources"]
    assert replay["status"] == "partial"


def test_github_inspection_pins_commit_and_reads_test_file(monkeypatch):
    urls = []
    sha = "a" * 40
    def fetch(url):
        urls.append(url)
        if "/commits/" in url:
            return json.dumps({"sha": sha})
        if "/git/trees/" in url:
            return json.dumps({"tree": [{"path": "README.md", "type": "blob"},
                                        {"path": "tests/test_leakage.py", "type": "blob"}]})
        return "def test_leakage():\n    assert available_at <= prediction_time"
    monkeypatch.setattr("arch_council.research_tools.fetch", fetch)
    text, provenance = inspect_source("https://github.com/example/project", "leakage")
    assert "tests/test_leakage.py:L1" in text
    assert sha in provenance
    assert all(sha in u for u in urls if "raw.githubusercontent.com" in u)


def test_arxiv_sections_are_labeled_excerpts(monkeypatch):
    monkeypatch.setattr("arch_council.research_tools.fetch", lambda _: (
        "<p>" + "intro " * 200 + "</p><h2>Methods</h2><p>Temporal split</p>"
        "<h2>Limitations</h2><p>Small dataset</p>"))
    text, provenance = inspect_source("https://arxiv.org/abs/2210.03629", "temporal split")
    assert "Temporal split" in text
    assert "Small dataset" in text
    assert "not full-paper review" in provenance


def test_inspection_metadata_reaches_ledger_and_decision_prompt(tmp_path):
    class Client:
        before_attempt = None

    store = SQLiteStore(tmp_path / "db")
    source = {"url": "https://github.com/example/project", "title": "Code",
              "excerpt": "test evidence", "provenance": "GitHub selected files at commit " + "a" * 40,
              "source_type": "github", "inspection_status": "inspected",
              "commit_sha": "a" * 40, "authority": "primary",
              "directness": "implementation", "retrieved_at": "2026-09-29T12:00:00+00:00"}
    council = Council(client=Client(), store=store, review_id="metadata", topic="Review",
                      models={"A": "a", "B": "b"})
    council.state = CouncilState(claims={"C001": Claim(
        id="C001", owner="A", statement="Inspect repository implementation")},
        requests={"Q001": ResearchRequest(id="Q001", owner="B", claim_id="C001",
                                           query="Inspect implementation", purpose="Check the code")})
    council.research = lambda _: {"status": "available", "sources": [source]}
    council.service_research()
    evidence = next(iter(council.state.evidence.values()))
    assert evidence.commit_sha == "a" * 40
    assert evidence.inspection_status == "inspected"
    packet = json.loads(council.prompt("judge", final=True))
    assert packet["evidence_index"][0]["directness"] == "implementation"
