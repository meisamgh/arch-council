import json
from pathlib import Path

import pytest

from arch_council.context import RepositoryContext
from arch_council.debate import ArchitectureDebate
from arch_council.research import ResearchPack, ResearchSource

SCORES = {
    "repository_fit": 3,
    "external_evidence_strength": 3,
    "testability": 3,
    "predicted_correctness_impact_unverified": 3,
    "complexity_maintainability": 3,
    "recovery_robustness": 3,
    "latency_cost": 3,
    "migration_reversibility": 3,
}


class FakeClient:
    def __init__(self, *, high_first_round: bool = False) -> None:
        self.calls: list[dict[str, object]] = []
        self.debate_calls = 0
        self.high_first_round = high_first_round

    def complete(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        system = str(kwargs.get("system", ""))
        system_lower = system.lower()
        if "evidence planner" in system_lower:
            return "text-to-sql query IR paper\ntext-to-sql architecture github repository"
        if "shared evidence pack" in system_lower and "missing evidence" in system_lower:
            return '{"gaps": []}'
        if "adversarial architecture council" in system_lower:
            self.debate_calls += 1
            first_round = self.debate_calls <= 3
            high = self.high_first_round and first_round
            return json.dumps(
                {
                    "position_markdown": f"debate-position-{self.debate_calls}",
                    "unresolved_objections": [
                        {
                            "target": "shared",
                            "category": "query_ir",
                            "claim": "IR boundary needs proof",
                            "impact": "high" if high else "low",
                            "status": "unresolved" if high else "resolved",
                            "evidence": [],
                        }
                    ],
                    "concessions": [],
                    "evidence_requests": [],
                    "material_architecture_change": False,
                }
            )
        if "impartial architecture arbiter" in system_lower:
            scores = {candidate: dict(SCORES) for candidate in ("A", "B", "C")}
            scores["B"]["external_evidence_strength"] = 4
            return json.dumps({"candidates": scores})
        if "final adr editor" in system_lower:
            return "# Executive Decision\nCandidate B wins."
        return f"response-{len(self.calls)}"


class FakeResearchClient:
    def __init__(self) -> None:
        self.queries: list[tuple[str, ...]] = []

    def search_many(self, queries, **_: object) -> ResearchPack:
        normalized = tuple(queries)
        self.queries.append(normalized)
        return ResearchPack(
            queries=normalized,
            sources=(
                ResearchSource(
                    source_id="S1",
                    query=normalized[0],
                    title="Evidence",
                    url="https://example.com/evidence",
                    content="A typed intermediate representation can separate planning from SQL.",
                    score=0.9,
                ),
            ),
        )


class VerboseClient(FakeClient):
    def complete(self, **kwargs: object) -> str:
        result = super().complete(**kwargs)
        system = str(kwargs.get("system", "")).lower()
        user = str(kwargs.get("user", ""))
        if "architecture council" not in system and (
            "Return a structured architecture proposal" in user
            or "Produce a final revised architecture" in user
        ):
            return "verbose-position-" + ("x" * 50_000)
        return result


def make_context(tmp_path: Path, text: str = "# repo context") -> RepositoryContext:
    return RepositoryContext(
        root=tmp_path,
        text=text,
        included_files=("README.md",),
        truncated=False,
        mode="repository",
    )


def test_dynamic_council_stops_after_first_converged_round(tmp_path: Path) -> None:
    client = FakeClient()
    debate = ArchitectureDebate(
        client,
        model_a="model-a",
        model_b="model-b",
        model_c="model-c",
        rounds=3,
    )
    result = debate.run(question="Which architecture is safer?", context=make_context(tmp_path))

    assert result.rounds_completed == 1
    assert result.rounds_requested == 3
    assert len(client.calls) == 11
    assert result.arbiter_scorecard.winner == "B"
    assert result.decision.startswith("# Executive Decision")


def test_high_impact_objection_forces_another_round(tmp_path: Path) -> None:
    client = FakeClient(high_first_round=True)
    debate = ArchitectureDebate(
        client,
        model_a="model-a",
        model_b="model-b",
        model_c="model-c",
        rounds=3,
    )
    result = debate.run(question="Question", context=make_context(tmp_path))

    assert result.rounds_completed == 2
    assert len(client.calls) == 14


def test_compact_state_is_not_duplicated_and_final_prompts_are_bounded(
    tmp_path: Path,
) -> None:
    client = VerboseClient(high_first_round=True)
    debate = ArchitectureDebate(
        client,
        model_a="model-a",
        model_b="model-b",
        model_c="model-c",
        rounds=2,
        summarizer_model="summary-model",
    )
    debate.run(question="Question", context=make_context(tmp_path, "r" * 50_000))

    debate_calls = [
        call
        for call in client.calls
        if "adversarial architecture council" in str(call["system"]).lower()
    ]
    for call in debate_calls[3:]:
        prompt = str(call["user"])
        assert prompt.count("Shared compact debate state:") == 1
        assert len(prompt) < 18_000

    revision_calls = [
        call
        for call in client.calls
        if "Produce a final revised architecture" in str(call["user"])
    ]
    assert len(revision_calls) == 3
    assert all(len(str(call["user"])) < 16_000 for call in revision_calls)

    adr_call = next(
        call
        for call in client.calls
        if "final adr editor" in str(call["system"]).lower()
    )
    assert "Initial A:" not in str(adr_call["user"])
    assert len(str(adr_call["user"])) < 25_000


def test_round_limit_is_one_through_three() -> None:
    client = FakeClient()
    with pytest.raises(ValueError, match="between 1 and 3"):
        ArchitectureDebate(client, model_a="a", model_b="b", model_c="c", rounds=0)
    with pytest.raises(ValueError, match="between 1 and 3"):
        ArchitectureDebate(client, model_a="a", model_b="b", model_c="c", rounds=4)


def test_initial_proposals_remain_blind(tmp_path: Path) -> None:
    client = FakeClient()
    debate = ArchitectureDebate(
        client,
        model_a="model-a",
        model_b="model-b",
        model_c="model-c",
        rounds=1,
    )
    debate.run(question="Question", context=make_context(tmp_path, "UNIQUE_REPO_EVIDENCE"))

    first_three = [str(call["user"]) for call in client.calls[:3]]
    assert all("UNIQUE_REPO_EVIDENCE" in prompt for prompt in first_three)
    assert "response-1" not in first_three[1]
    assert "response-1" not in first_three[2]
    assert "response-2" not in first_three[2]


def test_debate_round_is_incremental_and_chained(tmp_path: Path) -> None:
    client = FakeClient()
    debate = ArchitectureDebate(
        client,
        model_a="model-a",
        model_b="model-b",
        model_c="model-c",
        rounds=1,
    )
    debate.run(question="Question", context=make_context(tmp_path))

    debate_calls = [
        call
        for call in client.calls
        if "adversarial architecture council" in str(call["system"]).lower()
    ]
    assert len(debate_calls) == 3
    # The first debate response starts the discussion; later agents critique it
    # and extend it instead of receiving only the original proposals.
    assert "debate-position-1" in str(debate_calls[1]["user"])
    assert "debate-position-1" in str(debate_calls[2]["user"])
    assert "debate-position-2" in str(debate_calls[2]["user"])


def test_resume_reuses_completed_proposals(tmp_path: Path) -> None:
    first = FakeClient()
    saved: dict[str, object] = {}
    initial = ArchitectureDebate(
        first,
        model_a="model-a",
        model_b="model-b",
        model_c="model-c",
        rounds=1,
        checkpoint=lambda stage, payload: saved.__setitem__(stage, payload),
    )
    initial.run(question="Question", context=make_context(tmp_path))
    assert len(first.calls) == 11

    resumed = FakeClient()
    replay = ArchitectureDebate(
        resumed,
        model_a="model-a",
        model_b="model-b",
        model_c="model-c",
        rounds=1,
        load_checkpoint=lambda stage: saved.get(stage),
    )
    result = replay.run(question="Question", context=make_context(tmp_path))

    assert result.proposal_a == saved["proposals"]["a"]  # type: ignore[index]
    # Every expensive stage is checkpointed; a completed review needs no model calls.
    assert len(resumed.calls) == 0


def test_research_runs_after_blind_proposals_and_before_debate(tmp_path: Path) -> None:
    client = FakeClient()
    research = FakeResearchClient()
    debate = ArchitectureDebate(
        client,
        model_a="model-a",
        model_b="model-b",
        model_c="model-c",
        rounds=1,
        research_client=research,  # type: ignore[arg-type]
        research_query_count=2,
    )
    result = debate.run(question="Question", context=make_context(tmp_path))

    planner_calls = [
        call for call in client.calls if "evidence planner" in str(call["system"]).lower()
    ]
    coverage_calls = [
        call
        for call in client.calls
        if "shared evidence pack" in str(call["system"]).lower()
        and "missing evidence" in str(call["system"]).lower()
    ]
    debate_calls = [
        call
        for call in client.calls
        if "adversarial architecture council" in str(call["system"]).lower()
    ]
    assert len(planner_calls) == 1
    assert len(coverage_calls) == 3
    assert len(debate_calls) == 3
    assert "https://example.com/evidence" in str(debate_calls[0]["user"])
    assert result.research_queries == (
        "text-to-sql query IR paper",
        "text-to-sql architecture github repository",
    )
    assert result.research_evidence is not None


def test_arbiter_is_a_fresh_role_and_code_computes_winner(tmp_path: Path) -> None:
    client = FakeClient()
    debate = ArchitectureDebate(
        client,
        model_a="model-a",
        model_b="model-b",
        model_c="model-c",
        arbiter_model="model-arbiter",
        rounds=1,
    )
    result = debate.run(question="Question", context=make_context(tmp_path))

    arbiter_calls = [call for call in client.calls if call["model"] == "model-arbiter"]
    assert len(arbiter_calls) == 2
    assert "Debate transcript:" in str(arbiter_calls[0]["user"])
    assert result.arbiter_model == "model-arbiter"
    assert result.arbiter_scorecard.winner == "B"
    assert result.arbiter_scorecard.weighted_totals["B"] > result.arbiter_scorecard.weighted_totals["A"]
