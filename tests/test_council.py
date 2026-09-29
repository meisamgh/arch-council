import json

import pytest

from arch_council.council import Council, CouncilState, Turn, apply_turn, converged
from arch_council.runtime.persistence import SQLiteStore


def turn(**kwargs):
    return Turn.model_validate({"position": "A practical design", **kwargs})


def test_claim_ownership_and_unresolved_challenge_survive_silence():
    state = CouncilState()
    apply_turn(state, "A", turn(claims=["Compile features deterministically"]), "a1")
    apply_turn(state, "B", turn(challenges=[{
        "claim_id": "C001", "reason": "Missing leakage validation", "severity": "high"
    }]), "b1")
    apply_turn(state, "A", turn(), "a2")
    assert not converged(state, changed=False)
    with pytest.raises(ValueError, match="owner"):
        apply_turn(state, "B", turn(revisions=[{
            "claim_id": "C001", "statement": "Use arbitrary model generated code"
        }]), "b2")
    with pytest.raises(ValueError, match="challenger"):
        apply_turn(state, "A", turn(resolutions=[{
            "challenge_id": "X001", "reason": "I decided it was fine"
        }]), "a3")


class FakeClient:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail
        self.before_attempt = None

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        if self.before_attempt:
            self.before_attempt(kwargs["system"], kwargs["user"], kwargs["max_tokens"])
        if kwargs["stage"] == self.fail:
            raise RuntimeError("interrupted")
        if kwargs["stage"] == "decision":
            packet = json.loads(kwargs["user"].split("\nCoordinator stop reason:", 1)[0])
            claims = [c["id"] for c in packet["claim_index"]]
            open_objections = [c["id"] for c in packet["open_challenges"]]
            gaps = [r for r in packet["research_status"] if r["status"] != "available"]
            unresolved = sorted({c["claim"] for c in packet["open_challenges"]}
                                | {r["claim"] for r in gaps})
            return json.dumps({
                "recommendation": "Use a deterministic compiler. Validate leakage before launch.",
                "accepted_claim_ids": [cid for cid in claims if cid not in unresolved],
                "conditional_claim_ids": [], "rejected_claim_ids": [],
                "unresolved_claim_ids": unresolved,
                "open_challenge_ids": open_objections,
                "required_experiment_ids": [], "evidence_ids": [],
                "limitations": ["External evidence is incomplete"] if gaps else [],
                "minority_reasoning": ["An objection remains open"] if unresolved else [],
            })
        if kwargs["stage"] == "proposal-A":
            return json.dumps({"position": "Use a compiler", "claims": ["Compile features deterministically"]})
        if kwargs["stage"].endswith("-B") and kwargs["stage"].startswith("round"):
            return json.dumps({"position": "I support the compiler", "reviews": [
                {"claim_id": "C001", "verdict": "support", "reason": "A compiler is reproducible"}]})
        return json.dumps({"position": "Use a deterministic compiler", "claims": []})


def test_resume_reuses_each_turn_and_prompt_does_not_grow(tmp_path):
    store = SQLiteStore(tmp_path / "state.db")
    first = FakeClient(fail="round-1-B")
    options = {"store": store, "review_id": "r1", "topic": "Feature discovery",
               "context": "README " * 1000, "models": {"A": "a", "B": "b"}, "rounds": 2}
    with pytest.raises(RuntimeError, match="interrupted"):
        Council(client=first, **options).run()
    second = FakeClient()
    result = Council(client=second, **options).run()
    assert all(c["stage"] not in {"proposal-A", "proposal-B", "round-1-A"}
               for c in second.calls)
    assert result["stop_reason"] == "converged"
    assert max(len(c["system"]) + len(c["user"]) for c in second.calls) <= 16000
    third = FakeClient()
    Council(client=third, **options).run()
    assert third.calls == []


def test_resume_rejects_changed_topic(tmp_path):
    store = SQLiteStore(tmp_path / "state.db")
    kwargs = {"client": FakeClient(), "store": store, "review_id": "same", "models": {"A": "a", "B": "b"}}
    Council(topic="First topic", **kwargs).run()
    with pytest.raises(ValueError, match="configuration"):
        Council(topic="Changed topic", **kwargs).run()


def test_bad_reference_is_rejected_without_partial_mutation():
    state = CouncilState()
    with pytest.raises(ValueError, match="unknown claim"):
        apply_turn(state, "A", turn(claims=["Use typed formulas"], challenges=[{
            "claim_id": "missing", "reason": "Not a real claim", "severity": "high"
        }]), "a1")
    assert not state.claims


def test_blind_proposal_ignores_accidental_cross_agent_actions(tmp_path):
    class SpeculativeProposal(FakeClient):
        def complete(self, **kwargs):
            result = super().complete(**kwargs)
            if kwargs["stage"] == "proposal-B":
                return json.dumps({"position": "A proposal", "claims": ["Use typed formulas"],
                                   "challenges": [{"claim_id": "C-agents", "reason": "speculative",
                                                   "severity": "high"}]})
            return result
    result = Council(client=SpeculativeProposal(), store=SQLiteStore(tmp_path / "db"),
                     review_id="blind", topic="Feature discovery", models={"A": "a", "B": "b"},
                     rounds=1).run()
    assert result["stop_reason"] in {"max_rounds", "converged"}


def test_revision_reopens_objections_and_invalidates_support():
    state = CouncilState()
    apply_turn(state, "A", turn(claims=["Use typed formulas"]), "a1")
    apply_turn(state, "B", turn(challenges=[{
        "claim_id": "C001", "reason": "Test the compiler", "severity": "high"
    }]), "b1")
    apply_turn(state, "B", turn(resolutions=[{
        "challenge_id": "X001", "reason": "Evidence resolved it"
    }], reviews=[{"claim_id": "C001", "verdict": "support", "reason": "Typed execution helps"}]), "b2")
    assert converged(state, changed=False)
    apply_turn(state, "A", turn(revisions=[{
        "claim_id": "C001", "statement": "Use untyped execution instead"
    }]), "a2")
    assert not converged(state, changed=False)
    assert state.challenges["X001"].status == "open"


class ResearchClient(FakeClient):
    def complete(self, **kwargs):
        response = super().complete(**kwargs)
        if kwargs["stage"] == "round-1-B":
            return json.dumps({"position": "Need leakage evidence", "challenges": [{
                "claim_id": "C001", "reason": "Need point in time proof", "severity": "high"
            }], "evidence_requests": [{"claim_id": "C001", "query": "Point in time features",
                                       "source": "paper", "purpose": "Verify leakage controls"}]})
        if kwargs["stage"] == "round-1-B-evidence":
            assert "https://arxiv.org/abs/example" in kwargs["user"]
            return json.dumps({"position": "Evidence requires a leakage experiment", "reviews": [{
                "claim_id": "C001", "verdict": "experiment", "reason": "Paper is insufficient for this dataset"
            }], "experiments": [{"claim_id": "C001", "test": "Validate point in time joins",
                                 "metric": "future rows", "success_criterion": "zero future rows"}]})
        return response


def test_requester_receives_evidence_and_dissent_survives_resume(tmp_path):
    queries = []
    def research(request):
        queries.append(request.query)
        return {"status": "partial", "detail": "one repo unavailable", "sources": [{
            "url": "https://arxiv.org/abs/example", "title": "Leakage evaluation",
            "excerpt": "Point in time joins require timestamp controls"}]}
    store = SQLiteStore(tmp_path / "db")
    first = ResearchClient(fail="round-2-A")
    options = {"store": store, "review_id": "research", "topic": "Feature discovery",
               "models": {"A": "a", "B": "b"}, "research": research}
    with pytest.raises(RuntimeError):
        Council(client=first, **options).run()
    second = ResearchClient()
    result = Council(client=second, **options).run()
    assert len(queries) == 1
    assert result["stop_reason"] == "max_rounds"
    assert result["state"]["challenges"]["X001"]["status"] == "open"
    assert result["state"]["experiments"]["E001"]["metric"] == "future rows"
    decision = next(c for c in second.calls if c["stage"] == "decision")
    assert "Paper is insufficient" in decision["user"]


def test_prompt_keeps_complete_ids_and_objections_without_transcript(tmp_path):
    council = Council(client=FakeClient(), store=SQLiteStore(tmp_path / "db"), review_id="s",
                      topic="Feature discovery", models={"A": "a", "B": "b"})
    for i in range(8):
        apply_turn(council.state, "A", turn(claims=[f"Mechanism {i} " + "x" * 300]), f"a{i}")
    apply_turn(council.state, "B", turn(challenges=[{
        "claim_id": "C008", "reason": "Important minority objection at the end", "severity": "high"
    }]), "b1")
    prompt = council.prompt("A")
    data = json.loads(prompt)
    assert len(data["claim_index"]) == 8
    assert data["active_claims"][0]["id"] == "C008"
    assert "Important minority objection" in prompt
    assert len(prompt) < 16000


def test_attempt_budget_is_persisted_and_can_resume_with_higher_allowance(tmp_path):
    store = SQLiteStore(tmp_path / "db")
    options = {"store": store, "review_id": "budget", "topic": "Feature discovery",
               "models": {"A": "a", "B": "b"}}
    first = Council(client=FakeClient(), max_calls=2, **options).run()
    assert first["stop_reason"] == "budget_exhausted"
    assert first["budget"]["calls"] == 2
    client = FakeClient()
    second = Council(client=client, max_calls=8, **options).run()
    assert second["stop_reason"] == "converged"
    assert second["budget"]["calls"] == 5
    assert not any(c["stage"].startswith("proposal") for c in client.calls)


def test_transport_retries_consume_shared_budget(tmp_path, monkeypatch):
    import requests

    from arch_council.client import AnthropicGatewayClient
    from arch_council.council import BudgetExceeded

    calls = []
    def unavailable(*args, **kwargs):
        calls.append(kwargs)
        raise requests.Timeout("offline fixture")
    monkeypatch.setattr("arch_council.client.requests.post", unavailable)
    monkeypatch.setattr("arch_council.client.time.sleep", lambda _: None)
    client = AnthropicGatewayClient("fixture-key", "https://example.com", max_retries=5)
    council = Council(client=client, store=SQLiteStore(tmp_path / "db"), review_id="retry",
                      topic="Topic", models={"A": "a", "B": "b"}, max_calls=2)
    with pytest.raises(BudgetExceeded):
        client.complete(model="a", system="instructions", user="question", max_tokens=100)
    assert len(calls) == 2
    assert council.load("budget")["calls"] == 2


def test_invalid_output_repair_is_bounded_and_audited(tmp_path):
    class InvalidFirst(FakeClient):
        def complete(self, **kwargs):
            result = super().complete(**kwargs)
            return "not JSON" if len(self.calls) == 1 else result
    store = SQLiteStore(tmp_path / "db")
    client = InvalidFirst()
    result = Council(client=client, store=store, review_id="repair", topic="Topic",
                     models={"A": "a", "B": "b"}).run()
    assert result["budget"]["calls"] == 6
    assert store.checkpoint_payload("repair", "raw:proposal-A:0") == "not JSON"
    assert any("Validation errors" in c["user"] for c in client.calls)


def test_long_position_repair_receives_candidate_and_field_limit(tmp_path):
    long_position = "Important customer evidence. " * 40

    class LongFirst(FakeClient):
        def complete(self, **kwargs):
            super().complete(**kwargs)
            if len(self.calls) == 1:
                return json.dumps({"position": long_position,
                                   "claims": ["Validate the shared customer problem"]})
            assert "Invalid candidate" in kwargs["user"]
            assert "position: string_too_long" in kwargs["user"]
            assert long_position in kwargs["user"]
            return json.dumps({"position": "Validate customer evidence before choosing a prototype.",
                               "claims": ["Validate the shared customer problem"]})

    client = LongFirst()
    council = Council(client=client, store=SQLiteStore(tmp_path / "db"), review_id="long",
                      topic="Give a polished 850-word final answer", models={"A": "a", "B": "b"})
    result = council.invoke("proposal-A", "a", council.prompt("A", blind=True), "A")
    assert len(result.position) <= 800
    assert result.claims == ["Validate the shared customer problem"]
    assert len(client.calls) == 2
    assert "final report" in client.calls[0]["system"]


def test_long_experiment_metric_repair_receives_all_field_limits(tmp_path):
    long_metric = "false-positive survivor count and mean apparent lift " * 4

    class LongMetricFirst(FakeClient):
        def complete(self, **kwargs):
            super().complete(**kwargs)
            if len(self.calls) == 1:
                return json.dumps({
                    "position": "Test the feature-selection funnel against a null target.",
                    "experiments": [{
                        "claim_id": "C001",
                        "test": "Shuffle the target and run the complete selection funnel.",
                        "metric": long_metric,
                        "success_criterion": "False-positive rate stays below the configured limit.",
                    }],
                })
            repair_prompt = kwargs["user"]
            assert "experiments.0.metric: string_too_long" in repair_prompt
            assert "experiment metric <=100" in repair_prompt
            assert "experiment success_criterion <=200" in repair_prompt
            assert long_metric in repair_prompt
            return json.dumps({
                "position": "Test the feature-selection funnel against a null target.",
                "experiments": [{
                    "claim_id": "C001",
                    "test": "Shuffle the target and run the complete selection funnel.",
                    "metric": "false-positive survivor rate",
                    "success_criterion": "False-positive rate stays below the configured limit.",
                }],
            })

    client = LongMetricFirst()
    council = Council(client=client, store=SQLiteStore(tmp_path / "db"),
                      review_id="long-metric", topic="Topic", models={"A": "a", "B": "b"})
    apply_turn(council.state, "A", turn(claims=["Use a staged feature-selection funnel"]),
               "proposal-A")
    result = council.invoke("round-1-B", "b", council.prompt("B"), "B")
    assert result.experiments[0].metric == "false-positive survivor rate"
    assert len(client.calls) == 2
    assert "experiment metric <=100 chars" in client.calls[0]["system"]


def test_resume_after_two_cached_invalid_positions_can_repair(tmp_path):
    store = SQLiteStore(tmp_path / "db")
    for attempt in (0, 1):
        store.checkpoint("retry", f"raw:proposal-A:{attempt}",
                         json.dumps({"position": "x" * 850}))
    client = FakeClient()
    council = Council(client=client, store=store, review_id="retry", topic="Topic",
                      models={"A": "a", "B": "b"})
    result = council.invoke("proposal-A", "a", council.prompt("A", blind=True), "A")
    assert len(result.position) <= 800
    assert len(client.calls) == 1
    assert council.load("raw:proposal-A:2") is not None


def test_non_object_output_is_repaired(tmp_path):
    class ArrayFirst(FakeClient):
        def complete(self, **kwargs):
            normal = super().complete(**kwargs)
            return "[]" if len(self.calls) == 1 else normal

    client = ArrayFirst()
    council = Council(client=client, store=SQLiteStore(tmp_path / "db"), review_id="array",
                      topic="Topic", models={"A": "a", "B": "b"})
    assert council.invoke("proposal-A", "a", council.prompt("A", blind=True), "A").position
    assert len(client.calls) == 2


def test_repeated_oversized_positions_stop_without_silent_truncation(tmp_path):
    from arch_council.client import LLMError

    class AlwaysLong(FakeClient):
        def complete(self, **kwargs):
            super().complete(**kwargs)
            return json.dumps({"position": "critical caveat " * 3000})

    client = AlwaysLong()
    council = Council(client=client, store=SQLiteStore(tmp_path / "db"), review_id="bounded",
                      topic="Topic", models={"A": "a", "B": "b"}, max_input_chars=8000)
    with pytest.raises(LLMError, match="after two repairs"):
        council.invoke("proposal-A", "a", council.prompt("A", blind=True), "A")
    assert len(client.calls) == 3
    assert council.load("budget")["calls"] == 3
    assert council.load("turn:proposal-A") is None
    assert len(json.loads(council.load("raw:proposal-A:2"))["position"]) > 800
    assert all(len(c["system"]) + len(c["user"]) <= 8000 for c in client.calls)
    assert client.calls[2]["user"].count("Invalid candidate:") == 1


def test_long_topic_is_preserved_and_total_prompt_cap_still_applies(tmp_path):
    from arch_council.council import BudgetExceeded

    topic = "Review customer discovery. " * 120 + "FINAL_REQUIREMENT"
    client = FakeClient()
    council = Council(client=client, store=SQLiteStore(tmp_path / "db"), review_id="long-topic",
                      topic=topic, models={"A": "a", "B": "b"}, max_input_chars=24000)
    assert json.loads(council.prompt("A", blind=True))["topic"] == topic
    assert council.run()["stop_reason"] == "converged"
    assert all("FINAL_REQUIREMENT" in call["user"] for call in client.calls)
    council.topic = "x" * 12000
    council.max_input_chars = 8000
    with pytest.raises(BudgetExceeded):
        council.prompt("A", blind=True)


@pytest.mark.parametrize("topic", ["", "   ", "x" * 12001])
def test_topic_limit_rejects_empty_or_excessive_input(tmp_path, topic):
    with pytest.raises(ValueError, match="12000"):
        Council(client=FakeClient(), store=SQLiteStore(tmp_path / "db"), review_id="invalid",
                topic=topic, models={"A": "a", "B": "b"})
