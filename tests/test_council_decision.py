import json

import pytest

from arch_council.council import (
    Council,
    CouncilState,
    ResearchRequest,
    Turn,
    apply_turn,
    converged,
    render_decision,
    validate_decision,
)
from arch_council.runtime.persistence import SQLiteStore


def state_with_claim():
    state = CouncilState()
    apply_turn(state, "A", Turn(position="Propose", claims=["Use a deterministic compiler"]), "a")
    apply_turn(state, "B", Turn(position="Support", reviews=[{
        "claim_id": "C001", "verdict": "support", "reason": "The compiler is reproducible"}]), "b")
    return state


def decision(**overrides):
    return {
        "recommendation": "Use the compiler after validation.",
        "accepted_claim_ids": ["C001"],
        "conditional_claim_ids": [],
        "rejected_claim_ids": [],
        "unresolved_claim_ids": [],
        "open_challenge_ids": [],
        "required_experiment_ids": [],
        "evidence_ids": [],
        "limitations": [],
        "minority_reasoning": [],
        **overrides,
    }


def test_decision_rejects_unknown_and_missing_claim_ids():
    state = state_with_claim()
    with pytest.raises(ValueError, match="unknown claim"):
        validate_decision(decision(accepted_claim_ids=["C099"]), state)
    with pytest.raises(ValueError, match="classify every claim"):
        validate_decision(decision(accepted_claim_ids=[]), state)


def test_open_objection_cannot_be_reported_as_unqualified_acceptance():
    state = state_with_claim()
    apply_turn(state, "B", Turn(position="Object", challenges=[{
        "claim_id": "C001", "reason": "Leakage checks are missing", "severity": "high"}]), "objection")
    with pytest.raises(ValueError, match="open challenge"):
        validate_decision(decision(open_challenge_ids=["X001"]), state)
    with pytest.raises(ValueError, match="open_challenge_ids"):
        validate_decision(decision(accepted_claim_ids=[], unresolved_claim_ids=["C001"]), state)
    resolved = validate_decision(decision(accepted_claim_ids=[], unresolved_claim_ids=["C001"],
                                          open_challenge_ids=["X001"],
                                          minority_reasoning=["B objects to missing leakage checks"]), state)
    assert "X001" in render_decision(resolved, state)


def test_terminal_research_gap_can_converge_only_with_explicit_action():
    state = state_with_claim()
    state.requests["Q001"] = ResearchRequest(
        id="Q001", owner="B", claim_id="C001", query="Search official docs",
        purpose="Verify implementation details", status="unavailable")
    assert not converged(state, changed=False)
    apply_turn(state, "A", Turn(position="Accept uncertainty", gap_actions=[{
        "request_id": "Q001", "disposition": "accept_with_uncertainty",
        "reason": "Both reviewers acknowledge missing external verification"}]), "gap-a")
    assert not converged(state, changed=False)
    apply_turn(state, "B", Turn(position="Accept uncertainty", gap_actions=[{
        "request_id": "Q001", "disposition": "accept_with_uncertainty",
        "reason": "Proceed conditionally with stated verification gap"}]), "gap-b")
    assert converged(state, changed=False)
    with pytest.raises(ValueError, match="research gap"):
        validate_decision(decision(), state)
    validated = validate_decision(decision(accepted_claim_ids=[],
                                            conditional_claim_ids=["C001"],
                                            limitations=["External evidence unavailable"]), state)
    assert "External evidence unavailable" in render_decision(validated, state)


def test_pending_request_still_blocks_convergence():
    state = state_with_claim()
    state.requests["Q001"] = ResearchRequest(
        id="Q001", owner="B", claim_id="C001", query="Search official docs",
        purpose="Verify implementation details")
    assert not converged(state, changed=False)


@pytest.mark.parametrize("disposition,verdict", [
    ("reject", "reject"), ("experiment_required", "experiment"),
])
def test_terminal_gap_rejection_or_experiment_can_converge(disposition, verdict):
    state = state_with_claim()
    state.requests["Q001"] = ResearchRequest(
        id="Q001", owner="B", claim_id="C001", query="Find primary evidence",
        purpose="Resolve external uncertainty", status="partial")
    extra = {"experiments": [{"claim_id": "C001", "test": "Run a controlled validation",
                              "metric": "error rate", "success_criterion": "Below baseline"}]}
    apply_turn(state, "A", Turn(position="Resolve gap", gap_actions=[{
        "request_id": "Q001", "disposition": disposition,
        "reason": "This evidence gap changes the decision"}],
        **(extra if disposition == "experiment_required" else {})), "gap-a")
    apply_turn(state, "B", Turn(position="Agree on gap", reviews=[{
        "claim_id": "C001", "verdict": verdict, "reason": "The evidence is insufficient"}],
        gap_actions=[{"request_id": "Q001", "disposition": disposition,
                      "reason": "The research gap is material"}]), "gap-b")
    assert converged(state, changed=False)
    if disposition == "reject":
        validated = validate_decision(decision(accepted_claim_ids=[],
                                                rejected_claim_ids=["C001"]), state)
    else:
        validated = validate_decision(decision(accepted_claim_ids=[],
                                                unresolved_claim_ids=["C001"],
                                                required_experiment_ids=["E001"],
                                                minority_reasoning=["Validation pending"]), state)
    assert validated.recommendation


def test_final_decision_repair_uses_validation_errors_and_preserves_raw(tmp_path):
    class Client:
        before_attempt = None

        def __init__(self):
            self.calls = []

        def complete(self, **kwargs):
            self.calls.append(kwargs)
            if self.before_attempt:
                self.before_attempt(kwargs["system"], kwargs["user"], kwargs["max_tokens"])
            if len(self.calls) == 1:
                return json.dumps(decision(accepted_claim_ids=["C999"]))
            assert "unknown claim" in kwargs["user"]
            return json.dumps(decision())

    store = SQLiteStore(tmp_path / "db")
    client = Client()
    council = Council(client=client, store=store, review_id="decision", topic="Review",
                      models={"A": "a", "B": "b"})
    council.state = state_with_claim()
    result = council.decide("decision", "b", council.prompt("judge", final=True))
    assert result.accepted_claim_ids == ["C001"]
    assert len(client.calls) == 2
    assert store.checkpoint_payload("decision", "raw:decision:0")
    assert store.checkpoint_payload("decision", "decision:validated")
