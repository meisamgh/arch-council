import json

from arch_council.governance import (
    ARBITER_WEIGHTS,
    DebateSignal,
    EvidenceRequest,
    aggregate_debate_state,
    dedupe_evidence_requests,
    parse_arbiter_scorecard,
    parse_claim_ledger,
    parse_debate_signal,
    should_continue_debate,
)


def test_compact_debate_state_reduces_context_and_preserves_signals() -> None:
    response = '{"position_markdown":"minority position: keep queue","unresolved_objections":[{"claim":"durability","impact":"high","status":"unresolved"}],"concessions":[{"category":"cost","accepted_from":"Architect B"}],"evidence_requests":[{"category":"evaluation","question":"Which benchmark measures leakage?","priority":"high"}],"material_architecture_change":true}'
    state = aggregate_debate_state((response, response, response))
    assert len(state.to_prompt()) < len(response) * 3
    assert state.evidence_requests[0].question == "Which benchmark measures leakage?"
    assert "minority" in state.to_prompt().lower()


def test_evidence_requests_deduplicate_by_taxonomy_category() -> None:
    requests = [
        EvidenceRequest("query_ir", "Need IR paper evidence", "medium"),
        EvidenceRequest("query_ir", "Need SQL semantic IR evidence", "high"),
        EvidenceRequest("workflow_recovery", "Need durable recovery evidence", "medium"),
    ]
    result = dedupe_evidence_requests(requests)
    assert [item.category for item in result] == ["query_ir", "workflow_recovery"]
    assert result[0].priority == "high"


def test_invalid_structured_debate_output_does_not_fake_convergence() -> None:
    signal = parse_debate_signal("not json")
    assert signal.has_high_impact_unresolved is True
    assert signal.material_architecture_change is True


def test_claim_ledger_preserves_targeted_challenges_and_experiments() -> None:
    response = json.dumps(
        {
            "claims": [
                {
                    "claim_id": "A-R1-C1",
                    "statement": "Formula execution must be deterministic Python.",
                    "status": "accepted",
                    "evidence": ["REPOSITORY"],
                    "supersedes": [],
                }
            ],
            "challenges": [
                {
                    "challenge_id": "B-R1-X1",
                    "target_claim_id": "A-R1-C1",
                    "reason": "The compiler boundary has no latency benchmark.",
                    "severity": "high",
                    "status": "open",
                    "evidence_required": "Benchmark compilation and execution latency.",
                }
            ],
            "proposed_experiments": [
                {
                    "experiment_id": "C-R1-E1",
                    "target_claim_ids": ["A-R1-C1"],
                    "hypothesis": "Deterministic execution prevents invalid formulas.",
                    "test": "Compare compiled and direct LLM formula execution.",
                    "metric": "invalid formula rate",
                    "success_criterion": "Compiled invalid rate is below one percent.",
                }
            ],
        }
    )
    ledger = parse_claim_ledger((response, response, response))
    assert ledger.claims[0].claim_id == "A-R1-C1"
    assert ledger.challenges[0].target_claim_id == "A-R1-C1"
    assert ledger.experiments[0].target_claim_ids == ["A-R1-C1"]


def test_open_high_severity_claim_challenge_prevents_convergence() -> None:
    response = json.dumps(
        {
            "challenges": [
                {
                    "challenge_id": "B-R1-X1",
                    "target_claim_id": "A-R1-C1",
                    "reason": "No point-in-time leakage evidence exists.",
                    "severity": "high",
                    "status": "open",
                    "evidence_required": "Point-in-time validation test.",
                }
            ],
            "material_architecture_change": False,
        }
    )
    signal = parse_debate_signal(response)
    assert signal.has_high_impact_unresolved is True


def test_stop_rule_is_deterministic() -> None:
    stable = DebateSignal(False, (), False)
    assert should_continue_debate(round_number=1, max_rounds=3, signals=(stable, stable, stable)) is False

    unresolved = DebateSignal(True, (), False)
    assert should_continue_debate(
        round_number=1,
        max_rounds=3,
        signals=(stable, unresolved, stable),
    ) is True
    assert should_continue_debate(
        round_number=3,
        max_rounds=3,
        signals=(unresolved,),
    ) is False


def test_weighted_arbiter_score_is_computed_by_code() -> None:
    base = {criterion: 2 for criterion in ARBITER_WEIGHTS}
    candidates = {candidate: dict(base) for candidate in ("A", "B", "C")}
    candidates["C"]["external_evidence_strength"] = 4
    scorecard = parse_arbiter_scorecard(json.dumps({"candidates": candidates}))
    assert scorecard.winner == "C"
    assert scorecard.weighted_totals["C"] > scorecard.weighted_totals["A"]
