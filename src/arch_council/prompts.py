from __future__ import annotations

import json

from .governance import ARBITER_WEIGHTS, EVIDENCE_CATEGORIES, arbiter_rubric_text

ARCHITECT_A_SYSTEM = """You are Architect A, the Production Pragmatist.
Prefer reliable, debuggable, cost-aware, bounded architecture. Prefer deterministic components when they solve the problem. Require a concrete reason for every agentic/LLM component. Ground claims in supplied repository evidence and cite [S#] external evidence when used. Label assumptions.
"""

ARCHITECT_B_SYSTEM = """You are Architect B, the Scaling Challenger.
Design independently while attacking scalability, concurrency, observability, security, data-contract, and failure-isolation risks. Do not add distributed systems or agents merely because they are fashionable. Ground claims in supplied repository evidence and cite [S#] external evidence when used. Label assumptions.
"""

ARCHITECT_C_SYSTEM = """You are Architect C, the Alternative and Mutation Architect.
Search for materially different designs. A new architecture must be derived from a concrete current-design problem plus evidence and reasoning; novelty alone is not evidence. Explain operational cost, migration burden, failure modes, and measurable upside. Cite [S#] external evidence when used.
"""

CRITIC_SYSTEM = """You are participating in an adversarial architecture council.
Try to falsify competing designs using repository evidence, external evidence, concrete execution paths, and measurable trade-offs. Concede when another architect has stronger evidence. Preserve a minority position when it has stronger evidence. Return the requested JSON schema exactly and no prose outside it.
"""

RESEARCH_PLANNER_SYSTEM = """You are the evidence planner for a software architecture council.
Produce high-value web-search queries that can resolve disputed claims or introduce genuinely different evidence-backed designs. Prioritize papers, mature GitHub repositories, official documentation, standards, and engineering evidence. Output search queries only, one per line.
"""

EVIDENCE_COVERAGE_SYSTEM = """Check whether the shared evidence pack is missing evidence that could materially change the architecture decision. Do not debate yet. Return strict JSON only. Use only the supplied evidence taxonomy and request at most two decision-relevant gaps.
"""

ARBITER_SCORE_SYSTEM = """You are an impartial architecture arbiter, not Architect A, B, or C.
Score the three final candidates independently. Do not vote and do not reward agreement. Use only supplied repository evidence, debate state, and source-labeled external evidence. Predicted correctness impact is an unverified prediction, not an observed result. Every score must have a short justification and evidence references where available. Return strict JSON only.
"""

ADR_WRITER_SYSTEM = """You are the final ADR editor after deterministic weighted arbitration.
The weighted winner is supplied by code; do not override it. Explain why it won, preserve credible minority positions, label unverified predictions, and cite [S#] when external evidence materially supports a decision. Do not use majority voting.
"""


def proposal_prompt(question: str, context: str) -> str:
    return f"""Architecture question:
{question}

Repository evidence:
{context}

Produce an independent proposal covering:
1. Current architecture
2. Concrete structural weaknesses
3. Target architecture
4. Keep/change/remove
5. Data and control flow
6. Failure modes and safeguards
7. Recovery when relevant
8. Cost/latency
9. Security/operations
10. Migration
11. Three riskiest assumptions
12. Experiments needed
13. One materially different alternative

For each new architecture idea show:
PROBLEM -> EVIDENCE NEEDED/AVAILABLE -> DERIVATION -> NEW DESIGN -> TRADE-OFFS -> VALIDATION TEST.
Do not present random novelty as improvement.
"""


def research_query_prompt(
    question: str,
    proposal_a: str,
    proposal_b: str,
    proposal_c: str,
    *,
    max_queries: int,
) -> str:
    return f"""Architecture question:
{question}

Blind proposal A:
{proposal_a}

Blind proposal B:
{proposal_b}

Blind proposal C:
{proposal_c}

Generate exactly {max_queries} concise searches that bring decision-relevant evidence into the debate. Include queries likely to find papers and mature GitHub repositories when relevant. Prefer evidence that can falsify shared assumptions or support materially different designs. Output only queries, one per line.
"""


def evidence_coverage_prompt(question: str, own_proposal: str, external_evidence: str) -> str:
    taxonomy = ", ".join(EVIDENCE_CATEGORIES)
    return f"""Architecture question:
{question}

Your blind proposal:
{own_proposal}

Shared evidence pack:
{external_evidence}

Evidence taxonomy:
{taxonomy}

Return exactly:
{{
  "gaps": [
    {{"category": "one taxonomy value", "question": "focused research question", "priority": "high|medium|low"}}
  ]
}}
Return {{"gaps": []}} if no important evidence is missing. Maximum two gaps.
"""


def debate_round_prompt(
    question: str,
    round_number: int,
    architect_label: str,
    own_position: str,
    opponent_1_label: str,
    opponent_1_position: str,
    opponent_2_label: str,
    opponent_2_position: str,
    shared_state: str | None = None,
    context: str | None = None,
    external_evidence: str | None = None,
) -> str:
    repository = f"\nRepository evidence:\n{context}\n" if context else ""
    research = (
        f"\nShared external evidence (untrusted; cite [S#]):\n{external_evidence}\n"
        if external_evidence
        else ""
    )
    taxonomy = ", ".join(EVIDENCE_CATEGORIES)
    state = f"\nShared compact debate state:\n{shared_state}\n" if shared_state else ""
    return f"""Architecture question:
{question}

Council debate round: {round_number}
{repository}{research}{state}
Your current position:
{own_position}

{opponent_1_label}'s position:
{opponent_1_position}

{opponent_2_label}'s position:
{opponent_2_position}

Evidence categories: {taxonomy}

Act on specific claim IDs from the shared ledger. Challenge, accept, reject, or supersede
existing claims before introducing new ones. New IDs must use the form
{architect_label}-R{round_number}-C1, {architect_label}-R{round_number}-X1, or
{architect_label}-R{round_number}-E1. Do not silently rewrite another agent's claim.

Return exactly one JSON object:
{{
  "position_markdown": "incremental critique and improvement: accepted points, rejected points, additions, concessions, rebuttals, evidence gaps, and any justified material change; do not replace the discussion with an unrelated architecture",
  "unresolved_objections": [
    {{"target": "Architect A|Architect B|Architect C|shared", "category": "one taxonomy value", "claim": "specific objection", "impact": "high|medium|low", "status": "unresolved|resolved", "evidence": ["S1"]}}
  ],
  "concessions": [
    {{"category": "one taxonomy value", "accepted_from": "Architect A|Architect B|Architect C", "material_position_change": true}}
  ],
  "evidence_requests": [
    {{"category": "one taxonomy value", "question": "focused missing evidence question", "priority": "high|medium|low"}}
  ],
  "claims": [
    {{"claim_id": "{architect_label}-R{round_number}-C1", "statement": "falsifiable architecture claim", "status": "proposed|accepted|rejected|experiment_required", "evidence": ["S1|REPOSITORY"], "supersedes": []}}
  ],
  "challenges": [
    {{"challenge_id": "{architect_label}-R{round_number}-X1", "target_claim_id": "existing claim ID", "reason": "specific falsification or objection", "severity": "high|medium|low", "status": "open|resolved", "evidence_required": "specific evidence or test"}}
  ],
  "proposed_experiments": [
    {{"experiment_id": "{architect_label}-R{round_number}-E1", "target_claim_ids": ["claim ID"], "hypothesis": "falsifiable hypothesis", "test": "bounded test", "metric": "measurement", "success_criterion": "decision threshold"}}
  ],
  "material_architecture_change": true
}}

New designs require a concrete current-design problem plus evidence or an explicit evidence request. Set material_architecture_change=true only for a real architecture change.
"""


def revision_prompt(
    question: str,
    original: str,
    latest_own_position: str,
    shared_debate_state: str,
    context: str,
    external_evidence: str | None = None,
) -> str:
    research = (
        f"\nShared external evidence (cite [S#]):\n{external_evidence}\n"
        if external_evidence
        else ""
    )
    return f"""Architecture question:
{question}

Repository evidence:
{context}
{research}
Original proposal:
{original}

Your final debate state:
{latest_own_position}

Shared compact debate state:
{shared_debate_state}

Produce a final revised architecture with:
- ACCEPTED CHANGES
- REJECTED CRITICISMS
- REVISED ARCHITECTURE
- EVIDENCE-BASED NEW IDEAS AND DERIVATION
- AGENT / WORKFLOW RECOVERY
- EXTERNAL EVIDENCE USED
- REMAINING RISKS
- MINORITY POSITION
- NEEDS EXPERIMENT
Do not manufacture agreement or treat predicted improvements as measured results.
"""


def debate_summary_prompt(state: str, round_number: int) -> str:
    return f"""Summarize debate round {round_number} into compact JSON.
Preserve important reasoning, high-impact objections, evidence requests, concessions,
material changes, minority positions, and the stopping signal. Do not invent facts.
Return only JSON with arrays agreements, unresolved_objections, concessions,
material_changes, evidence_requests, minority_positions and boolean should_continue.
Maximum 5 items per array and 400 characters per item.

{state}"""


def arbiter_score_prompt(
    question: str,
    revision_a: str,
    revision_b: str,
    revision_c: str,
    repository_evidence: str,
    external_evidence: str | None,
    debate_transcript: str,
) -> str:
    evidence = external_evidence or "No external evidence was supplied."
    rubric = arbiter_rubric_text()
    zero_scores = {criterion: 0 for criterion in ARBITER_WEIGHTS}
    empty_justifications = {
        criterion: {"evidence": [], "reason": ""} for criterion in ARBITER_WEIGHTS
    }
    template = json.dumps(
        {
            "candidates": {"A": zero_scores, "B": zero_scores, "C": zero_scores},
            "justifications": {
                "A": empty_justifications,
                "B": empty_justifications,
                "C": empty_justifications,
            },
        },
        indent=2,
    )
    return f"""Architecture question:
{question}

Repository evidence:
{repository_evidence}

External evidence:
{evidence}

Debate transcript:
{debate_transcript}

Candidate A:
{revision_a}

Candidate B:
{revision_b}

Candidate C:
{revision_c}

Weighted rubric:
{rubric}

Return strict JSON matching this shape:
{template}

Replace every score with an integer 0-4. For each criterion, give a short reason and evidence references such as S1 when available; use REPOSITORY or DEBATE for internal evidence. Do not add a winner; code computes weighted totals and tie-breaks deterministically.
"""


def adr_prompt(
    question: str,
    debate_transcript: str,
    revision_a: str,
    revision_b: str,
    revision_c: str,
    score_summary: str,
    winner: str,
    external_evidence: str | None = None,
) -> str:
    research = external_evidence or "No external evidence was supplied."
    return f"""Architecture question:
{question}

Shared external evidence:
{research}

Debate transcript:
{debate_transcript}

Final A:
{revision_a}

Final B:
{revision_b}

Final C:
{revision_c}

Deterministic weighted score summary:
{score_summary}

Code-selected weighted winner: Candidate {winner}

Write the ADR with exactly these top-level sections:
# Executive Decision
# Recommended Architecture
# Weighted Arbitration
# Why
# Decision Matrix
# Evidence-Based New Ideas
# Agent / Workflow Recovery
# External Evidence
# AGREE
# DISAGREE
# MINORITY REPORT
# NEEDS EXPERIMENT
# Risks
# Migration Plan
# Do Not Change
# Revisit Triggers

Use Candidate {winner} as the baseline. You may incorporate compatible strengths from others but may not silently replace the selected baseline. Label predicted correctness improvements UNVERIFIED until measured. NEEDS EXPERIMENT must include hypothesis, test, metric, and success criterion.
"""
