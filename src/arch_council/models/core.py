from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ArchitectureDimension(StrEnum):
    BOUNDARIES = "boundaries"
    DATA_FLOW = "data_flow"
    RELIABILITY = "reliability"
    SECURITY = "security"
    SCALABILITY = "scalability"
    OBSERVABILITY = "observability"
    COST = "cost"
    OPERABILITY = "operability"
    MIGRATION = "migration"


class ClaimStatus(StrEnum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPERIMENT_REQUIRED = "experiment_required"


class ChallengeStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"


class CouncilClaim(BaseModel):
    model_config = ConfigDict(frozen=True)
    claim_id: str = Field(min_length=3, max_length=80)
    author: str = Field(min_length=1, max_length=40)
    statement: str = Field(min_length=8, max_length=700)
    status: ClaimStatus = ClaimStatus.PROPOSED
    evidence: list[str] = Field(default_factory=list, max_length=8)
    supersedes: list[str] = Field(default_factory=list, max_length=5)


class CouncilChallenge(BaseModel):
    model_config = ConfigDict(frozen=True)
    challenge_id: str = Field(min_length=3, max_length=80)
    author: str = Field(min_length=1, max_length=40)
    target_claim_id: str = Field(min_length=3, max_length=80)
    reason: str = Field(min_length=8, max_length=700)
    severity: str = Field(pattern="^(high|medium|low)$")
    status: ChallengeStatus = ChallengeStatus.OPEN
    evidence_required: str = Field(default="", max_length=500)


class CouncilExperiment(BaseModel):
    model_config = ConfigDict(frozen=True)
    experiment_id: str = Field(min_length=3, max_length=80)
    author: str = Field(min_length=1, max_length=40)
    target_claim_ids: list[str] = Field(min_length=1, max_length=8)
    hypothesis: str = Field(min_length=8, max_length=700)
    test: str = Field(min_length=8, max_length=700)
    metric: str = Field(min_length=2, max_length=300)
    success_criterion: str = Field(min_length=3, max_length=500)


class ClaimLedger(BaseModel):
    model_config = ConfigDict(frozen=True)
    claims: list[CouncilClaim] = Field(default_factory=list, max_length=24)
    challenges: list[CouncilChallenge] = Field(default_factory=list, max_length=24)
    experiments: list[CouncilExperiment] = Field(default_factory=list, max_length=16)


class ProjectProfile(BaseModel):
    model_config = ConfigDict(frozen=True)
    profile_id: str
    profile_hash: str
    repository: str
    readme: str
    extracted_facts: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class ArchitectureProposal(BaseModel):
    persona: str
    summary: str
    current_architecture: str
    target_architecture: str
    dimensions: dict[ArchitectureDimension, str]
    risks: list[str] = Field(default_factory=list)
    experiments: list[str] = Field(default_factory=list)
    dissent: list[str] = Field(default_factory=list)


class RoundSummary(BaseModel):
    round_number: int
    confirmed_facts: list[str] = Field(default_factory=list)
    agreements: list[str] = Field(default_factory=list)
    disagreements: dict[ArchitectureDimension, list[str]] = Field(default_factory=dict)
    evidence_needed: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    minority_reasoning: list[str] = Field(default_factory=list)
    current_recommendation: str = ""


class EvidenceCard(BaseModel):
    source_id: str
    source_type: str
    title: str
    url: str
    claim: str
    evidence: str
    limitations: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
