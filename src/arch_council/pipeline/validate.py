from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from ..models import ArchitectureDimension, ArchitectureProposal


class FeatureSpec(BaseModel):
    name: str
    available_at: str
    target_available_at: str
    source: str = ""
    transformation: str = ""


class OOSPlan(BaseModel):
    split_strategy: str
    train_period: str
    validation_period: str
    test_period: str
    untouched_test: bool = True
    leakage_controls: list[str] = Field(default_factory=list)


class FeatureSafetyResult(BaseModel):
    safe: bool
    leakage_flags: list[str] = Field(default_factory=list)
    oos_ready: bool
    required_controls: list[str] = Field(default_factory=list)


class ProposalValidationResult(BaseModel):
    valid: bool
    proposal: ArchitectureProposal | None = None
    errors: list[str] = Field(default_factory=list)


def validate_proposal(payload: object) -> ProposalValidationResult:
    try:
        proposal = ArchitectureProposal.model_validate(payload)
    except ValidationError as exc:
        return ProposalValidationResult(valid=False, errors=[str(exc)])
    missing = [dimension.value for dimension in ArchitectureDimension if dimension not in proposal.dimensions]
    if missing:
        return ProposalValidationResult(
            valid=False,
            proposal=proposal,
            errors=[f"missing architecture dimensions: {', '.join(missing)}"],
        )
    return ProposalValidationResult(valid=True, proposal=proposal)


def evaluate_feature_safety(features: list[FeatureSpec], plan: OOSPlan) -> FeatureSafetyResult:
    flags = [
        f"{feature.name}: feature timestamp is not before target timestamp"
        for feature in features
        if feature.available_at >= feature.target_available_at
    ]
    controls = list(plan.leakage_controls)
    required = ["chronological split", "untouched final test set", "fit transforms on train only"]
    for item in required:
        if not any(item.lower() in control.lower() for control in controls):
            controls.append(item)
    oos_ready = plan.untouched_test and plan.split_strategy.lower() in {
        "chronological",
        "time_series",
        "purged_time_series",
    }
    return FeatureSafetyResult(
        safe=not flags,
        leakage_flags=flags,
        oos_ready=oos_ready,
        required_controls=controls,
    )
