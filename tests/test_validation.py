from arch_council.models import ArchitectureDimension
from arch_council.pipeline.validate import (
    FeatureSpec,
    OOSPlan,
    evaluate_feature_safety,
    validate_proposal,
)


def test_feature_safety_flags_target_or_future_features() -> None:
    result = evaluate_feature_safety(
        [FeatureSpec(name="post_outcome", available_at="2025-02", target_available_at="2025-01")],
        OOSPlan(
            split_strategy="chronological",
            train_period="2020-2023",
            validation_period="2024",
            test_period="2025",
        ),
    )
    assert result.safe is False
    assert result.leakage_flags


def test_feature_safety_requires_temporal_oos_controls() -> None:
    result = evaluate_feature_safety(
        [FeatureSpec(name="lagged_signal", available_at="2024-01", target_available_at="2024-02")],
        OOSPlan(
            split_strategy="random",
            train_period="2020-2023",
            validation_period="2024",
            test_period="2025",
            untouched_test=False,
        ),
    )
    assert result.safe is True
    assert result.oos_ready is False
    assert "chronological split" in result.required_controls


def test_proposal_validation_requires_all_fixed_dimensions() -> None:
    payload = {
        "persona": "Sol",
        "summary": "summary",
        "current_architecture": "current",
        "target_architecture": "target",
        "dimensions": {dimension.value: "reason" for dimension in ArchitectureDimension},
    }
    result = validate_proposal(payload)
    assert result.valid is True
    assert result.proposal is not None


def test_proposal_validation_rejects_malformed_payload() -> None:
    result = validate_proposal({"persona": "Sol"})
    assert result.valid is False
    assert result.errors
