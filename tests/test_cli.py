import pytest

from arch_council.cli import _build_parser


def test_review_defaults() -> None:
    args = _build_parser().parse_args(
        ["review", "--repo", ".", "--question", "What architecture should we use?"]
    )
    assert args.rounds == 2
    assert args.research is False
    assert args.research_provider == "searxng"
    assert args.research_queries == 3
    assert args.research_results == 2
    assert args.evidence_inspections == 4
    assert args.arbiter_model is None


def test_background_review_flag_and_review_id() -> None:
    args = _build_parser().parse_args(
        ["review", "--repo", ".", "--question", "Question", "--background", "--review-id", "r1"]
    )
    assert args.background is True
    assert args.review_id == "r1"


def test_feature_validation_inputs_are_available() -> None:
    args = _build_parser().parse_args(
        [
            "review",
            "--repo",
            ".",
            "--question",
            "Question",
            "--feature-specs",
            "features.json",
            "--oos-plan",
            "oos.json",
        ]
    )
    assert args.feature_specs == "features.json"
    assert args.oos_plan == "oos.json"


def test_chat_command_is_available() -> None:
    args = _build_parser().parse_args(["chat", "--repo", ".", "--readme-only"])
    assert args.command == "chat"
    assert args.readme_only is True


def test_review_call_delay_defaults_to_zero_and_is_configurable() -> None:
    parser = _build_parser()
    base = ["review", "--repo", ".", "--question", "Question"]
    assert parser.parse_args(base).call_delay_seconds == 0.0
    assert parser.parse_args(base + ["--call-delay-seconds", "3"]).call_delay_seconds == 3.0


def test_chat_tool_budget_can_be_overridden() -> None:
    args = _build_parser().parse_args(
        ["chat", "--repo", ".", "--research", "--max-tool-steps", "4"]
    )
    assert args.research is True
    assert args.max_tool_steps == 4


def test_rounds_accepts_one_through_three() -> None:
    parser = _build_parser()
    for rounds in range(1, 4):
        args = parser.parse_args(
            ["review", "--repo", ".", "--question", "Question", "--rounds", str(rounds)]
        )
        assert args.rounds == rounds


def test_rounds_rejects_four() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            ["review", "--repo", ".", "--question", "Question", "--rounds", "4"]
        )


def test_arbiter_and_evidence_budget_can_be_overridden() -> None:
    args = _build_parser().parse_args(
        [
            "review",
            "--repo",
            ".",
            "--question",
            "Question",
            "--arbiter-model",
            "judge-model",
            "--evidence-inspections",
            "6",
        ]
    )
    assert args.arbiter_model == "judge-model"
    assert args.evidence_inspections == 6


def test_research_and_readme_only_flags_are_available() -> None:
    args = _build_parser().parse_args(
        [
            "review",
            "--repo",
            ".",
            "--question",
            "Question",
            "--readme-only",
            "--research",
            "--research-provider",
            "searxng",
            "--searxng-url",
            "http://127.0.0.1:8888",
            "--research-queries",
            "5",
            "--research-results",
            "4",
        ]
    )
    assert args.readme_only is True
    assert args.research is True
    assert args.research_provider == "searxng"
    assert args.searxng_url == "http://127.0.0.1:8888"
    assert args.research_queries == 5
    assert args.research_results == 4


def test_tavily_remains_an_optional_provider() -> None:
    args = _build_parser().parse_args(
        [
            "review",
            "--repo",
            ".",
            "--question",
            "Question",
            "--research",
            "--research-provider",
            "tavily",
        ]
    )
    assert args.research_provider == "tavily"


def test_readme_only_and_diff_base_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            [
                "review",
                "--repo",
                ".",
                "--question",
                "Question",
                "--readme-only",
                "--diff-base",
                "main",
            ]
        )
