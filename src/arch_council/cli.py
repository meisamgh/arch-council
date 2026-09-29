from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from dataclasses import replace
from pathlib import Path

from .agent import ArchitectureAgent
from .client import (
    AnthropicGatewayClient,
    GeminiClient,
    LLMError,
    OpenAIResponsesClient,
    ProviderRouterClient,
)
from .config import Settings
from .context import build_diff_context, build_readme_context, build_repository_context
from .debate import ArchitectureDebate, write_report
from .pipeline.validate import FeatureSpec, OOSPlan, evaluate_feature_safety
from .research import (
    CachedResearchClient,
    ResearchError,
    SearXNGResearchClient,
    TavilyResearchClient,
)
from .runtime.persistence import SQLiteStore


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arch-council",
        description="Run a bounded evidence-grounded architecture council over a local repository.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create an immutable README project profile")
    init.add_argument("--repo", required=True)
    init.add_argument("--state-db", default=".arch-council/state.sqlite3")
    resume = subparsers.add_parser("resume", help="Continue a council review, or inspect legacy checkpoints")
    resume.add_argument("--review-id", required=True)
    resume.add_argument("--state-db", default=".arch-council/state.sqlite3")
    resume.add_argument("--status-only", action="store_true", help="Inspect without continuing")
    resume.add_argument("--max-total-calls", type=int, help="Increase persisted attempt allowance")
    resume.add_argument("--max-total-input-chars", type=int)
    history = subparsers.add_parser("history", help="List persisted review reports")
    history.add_argument("--state-db", default=".arch-council/state.sqlite3")

    review = subparsers.add_parser("review", help="Review a repository architecture question")
    review.add_argument("--repo", help="Optional local repository; omit for topic-only discussion")
    review.add_argument("--question", "--topic", required=True,
                        help="Topic to discuss (council: up to 12000 characters; prompt cap still applies)")
    review.add_argument("--mode", choices=("council", "legacy"), default="council")
    review.add_argument("--readme-summary", choices=("auto", "always", "never"),
                        default="auto", help="README-only: summarize once above 8000 chars")
    review.add_argument("--summary-max-tokens", type=int, default=1200,
                        help="README summary output cap, 256-2000 tokens")
    review.add_argument("--agents", type=int, choices=(2, 3), default=2)
    review.add_argument("--max-prompt-chars", type=int, default=16000)
    review.add_argument("--max-output-tokens", type=int, default=1200)
    review.add_argument("--max-total-input-chars", type=int, default=160000)
    review.add_argument("--model-a", help="Override Architect A model")
    review.add_argument("--model-b", help="Override Architect B model")
    review.add_argument("--model-c", help="Override Architect C model")
    review.add_argument(
        "--arbiter-model",
        help="Final decision model; legacy mode also uses it for weighted scoring",
    )
    review.add_argument(
        "--rounds",
        type=int,
        choices=range(1, 4),
        default=2,
        metavar="1-3",
        help="Maximum debate rounds; deterministic convergence may stop earlier (default: 2)",
    )
    context_group = review.add_mutually_exclusive_group()
    context_group.add_argument(
        "--diff-base",
        help="Use git diff BASE...HEAD instead of broad repo context",
    )
    context_group.add_argument(
        "--readme-only",
        action="store_true",
        help="Share only the root README with the council; no source files or repo tree",
    )
    review.add_argument(
        "--research",
        action="store_true",
        help="Allow research on demand when agents request evidence",
    )
    review.add_argument(
        "--research-provider",
        choices=("searxng", "tavily"),
        default="searxng",
        help="External search provider used with --research (default: searxng)",
    )
    review.add_argument(
        "--searxng-url",
        help="SearXNG base URL; defaults to SEARXNG_URL or http://localhost:8080",
    )
    review.add_argument(
        "--research-queries",
        type=int,
        choices=range(1, 7),
        default=3,
        metavar="1-6",
        help="Council-wide query budget (default: 3); initial query count in legacy mode",
    )
    review.add_argument(
        "--research-results",
        type=int,
        choices=range(1, 6),
        default=2,
        metavar="1-5",
        help="Maximum search results per research query (default: 2)",
    )
    review.add_argument(
        "--evidence-inspections",
        type=int,
        choices=range(9),
        default=4,
        metavar="0-8",
        help="Council-wide source inspection budget (default: 4); per pass in legacy mode",
    )
    review.add_argument(
        "--max-context-chars",
        type=int,
        default=20_000,
        help="Maximum repository/diff/README characters sent to each architect (default: 20000)",
    )
    review.add_argument(
        "--output-dir",
        default="reports",
        help="Directory for generated Markdown reports",
    )
    review.add_argument("--review-id", help="Stable id used for checkpoints and resume")
    review.add_argument("--state-db", default=".arch-council/state.sqlite3")
    review.add_argument(
        "--background",
        action="store_true",
        help="Run the review in a detached worker and return its review ID",
    )
    review.add_argument("--feature-specs", help="JSON file containing feature specifications")
    review.add_argument("--oos-plan", help="JSON file containing an OOS validation plan")
    review.add_argument("--max-total-calls", type=int, help="Shared review-wide LLM call budget")
    review.add_argument(
        "--summarizer-model",
        default="groq/qwen/qwen3.8-27b",
        help="README summarizer model; also used for legacy debate summaries",
    )
    review.add_argument("--disable-summarizer", action="store_true")
    review.add_argument(
        "--call-delay-seconds",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Optional delay between review LLM calls; does not affect chat (default: 0)",
    )

    chat = subparsers.add_parser("chat", help="Run an interactive architecture agent")
    chat.add_argument("--repo", required=True, help="Path to the local repository")
    chat.add_argument("--model", help="Model used by the architecture agent")
    chat.add_argument("--readme-only", action="store_true", help="Use only the root README")
    chat.add_argument(
        "--research",
        action="store_true",
        help="Give the agent an external web-search tool",
    )
    chat.add_argument("--research-provider", choices=("searxng", "tavily"), default="searxng")
    chat.add_argument("--searxng-url", help="SearXNG URL; defaults to SEARXNG_URL")
    chat.add_argument(
        "--max-tool-steps",
        type=int,
        choices=range(6),
        default=2,
        metavar="0-5",
        help="Maximum search actions per question before forced synthesis (default: 2)",
    )
    chat.add_argument("--max-context-chars", type=int, default=20_000)
    return parser


def _run_chat(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    context = (build_readme_context if args.readme_only else build_repository_context)(
        args.repo, max_chars=args.max_context_chars
    )
    research_client = None
    research_label = "disabled"
    if args.research:
        if args.research_provider == "searxng":
            searxng_url = args.searxng_url or settings.searxng_url
            research_client = SearXNGResearchClient(searxng_url)
            research_label = f"searxng ({searxng_url})"
        elif not settings.tavily_api_key:
            raise RuntimeError("chat --research --research-provider tavily requires TAVILY_API_KEY")
        else:
            research_client = TavilyResearchClient(settings.tavily_api_key)
            research_label = "tavily"

    agent = ArchitectureAgent(
        AnthropicGatewayClient(settings.api_key, settings.base_url, settings.timeout_seconds, provider=settings.provider),
        model=args.model or settings.model_a,
        context=context,
        research_client=research_client,
        max_tool_steps=args.max_tool_steps,
    )
    print(f"Architecture agent for {context.root}. Type 'exit' or Ctrl-D to stop.")
    print(f"External search tool: {research_label}")
    if research_client is not None:
        print(
            f"Agent budget: up to {args.max_tool_steps} search action(s) + final synthesis; "
            f"at most {args.max_tool_steps + 1} LLM calls/question."
        )
    else:
        print("Agent budget: 1 LLM call/question; external search tool is disabled.")

    while True:
        try:
            question = input("\nYou: ")
        except EOFError:
            print()
            break
        if not question.strip():
            continue
        if question.strip().lower() in {"exit", "quit"}:
            break
        try:
            answer = agent.ask(question)
            for step in agent.last_trace:
                if step.action == "search":
                    print(f"[agent step {step.step}] SEARCH: {step.detail}")
                elif step.action == "search_error":
                    print(f"[agent step {step.step}] SEARCH FAILED: {step.detail}")
            print(f"\nArchCouncil: {answer}")
        except (LLMError, ValueError) as exc:
            print(f"Chat error: {exc}", file=sys.stderr)
            return 2
    return 0


def _run_init(args: argparse.Namespace) -> int:
    context = build_readme_context(args.repo)
    profile = SQLiteStore(args.state_db).save_profile(str(context.root), context.text)
    print(f"Project profile created: {profile.profile_id}")
    print(f"Profile hash: {profile.profile_hash}")
    return 0


def _run_resume(args: argparse.Namespace) -> int:
    store = SQLiteStore(args.state_db)
    launch = store.checkpoint_payload(args.review_id, "launch")
    if launch and not args.status_only:
        from .council_cli import run_council
        launch["state_db"] = args.state_db
        for key in ("max_total_calls", "max_total_input_chars"):
            if getattr(args, key, None) is not None:
                launch[key] = getattr(args, key)
        return run_council(argparse.Namespace(**launch))
    stages = ("proposals", "research", "summary", "decision")
    found = False
    for stage in stages:
        payload = store.checkpoint_payload(args.review_id, stage)
        if payload is not None:
            print(f"{stage}: complete")
            found = True
    for event in store.events(args.review_id):
        print(f"{event['created_at']} {event['stage']}: {event['event']}")
    if not found:
        print(f"No checkpoints found for review {args.review_id}")
        return 1
    return 0


def _run_history(args: argparse.Namespace) -> int:
    reports = SQLiteStore(args.state_db).report_history()
    if not reports:
        print("No persisted reports")
        return 1
    for report in reports:
        print(f"{report['review_id']}: {report.get('path', 'report unavailable')}")
    return 0


def _run_review(args: argparse.Namespace) -> int:
    if args.mode == "council" and not args.background:
        from .council_cli import run_council
        return run_council(args)
    if args.mode == "legacy" and not args.repo:
        raise ValueError("legacy mode requires --repo")
    settings = Settings.from_env()
    review_id = args.review_id or uuid.uuid4().hex
    if args.background:
        command = [sys.executable, "-m", "arch_council", "review"]
        skip = {"background", "command", "review_id"}
        for key, value in vars(args).items():
            if key in skip or value is None or value is False:
                continue
            option = "--" + key.replace("_", "-")
            command.append(option)
            if value is not True:
                command.append(str(value))
        command.extend(["--review-id", review_id, "--state-db", args.state_db])
        subprocess.Popen(
            command,
            cwd=os.getcwd(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print(f"Review queued in background. Review id: {review_id}")
        print(f"Monitor with: arch-council resume --review-id {review_id} --state-db {args.state_db}")
        return 0
    store = SQLiteStore(args.state_db)
    feature_safety = None
    if args.feature_specs or args.oos_plan:
        if not args.feature_specs or not args.oos_plan:
            raise RuntimeError("--feature-specs and --oos-plan must be provided together")
        features = [
            FeatureSpec.model_validate(item)
            for item in json.loads(Path(args.feature_specs).read_text(encoding="utf-8"))
        ]
        plan = OOSPlan.model_validate(
            json.loads(Path(args.oos_plan).read_text(encoding="utf-8"))
        )
        feature_safety = evaluate_feature_safety(features, plan)
    model_a = args.model_a or "justwoker/claude-opus-4-8"
    model_b = args.model_b or "justwoker/claude-opus-4-8"
    model_c = args.model_c or "justwoker/claude-opus-4-8"
    arbiter_model = args.arbiter_model or model_b

    if args.readme_only:
        context = build_readme_context(args.repo, max_chars=args.max_context_chars)
        if context.truncated:
            raise ValueError("README exceeds --max-context-chars; increase it before summarization")
    elif args.diff_base:
        context = build_diff_context(args.repo, args.diff_base, max_chars=args.max_context_chars)
    else:
        context = build_repository_context(args.repo, max_chars=args.max_context_chars)

    profile = store.save_profile(str(context.root), context.text)
    pinned_profile = store.checkpoint_payload(review_id, "profile")
    if isinstance(pinned_profile, dict) and pinned_profile.get("profile_hash") != profile.profile_hash:
        raise RuntimeError(
            "review profile changed since the checkpoint; use a new --review-id or restart explicitly"
        )
    store.checkpoint(
        review_id,
        "profile",
        {"profile_id": profile.profile_id, "profile_hash": profile.profile_hash},
    )

    research_client = None
    research_label = "disabled"
    if args.research:
        if args.research_provider == "searxng":
            searxng_url = args.searxng_url or settings.searxng_url
            research_client = SearXNGResearchClient(searxng_url)
            research_label = f"searxng ({searxng_url})"
        else:
            if not settings.tavily_api_key:
                raise RuntimeError(
                    "--research-provider tavily requires TAVILY_API_KEY. "
                    "Use the default SearXNG provider for keyless local research."
                )
            research_client = TavilyResearchClient(settings.tavily_api_key)
            research_label = "tavily"

        research_client = CachedResearchClient(research_client, store)

    # Max budget: proposals + optional planner/coverage + debate + revisions + scoring + ADR.
    max_llm_calls = 8 + (3 * args.rounds) + args.rounds + (4 if args.research else 0)
    print(f"Repository: {context.root}")
    print(f"Context mode: {context.mode}")
    print(f"Included files: {len(context.included_files)}")
    if context.truncated:
        print("Warning: context was truncated to the configured limit.")
    print(f"Architect A: {model_a} (Production Pragmatist)")
    print(f"Architect B: {model_b} (Scaling Challenger)")
    print(f"Architect C: {model_c} (Alternative/Mutation Architect)")
    print(f"Arbiter: {arbiter_model} (fresh impartial scoring role)")
    print(f"Maximum debate rounds: {args.rounds}; may stop earlier on deterministic convergence")
    print(f"External research: {research_label}")
    if args.research:
        print(
            f"Research budget: {args.research_queries} initial queries × up to "
            f"{args.research_results} results/query; deep inspections/pass: "
            f"{args.evidence_inspections}"
        )
    print(f"Maximum planned LLM calls: {max_llm_calls}")
    print(
        "Flow: 3 blind proposals → shared evidence → evidence coverage check → "
        "dynamic debate → 3 final revisions → weighted arbiter score → ADR"
    )

    recorder = lambda call_type, outcome, error: store.record_attempt(review_id, call_type, outcome, error)
    seekai = AnthropicGatewayClient(settings.seekai_api_key or settings.api_key, settings.seekai_base_url, settings.timeout_seconds, provider="seekai", request_delay_seconds=args.call_delay_seconds, attempt_recorder=recorder)
    vyceai = AnthropicGatewayClient(settings.vyceai_api_key or settings.api_key, settings.vyceai_base_url, settings.timeout_seconds, provider="vyceai", request_delay_seconds=args.call_delay_seconds, attempt_recorder=recorder)
    agentrouter = AnthropicGatewayClient(settings.agentrouter_api_key or settings.api_key, settings.agentrouter_base_url, settings.timeout_seconds, provider="agentrouter", request_delay_seconds=args.call_delay_seconds, attempt_recorder=recorder)
    agentrouter_messages = AnthropicGatewayClient(settings.agentrouter_api_key or settings.api_key, settings.agentrouter_base_url, settings.timeout_seconds, provider="agentrouter_messages", request_delay_seconds=args.call_delay_seconds, attempt_recorder=recorder)
    gemini = GeminiClient(settings.gemini_api_key or settings.api_key, settings.gemini_base_url, settings.timeout_seconds)
    groq = AnthropicGatewayClient(settings.groq_api_key or settings.api_key, settings.groq_base_url, settings.timeout_seconds, provider="groq", request_delay_seconds=args.call_delay_seconds, attempt_recorder=recorder)
    truesota = OpenAIResponsesClient(settings.truesota_api_key or settings.api_key, settings.truesota_base_url, settings.timeout_seconds)
    justwoker = AnthropicGatewayClient(settings.justwoker_api_key or settings.api_key, settings.justwoker_base_url, settings.timeout_seconds, provider="justwoker", request_delay_seconds=args.call_delay_seconds, attempt_recorder=recorder)
    client = ProviderRouterClient({
        "seekai/gpt-5.6-sol": (seekai, "gpt-5.6-sol"),
        "seekai/glm-5.2": (seekai, "glm-5.2"),
        "vyceai/claude-sonnet-4-6": (vyceai, "claude-sonnet-4-6"),
        "agentrouter/gpt-5.6-sol": (agentrouter, "gpt-5.6-sol"),
        "agentrouter/qwen/qwen3.8-27b": (agentrouter, "qwen/qwen3.8-27b"),
        "agentrouter/qwen/qwen3.6-27b": (agentrouter, "qwen/qwen3.6-27b"),
        "agentrouter/claude-sonnet-4-6": (agentrouter, "claude-sonnet-4-6"),
        "agentrouter-messages/claude-sonnet-4-6": (agentrouter_messages, "claude-sonnet-4-6"),
        "gemini/gemini-3.7-flash": (gemini, "gemini-3.7-flash"),
        "groq/qwen/qwen3.8-27b": (groq, "qwen/qwen3.8-27b"),
        "groq/qwen/qwen3.6-27b": (groq, "qwen/qwen3.6-27b"),
        "truesota/gpt-5.5": (truesota, "gpt-5.5"),
        "truesota/claude-opus-5": (truesota, "claude-opus-5"),
        "justwoker/claude-opus-4-8": (justwoker, "claude-opus-4-8"),
    }, max_total_calls=args.max_total_calls)
    debate = ArchitectureDebate(
        client,
        model_a=model_a,
        model_b=model_b,
        model_c=model_c,
        arbiter_model=arbiter_model,
        rounds=args.rounds,
        research_client=research_client,
        research_query_count=args.research_queries,
        research_results_per_query=args.research_results,
        evidence_inspection_limit=args.evidence_inspections,
        checkpoint=lambda stage, payload: store.checkpoint(review_id, stage, payload),
        load_checkpoint=lambda stage: store.checkpoint_payload(review_id, stage),
        feature_safety=feature_safety,
        summarizer_model=None if args.disable_summarizer else args.summarizer_model,
    )

    try:
        if args.readme_only:
            from .council_cli import CouncilRouter
            from .readme_summary import ReadmeSummarizer

            summary_config = [args.readme_summary, args.summarizer_model,
                              args.summary_max_tokens]
            previous_config = store.checkpoint_payload(review_id, "readme-summary-config")
            if previous_config is not None and previous_config != summary_config:
                raise ValueError("README summary configuration changed; use a new review ID")
            store.checkpoint(review_id, "readme-summary-config", summary_config)
            # Reuse the legacy review's call allowance, including summary/repair calls.
            client.routes[args.summarizer_model] = (
                CouncilRouter(settings, args.call_delay_seconds), args.summarizer_model,
            )
            context = replace(context, text=ReadmeSummarizer(
                client, store, args.summarizer_model, args.summary_max_tokens,
            ).prepare(context.text, args.readme_summary))
        result = debate.run(question=args.question, context=context)
    except (LLMError, ResearchError, ValueError) as exc:
        store.event(review_id, "review", "failed", {"error": str(exc)})
        print(f"Review error: {exc}", file=sys.stderr)
        return 2

    report = write_report(result, context, Path(args.output_dir))
    store.checkpoint(review_id, "report", {"path": str(report), "rounds_completed": result.rounds_completed})
    store.save_report(
        review_id,
        {
            "path": str(report),
            "rounds_completed": result.rounds_completed,
            "content": report.read_text(encoding="utf-8"),
        },
    )
    print(f"Review id: {review_id}")
    print(f"\nRounds completed: {result.rounds_completed}/{result.rounds_requested}")
    print(f"Weighted winner: Candidate {result.arbiter_scorecard.winner}")
    print(f"Report written to: {report}")
    print("\n=== FINAL ADR ===\n")
    print(result.decision)
    return 0


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "init":
            raise SystemExit(_run_init(args))
        if args.command == "resume":
            raise SystemExit(_run_resume(args))
        if args.command == "history":
            raise SystemExit(_run_history(args))
        if args.command == "review":
            raise SystemExit(_run_review(args))
        if args.command == "chat":
            raise SystemExit(_run_chat(args))
        parser.error(f"Unknown command: {args.command}")
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
