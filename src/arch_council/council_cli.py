from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

from .client import AnthropicGatewayClient, GeminiClient, OpenAIResponsesClient
from .config import Settings
from .context import build_diff_context, build_readme_context, build_repository_context
from .council import Council
from .research import CachedResearchClient, SearXNGResearchClient, TavilyResearchClient
from .research_tools import CouncilResearch
from .runtime.persistence import SQLiteStore


class CouncilRouter:
    """Route explicit aliases; never send another provider's key as a fallback."""
    def __init__(self, settings: Settings, delay: float = 0):
        self.settings, self.delay = settings, delay
        self.clients = {}
        self.before_attempt = None

    def complete(self, *, model: str, **kwargs):
        provider, sep, name = model.partition("/")
        if not sep:
            provider, name = self.settings.provider, model
        if provider not in self.clients:
            key = getattr(self.settings, f"{provider}_api_key", None)
            if not key and provider == self.settings.provider:
                key = self.settings.api_key
            if not key:
                raise ValueError(f"Missing {provider.upper()}_API_KEY")
            base = getattr(self.settings, f"{provider}_base_url", None)
            if base is None:
                raise ValueError(f"Unsupported provider: {provider}")
            if provider == "gemini":
                client = GeminiClient(key, base, self.settings.timeout_seconds)
            elif provider == "truesota":
                client = OpenAIResponsesClient(key, base, self.settings.timeout_seconds)
            else:
                client = AnthropicGatewayClient(key, base, self.settings.timeout_seconds,
                                               provider=provider, request_delay_seconds=self.delay)
            self.clients[provider] = client
        client = self.clients[provider]
        client.before_attempt = self.before_attempt
        return client.complete(model=name, **kwargs)


def run_council(args):
    import fcntl

    args.review_id = args.review_id or uuid.uuid4().hex
    lock_root = Path(args.state_db).resolve().parent
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / (hashlib.sha256(args.review_id.encode()).hexdigest() + ".lock")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("This review already has an active worker; use resume --status-only") from exc
        return _run_council_locked(args)


def _run_council_locked(args):
    # Old saved launch records predate README summarization; preserve their mode.
    for key, default in (("readme_summary", None), ("summary_max_tokens", 1200),
                         ("summarizer_model", "groq/qwen/qwen3.8-27b")):
        if not hasattr(args, key):
            setattr(args, key, default)
    settings = Settings.from_env()
    store = SQLiteStore(args.state_db)
    rid = args.review_id or uuid.uuid4().hex
    args.review_id = rid
    context = ""
    if args.repo:
        if args.readme_only:
            readme = build_readme_context(args.repo, max_chars=args.max_context_chars)
            if readme.truncated:
                raise ValueError("README exceeds --max-context-chars; increase it to include the "
                                 "whole source before summarization")
            context = readme.text
        elif args.diff_base:
            context = build_diff_context(args.repo, args.diff_base, max_chars=args.max_context_chars).text
        else:
            context = build_repository_context(args.repo, max_chars=args.max_context_chars).text
    models = {"A": args.model_a or settings.model_a, "B": args.model_b or settings.model_b}
    if args.agents == 3:
        models["C"] = args.model_c or settings.model_c
    models = {a: m if "/" in m else f"{settings.provider}/{m}" for a, m in models.items()}
    # Persist resolved aliases for future resumes even when environment defaults change.
    args.model_a, args.model_b = models["A"], models["B"]
    args.model_c = models.get("C")
    research = None
    if args.research:
        search = (SearXNGResearchClient(args.searxng_url or settings.searxng_url)
                  if args.research_provider == "searxng"
                  else TavilyResearchClient(settings.tavily_api_key or ""))
        research = CouncilResearch(CachedResearchClient(search, store),
                                   inspections=args.evidence_inspections,
                                   results=args.research_results, store=store, review_id=rid)
    if args.feature_specs or args.oos_plan:
        raise ValueError("Use --mode legacy for the separate feature-spec/OOS validation workflow")
    limits = {"max_calls": args.max_total_calls if args.max_total_calls is not None else 12,
              "max_input_chars": args.max_prompt_chars,
              "max_output_tokens": args.max_output_tokens, "max_queries": args.research_queries,
              "max_total_input_chars": args.max_total_input_chars}
    council = Council(client=CouncilRouter(settings, args.call_delay_seconds), store=store,
                      review_id=rid, topic=args.question, context=context, models=models,
                      rounds=args.rounds, research=research, **limits,
                      decision_model=args.arbiter_model,
                      readme_summary=args.readme_summary if args.readme_only else None,
                      summarizer_model=args.summarizer_model,
                      summary_max_tokens=args.summary_max_tokens,
                      identity={"repo": str(Path(args.repo).resolve()) if args.repo else None,
                                "research_provider": args.research_provider,
                                "searxng_url": args.searxng_url or settings.searxng_url,
                                "inspections": args.evidence_inspections,
                                "results": args.research_results})
    # No credentials stored. The runner separately verifies the immutable manifest.
    previous = store.checkpoint_payload(rid, "launch")
    if previous is None:
        store.checkpoint(rid, "launch", vars(args))
    print(f"Review ID: {rid}\nAgents: {models}\nMaximum rounds: {args.rounds}")
    print(f"Attempt budget: {limits['max_calls']} (includes retries/repairs); "
          f"prompt cap: {args.max_prompt_chars} chars; output cap: {args.max_output_tokens} tokens")
    print("Research: on demand" if research else "Research: disabled")
    print("Debate state: deterministic; README summaries: validated, bounded, cached when enabled")
    print(f"Decision model: {args.arbiter_model or models['B']}; "
          f"README summary: {args.readme_summary if args.readme_only else 'not applicable'}")
    result = council.run()
    store.checkpoint(rid, "launch", vars(args))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Hash the user-selected identifier, never use it as an unchecked path.
    path = output / f"council-{hashlib.sha256(rid.encode()).hexdigest()[:16]}.md"
    content = (f"# Council review\n\n{args.question}\n\nStop: {result['stop_reason']}\n\n"
               f"{result['decision']}\n\n## Audit: claims, dissent, evidence and experiments\n\n"
               f"```json\n{json.dumps(result['state'], indent=2, ensure_ascii=False)}\n```\n\n"
               f"Budget usage: {json.dumps(result['budget'])}\n")
    path.write_text(content, encoding="utf-8")
    store.save_report(rid, {**result, "path": str(path.resolve()), "content": content})
    print(f"Stop: {result['stop_reason']}\nReport: {path}\n{result['decision']}")
    return 2 if result["stop_reason"] == "budget_exhausted" else 0
