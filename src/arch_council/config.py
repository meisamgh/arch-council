from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    api_key: str
    provider: str = "vyceai"
    base_url: str = "https://vyceai.com"
    seekai_base_url: str = "https://seekai.cc"
    vyceai_base_url: str = "https://vyceai.com"
    agentrouter_base_url: str = "https://agentrouter.org"
    seekai_api_key: str | None = None
    vyceai_api_key: str | None = None
    agentrouter_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_api_key: str | None = None
    truesota_base_url: str = "https://true-sota.com"
    truesota_api_key: str | None = None
    justwoker_base_url: str = "https://api.justwoker.icu"
    justwoker_api_key: str | None = None
    model_a: str = "claude-sonnet-4-6"
    model_b: str = "claude-sonnet-4-6"
    model_c: str = "claude-sonnet-4-6"
    searxng_url: str = "http://localhost:8080"
    tavily_api_key: str | None = None
    timeout_seconds: int = 180

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        provider = os.getenv("ARCH_COUNCIL_PROVIDER", "vyceai").strip().lower()
        api_key = os.getenv("ARCH_COUNCIL_API_KEY", "").strip()
        if not api_key:
            api_key = os.getenv(f"{provider.upper()}_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("API key is missing. Set ARCH_COUNCIL_API_KEY or <PROVIDER>_API_KEY.")
        tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip() or None
        return cls(
            api_key=api_key,
            provider=provider,
            base_url=os.getenv("ARCH_COUNCIL_BASE_URL", "https://vyceai.com").rstrip("/"),
            seekai_base_url=os.getenv("SEEKAI_BASE_URL", "https://seekai.cc").rstrip("/"),
            vyceai_base_url=os.getenv("VYCEAI_BASE_URL", "https://vyceai.com").rstrip("/"),
            agentrouter_base_url=os.getenv("AGENTROUTER_BASE_URL", "https://agentrouter.org").rstrip("/"),
            seekai_api_key=os.getenv("SEEKAI_API_KEY", "").strip() or None,
            vyceai_api_key=os.getenv("VYCEAI_API_KEY", "").strip() or None,
            agentrouter_api_key=os.getenv("AGENTROUTER_API_KEY", "").strip() or None,
            gemini_base_url=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/"),
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip() or None,
            groq_base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/"),
            groq_api_key=os.getenv("GROQ_API_KEY", "").strip() or None,
            truesota_base_url=os.getenv("TRUESOTA_BASE_URL", "https://true-sota.com").rstrip("/"),
            truesota_api_key=os.getenv("TRUESOTA_API_KEY", "").strip() or None,
            justwoker_base_url=os.getenv("JUSTWOKER_BASE_URL", "https://api.justwoker.icu").rstrip("/"),
            justwoker_api_key=os.getenv("JUSTWOKER_API_KEY", "").strip() or None,
            model_a=os.getenv("ARCHITECT_A_MODEL", "gpt-5.6-sol"),
            model_b=os.getenv("ARCHITECT_B_MODEL", "gpt-5.6-sol"),
            model_c=os.getenv("ARCHITECT_C_MODEL", "gpt-5.6-sol"),
            searxng_url=os.getenv("SEARXNG_URL", "http://localhost:8080").rstrip("/"),
            tavily_api_key=tavily_api_key,
            timeout_seconds=int(os.getenv("ARCH_COUNCIL_TIMEOUT_SECONDS", "180")),
        )
