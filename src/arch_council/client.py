from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests
from pydantic import BaseModel, ValidationError

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 524}


class LLMError(RuntimeError):
    """Raised when the gateway cannot produce a usable model response."""


@dataclass
class OpenAIResponsesClient:
    """Client for OpenAI Responses-compatible gateways such as True-SOTA."""

    api_key: str
    base_url: str
    timeout_seconds: int = 180
    before_attempt: Callable[[str, str, int], None] | None = None

    def complete(self, *, model: str, system: str, user: str, max_tokens: int = 1000, **_: Any) -> str:
        root = self.base_url.rstrip("/")
        url = f"{root}/v1/responses" if not root.endswith("/v1") else f"{root}/responses"
        if self.before_attempt:
            self.before_attempt(system, user, max_tokens)
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "instructions": system,
                "input": user,
                # Reasoning models may consume hidden reasoning tokens before producing
                # visible text; a 1,000-token cap often yields status=incomplete.
                "max_output_tokens": max_tokens,
            },
            timeout=self.timeout_seconds,
        )
        if not response.ok:
            raise LLMError(f"True-SOTA HTTP {response.status_code}: {response.text[:700]}")
        try:
            data = response.json()
            if isinstance(data.get("output_text"), str) and data["output_text"].strip():
                return data["output_text"].strip()
            for item in data.get("output", []):
                for content in item.get("content", []):
                    text = content.get("text")
                    if isinstance(text, str) and text.strip():
                        return text.strip()
        except (ValueError, TypeError, AttributeError):
            pass
        status = data.get("status") if isinstance(data, dict) else None
        raise LLMError(
            f"True-SOTA Responses returned no final text (status={status}): "
            f"{response.text[:700]}"
        )


@dataclass
class AnthropicGatewayClient:
    api_key: str
    base_url: str
    timeout_seconds: int = 180
    max_retries: int = 2
    provider: str = "anthropic"
    request_delay_seconds: float = 0.0
    _call_number: int = 0
    max_total_calls: int | None = None
    total_attempts: int = 0
    attempt_recorder: Callable[[str, str, str | None], None] | None = None
    before_attempt: Callable[[str, str, int], None] | None = None

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        stage: str = "unknown",
        max_tokens: int = 1000,
        temperature: float = 0.2,
    ) -> str:
        if self._call_number and self.request_delay_seconds > 0:
            time.sleep(self.request_delay_seconds)
        self._call_number += 1
        print(
            f"[LLM] stage={stage} model={model} "
            f"prompt_chars={len(system) + len(user)} max_tokens={max_tokens}",
            flush=True,
        )
        is_anthropic = self.provider in {"anthropic", "claude", "justwoker", "agentrouter_messages"}
        if self.provider == "agentrouter_messages":
            url = f"{self.base_url.rstrip('/')}/anthropic/v1/messages"
        else:
            root = self.base_url.rstrip("/")
            versioned_root = root if root.endswith("/v1") else f"{root}/v1"
            url = f"{versioned_root}/messages" if is_anthropic else f"{versioned_root}/chat/completions"
        if is_anthropic:
            payload = {"model": model, "max_tokens": max_tokens, "temperature": temperature, "system": system, "messages": [{"role": "user", "content": user}]}
        else:
            payload = {"model": model, "max_tokens": max_tokens, "temperature": temperature, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if is_anthropic:
            headers["anthropic-version"] = "2023-06-01"
            # Anthropic-compatible gateways commonly require x-api-key even
            # when they also accept the OpenAI-style Bearer header.
            headers["x-api-key"] = self.api_key

        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            if self.max_total_calls is not None and self.total_attempts >= self.max_total_calls:
                raise LLMError(
                    f"total LLM attempt budget exhausted ({self.max_total_calls})"
                )
            self.total_attempts += 1
            if self.before_attempt:
                self.before_attempt(system, user, max_tokens)
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                last_error = f"network error: {exc}"
                if self.attempt_recorder:
                    self.attempt_recorder("primary", "network_failure", last_error)
                if attempt >= self.max_retries:
                    raise LLMError(last_error) from exc
                time.sleep(min(120.0, (2**attempt) + random.uniform(0, 0.25)))
                continue

            if response.ok:
                try:
                    data = response.json()
                except ValueError as exc:
                    raise LLMError(
                        f"Gateway returned HTTP {response.status_code} but non-JSON content: "
                        f"{response.text[:500]}"
                    ) from exc
                try:
                    result = self._extract_text(data) if is_anthropic else self._extract_openai_text(data)
                except LLMError as exc:
                    if self.attempt_recorder:
                        self.attempt_recorder("primary", "invalid_output", str(exc))
                    raise
                if self.attempt_recorder:
                    self.attempt_recorder("primary", "success", None)
                return result

            request_id = response.headers.get("x-oneapi-request-id", "unknown")
            cf_ray = response.headers.get("cf-ray", "unknown")
            content_type = response.headers.get("content-type", "")
            preview = response.text[:700].replace("\n", " ")
            last_error = (
                f"HTTP {response.status_code}; request_id={request_id}; cf_ray={cf_ray}; "
                f"content_type={content_type}; response={preview}"
            )

            quota_exhausted = response.status_code == 429 and any(
                marker in response.text.lower()
                for marker in ("quota_exceeded", "entitlement exhausted", "insufficient quota", "plan entitlement", "rpm exhausted")
            )

            if quota_exhausted or response.status_code not in RETRYABLE_STATUS_CODES or attempt >= self.max_retries:
                if self.attempt_recorder:
                    outcome = "client_error" if quota_exhausted else ("rate_limited" if response.status_code == 429 else "client_error")
                    self.attempt_recorder("primary", outcome, last_error)
                raise LLMError(last_error)

            retry_after = response.headers.get("retry-after")
            try:
                delay = float(retry_after) if retry_after else float(2**attempt)
            except ValueError:
                delay = float(2**attempt)
            if self.attempt_recorder:
                outcome = "rate_limited" if response.status_code == 429 else "server_failure"
                self.attempt_recorder("primary", outcome, last_error)
            time.sleep(min(delay + random.uniform(0, 0.25), 120.0))

        raise LLMError(last_error or "Unknown gateway failure")

    def complete_structured(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
        max_repair_attempts: int = 1,
    ) -> BaseModel:
        """Return validated JSON; repair malformed model output separately from transport retry."""
        prompt = user
        for repair_attempt in range(max_repair_attempts + 1):
            raw = self.complete(model=model, system=system, user=prompt)
            try:
                return schema.model_validate_json(raw)
            except (ValidationError, ValueError) as exc:
                if self.attempt_recorder:
                    self.attempt_recorder("repair", "invalid_output", str(exc))
                if repair_attempt >= max_repair_attempts:
                    raise LLMError(f"structured output validation failed: {exc}") from exc
                prompt = (
                    f"{user}\n\nYour previous output was invalid. Return only valid JSON matching "
                    f"the required schema. Validation errors:\n{exc}"
                )

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        content = data.get("content")
        if not isinstance(content, list):
            raise LLMError(f"Unexpected Anthropic response shape: {data}")

        text_parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        text = "\n".join(part for part in text_parts if part).strip()
        if not text:
            raise LLMError(f"No text content returned by model: {data}")
        return text

    @staticmethod
    def _extract_openai_text(data: dict[str, Any]) -> str:
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected OpenAI-compatible response shape: {data}") from exc
        if not isinstance(text, str) or not text.strip():
            raise LLMError(f"No text content returned by model: {data}")
        return text.strip()


class ProviderRouterClient:
    """Route logical model aliases to provider-specific clients."""

    def __init__(self, routes: dict[str, tuple[AnthropicGatewayClient, str]], max_total_calls: int | None = None) -> None:
        self.routes = routes
        self.max_total_calls = max_total_calls
        self.total_calls = 0

    def complete(self, *, model: str, system: str, user: str, **kwargs: Any) -> str:
        if self.max_total_calls is not None and self.total_calls >= self.max_total_calls:
            raise LLMError(f"review call budget exhausted ({self.max_total_calls})")
        self.total_calls += 1
        try:
            client, provider_model = self.routes[model]
        except KeyError as exc:
            raise LLMError(f"No provider route configured for model alias: {model}") from exc
        return client.complete(model=provider_model, system=system, user=user, **kwargs)

    def complete_structured(self, *, model: str, system: str, user: str, schema: type[BaseModel], **kwargs: Any) -> BaseModel:
        if self.max_total_calls is not None and self.total_calls >= self.max_total_calls:
            raise LLMError(f"review call budget exhausted ({self.max_total_calls})")
        self.total_calls += 1
        try:
            client, provider_model = self.routes[model]
        except KeyError as exc:
            raise LLMError(f"No provider route configured for model alias: {model}") from exc
        return client.complete_structured(model=provider_model, system=system, user=user, schema=schema, **kwargs)


@dataclass
class GeminiClient:
    api_key: str
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    timeout_seconds: int = 180
    before_attempt: Callable[[str, str, int], None] | None = None

    def complete(self, *, model: str, system: str, user: str, max_tokens: int = 1000, temperature: float = 0.2, **_: Any) -> str:
        if self.before_attempt:
            self.before_attempt(system, user, max_tokens)
        response = requests.post(
            f"{self.base_url.rstrip('/')}/models/{model}:generateContent",
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json={"systemInstruction": {"parts": [{"text": system}]}, "contents": [{"role": "user", "parts": [{"text": user}]}], "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature}},
            timeout=self.timeout_seconds,
        )
        if not response.ok:
            raise LLMError(f"Gemini HTTP {response.status_code}: {response.text[:500]}")
        try:
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected Gemini response shape: {response.text[:500]}") from exc
