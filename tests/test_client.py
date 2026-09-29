import pytest
from pydantic import BaseModel

from arch_council.client import AnthropicGatewayClient, LLMError, ProviderRouterClient


def test_extract_text_from_anthropic_response() -> None:
    data = {
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]
    }
    assert AnthropicGatewayClient._extract_text(data) == "hello\nworld"


def test_extract_text_rejects_missing_content() -> None:
    with pytest.raises(LLMError):
        AnthropicGatewayClient._extract_text({"content": []})


def test_extracts_openai_compatible_response() -> None:
    assert AnthropicGatewayClient._extract_openai_text(
        {"choices": [{"message": {"content": "hello"}}]}
    ) == "hello"


def test_router_enforces_shared_call_budget() -> None:
    class FakeClient:
        def complete(self, **kwargs: object) -> str:
            return "ok"

    router = ProviderRouterClient({"a": (FakeClient(), "model")}, max_total_calls=1)  # type: ignore[arg-type]
    assert router.complete(model="a", system="", user="") == "ok"
    with pytest.raises(LLMError, match="budget exhausted"):
        router.complete(model="a", system="", user="")


def test_structured_output_is_repaired() -> None:
    class Output(BaseModel):
        answer: str

    class RepairClient(AnthropicGatewayClient):
        def __init__(self) -> None:
            super().__init__("key", "https://example.com")
            self.responses = ["{bad", '{"answer":"ok"}']

        def complete(self, **kwargs: object) -> str:
            return self.responses.pop(0)

    result = RepairClient().complete_structured(
        model="model", system="system", user="return JSON", schema=Output
    )
    assert result.answer == "ok"
