from pathlib import Path

from arch_council.agent import ArchitectureAgent
from arch_council.context import RepositoryContext
from arch_council.research import ResearchPack, ResearchSource


class SequenceClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeResearchClient:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search_many(self, queries, **kwargs):
        query = next(iter(queries))
        self.queries.append(query)
        return ResearchPack(
            queries=(query,),
            sources=(
                ResearchSource(
                    source_id="S1",
                    query=query,
                    title="Checkpointing guide",
                    url="https://example.com/checkpointing",
                    content="Durable checkpoints allow workflows to resume after failure.",
                    score=0.9,
                ),
            ),
        )


def _context() -> RepositoryContext:
    return RepositoryContext(
        root=Path("."),
        text="README evidence",
        included_files=(),
        truncated=False,
        mode="readme",
    )


def test_agent_preserves_conversation_history_without_tools() -> None:
    client = SequenceClient(["first answer", "second answer"])
    agent = ArchitectureAgent(client, "model", _context())

    assert agent.ask("What is the current boundary?") == "first answer"
    assert agent.ask("What should change?") == "second answer"

    assert len(agent.history) == 4
    assert "What is the current boundary?" in str(client.calls[1]["user"])


def test_agent_can_choose_search_then_answer() -> None:
    client = SequenceClient(
        [
            '{"action":"search","query":"workflow checkpoint recovery"}',
            '{"action":"answer","answer":"Use durable checkpoints [R1.1]."}',
        ]
    )
    research = FakeResearchClient()
    agent = ArchitectureAgent(
        client,
        "model",
        _context(),
        research_client=research,
        max_tool_steps=2,
    )

    answer = agent.ask("How should recovery work?")

    assert answer == "Use durable checkpoints [R1.1]."
    assert research.queries == ["workflow checkpoint recovery"]
    assert len(client.calls) == 2
    assert agent.last_trace[0].action == "search"
    assert "[R1.1]" in str(client.calls[1]["user"])


def test_agent_forces_final_synthesis_after_tool_budget() -> None:
    client = SequenceClient(
        [
            '{"action":"search","query":"checkpoint recovery"}',
            '{"action":"search","query":"idempotent workflow retries"}',
            "Final bounded synthesis.",
        ]
    )
    research = FakeResearchClient()
    agent = ArchitectureAgent(
        client,
        "model",
        _context(),
        research_client=research,
        max_tool_steps=2,
    )

    answer = agent.ask("Design recovery.")

    assert answer == "Final bounded synthesis."
    assert len(research.queries) == 2
    assert len(client.calls) == 3


def test_non_json_decision_is_treated_as_answer() -> None:
    client = SequenceClient(["A direct architecture answer."])
    research = FakeResearchClient()
    agent = ArchitectureAgent(client, "model", _context(), research_client=research)

    assert agent.ask("Do we need external evidence?") == "A direct architecture answer."
    assert research.queries == []
