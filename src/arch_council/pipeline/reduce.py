from __future__ import annotations

from ..governance import DebateState, aggregate_debate_state


def reduce_round(responses: tuple[str, str, str]) -> DebateState:
    """Deterministic first choice for reducing structured debate output."""
    return aggregate_debate_state(responses)
