"""Claude orchestrator: conversation + tool use over `recall_memory`.

TODO(stream-2): implement the real Anthropic tool-use loop (messages.create with
tools=[RECALL_MEMORY_TOOL], handle tool_use blocks, feed tool_result, loop until end_turn).
Keep per-session history in memory keyed by session_id (cap ~20 turns).
"""

from __future__ import annotations

from datetime import UTC, datetime

from megamind.agent.tools import RecallMemory
from megamind.config import Settings
from megamind.models import ChatResponse, ToolCallRecord

SYSTEM_PROMPT = """You are Megamind, a calm assistant living in the wearer's AR headset.
You have a tool, recall_memory, that searches what the wearer has SEEN through their headset
cameras. Call it whenever a question depends on past perception (where something is, what
text said, who/what was there, what happened earlier). Current time (UTC): {now}.
Convert relative times into ISO-8601 ranges for the tool. Answer in one or two short spoken
sentences. If confidence is not_found, say you do not have a memory of it; never guess.
Mention when something was seen (e.g. "about 20 minutes ago")."""


class ClaudeAgent:
    def __init__(self, settings: Settings, recall: RecallMemory) -> None:
        self.settings = settings
        self.recall = recall
        self.histories: dict[str, list[dict]] = {}

    async def chat(self, session_id: str, text: str, device_id: str | None) -> ChatResponse:
        if not self.settings.anthropic_api_key:
            return await self._no_llm_fallback(text, device_id)
        # TODO(stream-2): real Claude tool-use loop
        return await self._no_llm_fallback(text, device_id)

    async def _no_llm_fallback(self, text: str, device_id: str | None) -> ChatResponse:
        """Deterministic path used before Claude is wired: always recall, echo evidence."""
        r = await self.recall(query=text, device_id=device_id)
        answer = r.evidence if r.confidence != "not_found" else "I don't have a memory of that."
        return ChatResponse(
            transcript=text,
            answer=answer,
            cited=r.cited,
            tool_calls=[ToolCallRecord(name="recall_memory", args={"query": text, "k": 12}, confidence=r.confidence)],
        )

    @staticmethod
    def system_prompt() -> str:
        return SYSTEM_PROMPT.format(now=datetime.now(UTC).isoformat(timespec="seconds"))
