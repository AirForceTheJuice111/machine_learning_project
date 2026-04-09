from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def truncate_output(text: str, max_length: int = 4000) -> str:
    """Keep the beginning and end of large tool outputs."""
    if text is None:
        return ""

    normalized = text.strip()
    if len(normalized) <= max_length:
        return normalized

    head_length = int(max_length * 0.4)
    tail_length = max_length - head_length - len("\n...\n[output truncated]\n...\n")
    if tail_length < 0:
        tail_length = max_length // 2
        head_length = max_length - tail_length

    head = normalized[:head_length]
    tail = normalized[-tail_length:]
    return f"{head}\n...\n[output truncated]\n...\n{tail}"


@dataclass
class ConversationMemory:
    """Owns the message history sent to the LLM."""

    system_prompt: str
    max_tool_output_chars: int = 6000
    messages: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.messages.append({"role": "system", "content": self.system_prompt})

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(
        self,
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        message: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if tool_calls:
            message["tool_calls"] = tool_calls
        self.messages.append(message)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        safe_content = truncate_output(content, self.max_tool_output_chars)
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": safe_content,
            }
        )

    def snapshot(self) -> list[dict[str, Any]]:
        return list(self.messages)
