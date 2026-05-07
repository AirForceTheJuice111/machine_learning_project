from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APITimeoutError, OpenAI


@dataclass
class LLMToolCall:
    """A normalized tool call emitted by the model."""

    id: str
    name: str
    arguments: str


@dataclass
class LLMResponse:
    """A normalized assistant response."""

    content: str
    tool_calls: list[LLMToolCall]
    assistant_message: dict[str, Any]


class OpenAILLMClient:
    """Wraps OpenAI chat completions and tool binding."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.1,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.request_timeout_seconds = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "180"))
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.request_timeout_seconds,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=self.temperature,
            )
        except APITimeoutError as exc:
            raise RuntimeError(
                "LLM 请求超时。请检查 BASE_URL / 网络连通性，或增大环境变量 "
                "LLM_REQUEST_TIMEOUT_SECONDS 后重试。"
            ) from exc
        except APIConnectionError as exc:
            raise RuntimeError(
                "LLM 连接失败。请检查 BASE_URL 是否正确、网络是否可达，以及代理/防火墙配置。"
            ) from exc
        message = response.choices[0].message
        tool_calls = []
        assistant_tool_calls: list[dict[str, Any]] = []

        for tool_call in message.tool_calls or []:
            tool_calls.append(
                LLMToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                )
            )
            assistant_tool_calls.append(
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            )

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }
        if assistant_tool_calls:
            assistant_message["tool_calls"] = assistant_tool_calls

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            assistant_message=assistant_message,
        )

    def chat(self, prompt: str, *, system_prompt: str | None = None) -> str:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.complete(messages=messages, tools=[]).content
