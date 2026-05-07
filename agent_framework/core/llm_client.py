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

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
        return raw_value.strip().lower() in {"1", "true", "yes", "on"}

    def _uses_thinking_mode(self) -> bool:
        # Default to standard chat mode so model switching only requires
        # updating env vars like API_KEY / BASE_URL / BASE_MODEL.
        return self._env_flag("LLM_ENABLE_THINKING", False) or self._env_flag(
            "DEEPSEEK_ENABLE_THINKING", False
        )

    def _build_request_kwargs(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "auto"
        if self._uses_thinking_mode():
            request_kwargs["reasoning_effort"] = os.getenv("DEEPSEEK_REASONING_EFFORT", "high")
            request_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        return request_kwargs

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        try:
            response = self.client.chat.completions.create(**self._build_request_kwargs(messages=messages, tools=tools))
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
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content:
            assistant_message["reasoning_content"] = reasoning_content
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
