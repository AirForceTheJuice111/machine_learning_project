from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ToolResult:
    """Structured tool execution result."""

    status: str
    content: str

    def render(self, tool_name: str) -> str:
        return f"[{tool_name}] status={self.status}\n{self.content}"


class Tool(Protocol):
    name: str
    description: str
    parameters_schema: dict[str, Any]

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        ...


class ToolRegistry:
    """Stores tools and executes them safely."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def openai_schemas(self) -> list[dict[str, Any]]:
        schemas = []
        for tool in self._tools.values():
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters_schema,
                    },
                }
            )
        return schemas

    def execute(self, name: str, raw_arguments: str) -> ToolResult:
        if name not in self._tools:
            return ToolResult(
                status="error",
                content=f"未知工具: {name}",
            )

        try:
            arguments = json.loads(raw_arguments or "{}")
            if not isinstance(arguments, dict):
                return ToolResult(
                    status="error",
                    content=f"工具参数必须是 JSON object，收到: {type(arguments).__name__}",
                )
        except json.JSONDecodeError as exc:
            return ToolResult(
                status="error",
                content=f"工具参数解析失败: {exc}",
            )

        try:
            return self._tools[name].run(arguments)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                content=(
                    f"工具 {name} 发生未捕获异常: {exc}\n"
                    f"{traceback.format_exc(limit=3)}"
                ),
            )
