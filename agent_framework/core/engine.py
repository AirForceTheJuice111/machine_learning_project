from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel

from agent_framework.core.llm_client import OpenAILLMClient
from agent_framework.core.memory import ConversationMemory
from agent_framework.tools.tool_registry import ToolRegistry


@dataclass
class AgentEngine:
    """A bounded ReAct loop with defensive exception handling."""

    llm_client: OpenAILLMClient
    tool_registry: ToolRegistry
    memory: ConversationMemory
    console: Console
    max_iterations: int = 10

    def run(self, user_input: str) -> str:
        self.memory.add_user(user_input)

        for iteration in range(1, self.max_iterations + 1):
            self.console.print(
                Panel.fit(
                    f"Iteration {iteration}/{self.max_iterations}",
                    title="Agent Loop",
                    border_style="cyan",
                )
            )

            try:
                response = self.llm_client.complete(
                    messages=self.memory.snapshot(),
                    tools=self.tool_registry.openai_schemas(),
                )
            except Exception as exc:  # noqa: BLE001
                error_message = f"LLM 调用失败，主循环已安全退出: {exc}"
                self.console.print(Panel(error_message, title="LLM Error", border_style="red"))
                return error_message

            self.memory.add_assistant(
                content=response.assistant_message.get("content"),
                tool_calls=response.assistant_message.get("tool_calls"),
            )

            if response.content.strip():
                self.console.print(
                    Panel(
                        response.content,
                        title="Assistant",
                        border_style="green",
                    )
                )

            if not response.tool_calls:
                return response.content or "Agent 已完成，但没有返回额外文本。"

            for tool_call in response.tool_calls:
                self.console.print(
                    Panel.fit(
                        f"{tool_call.name}({tool_call.arguments})",
                        title="Tool Call",
                        border_style="yellow",
                    )
                )
                result = self.tool_registry.execute(
                    name=tool_call.name,
                    raw_arguments=tool_call.arguments,
                )
                rendered_result = result.render(tool_call.name)
                self.memory.add_tool_result(tool_call.id, rendered_result)
                self.console.print(
                    Panel(
                        rendered_result,
                        title="Observation",
                        border_style="magenta",
                    )
                )

        limit_message = (
            f"已达到最大迭代次数 {self.max_iterations}，"
            "为避免无限循环，执行已停止。请基于当前观察结果调整提示词或工具策略。"
        )
        self.console.print(
            Panel(limit_message, title="Safety Stop", border_style="red")
        )
        return limit_message
