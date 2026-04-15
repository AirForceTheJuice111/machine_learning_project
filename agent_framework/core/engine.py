from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

from rich.console import Console
from rich.panel import Panel

from agent_framework.core.llm_client import OpenAILLMClient
from agent_framework.core.memory import ConversationMemory
from agent_framework.tools.tool_registry import ToolRegistry


@dataclass
class CompletionCheckResult:
    """Represents whether the current task can safely stop."""

    is_complete: bool
    final_answer: str | None = None
    feedback: str | None = None


CompletionChecker = Callable[[ConversationMemory, str], CompletionCheckResult]


@dataclass
class AgentEngine:
    """A completion-driven ReAct loop with a hard time ceiling."""

    llm_client: OpenAILLMClient
    tool_registry: ToolRegistry
    memory: ConversationMemory
    console: Console
    max_runtime_seconds: float | None = None
    completion_checker: CompletionChecker | None = None

    def _evaluate_completion(self, assistant_content: str) -> CompletionCheckResult:
        if self.completion_checker is None:
            return CompletionCheckResult(is_complete=False)
        return self.completion_checker(self.memory, assistant_content)

    def run(self, user_input: str) -> str:
        self.memory.add_user(user_input)
        start_time = time.monotonic()
        deadline = (
            start_time + self.max_runtime_seconds
            if self.max_runtime_seconds is not None
            else None
        )
        iteration = 0

        while True:
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                break
            iteration += 1
            if deadline is None:
                loop_header = f"Iteration {iteration} (no explicit time limit)"
            else:
                remaining_seconds = max(0.0, deadline - now)
                loop_header = (
                    f"Iteration {iteration} "
                    f"(remaining: {remaining_seconds:.1f}s / total: {self.max_runtime_seconds:.1f}s)"
                )
            self.console.print(
                Panel.fit(
                    loop_header,
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

            completion = self._evaluate_completion(response.content or "")
            if completion.is_complete:
                return completion.final_answer or response.content or "Agent 已完成，但没有返回额外文本。"

            if not response.tool_calls:
                if self.completion_checker is not None:
                    reminder = completion.feedback or (
                        "任务尚未满足结束条件，请继续补齐缺失步骤后再返回最终答案。"
                    )
                    self.memory.add_user(reminder)
                    self.console.print(
                        Panel(
                            reminder,
                            title="Continue Required",
                            border_style="yellow",
                        )
                    )
                    continue
                return response.content or "Agent 已完成，但没有返回额外文本。"

        elapsed_seconds = time.monotonic() - start_time
        if self.max_runtime_seconds is None:
            limit_message = (
                "执行被中止，但未设置显式时间上限。"
                "请检查调用方是否提供了完成条件或时间预算。"
            )
        else:
            limit_message = (
                f"已达到时间上限 {self.max_runtime_seconds:.1f}s "
                f"(实际耗时 {elapsed_seconds:.1f}s)，"
                "当前仍未满足完成条件。为避免无限循环，执行已停止。请基于当前观察结果调整提示词、完成判定或工具策略。"
            )
        self.console.print(
            Panel(limit_message, title="Safety Stop", border_style="red")
        )
        return limit_message
