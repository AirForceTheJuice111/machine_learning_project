from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from agent_framework.core.engine import AgentEngine
from agent_framework.core.llm_client import OpenAILLMClient
from agent_framework.core.memory import ConversationMemory
from agent_framework.tools.cuda_probe_tools import (
    RunProbeFrequencyTool,
    RunProbeLatencyTool,
    RunProbeShmemTool,
)
from agent_framework.tools.system_tools import BashRunnerTool, ReadFileTool, WriteFileTool
from agent_framework.tools.tool_registry import ToolRegistry


SYSTEM_PROMPT = """你是一个本地 GPU 性能分析 Agent 的执行核心。

请严格遵循 ReAct 模式：
1. 先思考，再决定是否调用工具。
2. 如果调用工具，必须基于观察结果继续推理。
3. 当工具返回错误、超时、用户拒绝或日志截断时，不要崩溃，不要假装成功，要明确反思并调整策略。
4. 优先做小步、安全、可验证的动作。
5. 如果需要修改文件，优先先读取现有内容再写入。
6. 如果需要测量 GPU 实际频率，优先使用专用工具 run_probe_frequency，而不是手动编译并解析 stdout。
7. 如果需要测量 shared memory bank conflict，优先使用专用工具 run_probe_shmem，而不是手动编译并解析 stdout。
8. 如果需要测量缓存或全局内存延迟曲线，优先使用专用工具 run_probe_latency，而不是手动编译并解析 stdout。
9. 在任务可以完成时，直接给出最终答案，不要无休止调用工具。
"""


def build_registry(console: Console) -> ToolRegistry:
    project_root = Path(__file__).resolve().parent.parent

    def approval_handler(prompt: str) -> bool:
        console.print(Panel(prompt, title="Human Approval", border_style="yellow"))
        return Confirm.ask("是否批准执行？", default=False, console=console)

    registry = ToolRegistry()
    registry.register(BashRunnerTool(approval_handler=approval_handler))
    registry.register(ReadFileTool())
    registry.register(WriteFileTool(approval_handler=approval_handler))
    registry.register(RunProbeFrequencyTool(default_project_root=project_root))
    registry.register(RunProbeShmemTool(default_project_root=project_root))
    registry.register(RunProbeLatencyTool(default_project_root=project_root))
    return registry


def main() -> None:
    console = Console()
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        console.print(
            Panel(
                "缺少 OPENAI_API_KEY，无法启动 Agent。",
                title="Configuration Error",
                border_style="red",
            )
        )
        return

    memory = ConversationMemory(
        system_prompt=SYSTEM_PROMPT,
        max_tool_output_chars=int(os.getenv("AGENT_MAX_TOOL_CHARS", "6000")),
    )
    llm_client = OpenAILLMClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
    )
    engine = AgentEngine(
        llm_client=llm_client,
        tool_registry=build_registry(console),
        memory=memory,
        console=console,
        max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "10")),
    )

    console.print(
        Panel(
            "Bullet-proof Execution Engine 已启动。\n输入 `exit` 或 `quit` 结束会话。",
            title="Agent Framework",
            border_style="cyan",
        )
    )

    while True:
        user_input = Prompt.ask("[bold blue]User[/bold blue]")
        if user_input.strip().lower() in {"exit", "quit"}:
            console.print("[bold green]Bye.[/bold green]")
            break

        final_answer = engine.run(user_input)
        console.print(
            Panel(
                final_answer,
                title="Final Answer",
                border_style="green",
            )
        )


if __name__ == "__main__":
    main()
