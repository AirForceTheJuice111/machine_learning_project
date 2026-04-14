from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from agent_framework.core.engine import AgentEngine
from agent_framework.core.llm_client import OpenAILLMClient
from agent_framework.core.memory import ConversationMemory
from agent_framework.tools.cuda_probe_tools import CompileAndRunCudaSourceTool
from agent_framework.tools.system_tools import ReadFileTool, WriteFileTool
from agent_framework.tools.tool_registry import ToolRegistry


SYSTEM_PROMPT = """你是一个本地 GPU 性能分析 Agent 的执行核心。

请严格遵循 ReAct 模式：
1. 先思考，再决定是否调用工具。
2. 如果调用工具，必须基于观察结果继续推理。
3. 当工具返回错误、超时、用户拒绝或日志截断时，不要崩溃，不要假装成功，要明确反思并调整策略。
4. 优先做小步、安全、可验证的动作。
5. 如果需要修改文件，优先先读取现有内容再写入。
6. 严禁使用外部 benchmark，严禁下载第三方 benchmark；所有测量都必须基于你当前自主生成的本地 CUDA C++ 源码。
7. 自生成 CUDA 源码必须写入项目根目录下的 `generated_cuda/` 专用目录，再使用 compile_and_run_cuda_source 编译、运行；如需 ncu，只能用于分析你自己刚生成并编译出的本地二进制。
8. 不要依赖 target_spec 中的 run、外部可执行文件或互联网资源来完成测量。
9. 在任务可以完成时，直接给出最终答案，不要无休止调用工具。
"""


def build_registry(
    console: Console,
    *,
    auto_approve: bool = False,
) -> ToolRegistry:
    project_root = Path(__file__).resolve().parent.parent

    def approval_handler(prompt: str) -> bool:
        if auto_approve:
            return True
        console.print(Panel(prompt, title="Human Approval", border_style="yellow"))
        return Confirm.ask("是否批准执行？", default=False, console=console)

    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(
        WriteFileTool(
            approval_handler=approval_handler,
            require_approval=not auto_approve,
        )
    )
    registry.register(CompileAndRunCudaSourceTool(default_project_root=project_root))
    return registry


def main() -> None:
    console = Console()
    api_key = os.getenv("API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-5.4")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        console.print(
            Panel(
                "缺少 API_KEY，无法启动 Agent。",
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
