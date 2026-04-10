import argparse
import json
import os
import re
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from agent_framework.core.engine import AgentEngine
from agent_framework.core.llm_client import OpenAILLMClient
from agent_framework.core.memory import ConversationMemory
from agent_framework.main import SYSTEM_PROMPT, build_registry

EVALUATOR_PROMPT_ADDITION = """
---------------------
[注意：你当前处于自动化评测模式。]
你将收到一份包含多个硬件指标要求的 target_spec.json 内容。
你的任务是：
1. 思考如何利用现有的 probe 工具（频率探针、共享内存探针、延迟扫频探针）或者系统命令工具（bash_runner）获取这些指标。
2. 调度工具并运行探针。
3. 从探针的原始 CSV/日志中推断、计算或提取出具体数值。
4. 一旦你确定了所有的指标值，你必须输出一份纯 JSON 格式的最终答案，必须被包裹在 Markdown 的 JSON 代码块中（即 ```json 和 ``` 之间），并且键名必须严格匹配要求。

示例格式：
```json
{
  "actual_boost_clock_mhz": 1234.5,
  "dram_latency_cycles": 440.0
}
```
在得出最终 JSON 前，不要轻易结束。
"""

def extract_json(text: str) -> dict:
    # Try to find JSON block
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Fallback to parse entire text if no markdown block
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}

def main() -> None:
    parser = argparse.ArgumentParser(description="Automated Hardware Probe Evaluator")
    parser.add_argument("--target-spec", type=str, required=True, help="Path to target_spec.json")
    parser.add_argument("--output", type=str, default="results.json", help="Path to write results.json")
    args = parser.parse_args()

    console = Console()
    spec_path = Path(args.target_spec).resolve()
    out_path = Path(args.output).resolve()

    if not spec_path.exists():
        console.print(Panel(f"Error: 找不到文件 {spec_path}", title="File Not Found", border_style="red"))
        return

    with open(spec_path, 'r', encoding='utf-8') as f:
        try:
            target_spec = json.load(f)
        except json.JSONDecodeError as exc:
            console.print(Panel(f"解析 {spec_path} 失败: {exc}", title="JSON Error", border_style="red"))
            return

    targets = target_spec.get("targets", [])
    run_executable = target_spec.get("run", "")
    if not targets:
        console.print("[yellow]Warning: target_spec.json 中没有找到 targets 列表。[/yellow]")

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        console.print(Panel("缺少 OPENAI_API_KEY，无法启动评测 Agent。", title="Config Error", border_style="red"))
        return

    memory = ConversationMemory(
        system_prompt=SYSTEM_PROMPT + EVALUATOR_PROMPT_ADDITION,
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
        max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "15")),
    )

    prompt = (
        f"请开始评测任务。以下是需要测量的硬件指标目标（targets）：\n"
        f"{json.dumps(targets, ensure_ascii=False, indent=2)}\n\n"
    )
    if run_executable:
        prompt += (
            f"目标测试算子路径（run）：{run_executable}\n"
            "如果你需要进行算子级的指标分析，请优先通过 ncu 对此可执行文件进行分析（例如：使用 bash_runner 执行 ncu 命令获取详细指标）。\n\n"
        )
    prompt += (
        "请按需编译和运行现有的 CUDA probe（或执行其他命令），推断上述数值，"
        "并确保在最终回复里输出标准的 JSON Block。"
    )

    console.print(Panel(f"评测目标加载完成，共 {len(targets)} 项。\n开始调用 Agent 核心...", title="Evaluator Started", border_style="cyan"))
    final_answer = engine.run(prompt)

    # Attempt to extract JSON from the final answer
    result_dict = extract_json(final_answer)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not result_dict:
        console.print("[bold red]Agent 未能返回符合格式的 JSON 结果。将保存原始输出作为错误日志。[/bold red]")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({"error": "Failed to extract JSON", "raw_output": final_answer}, f, indent=2, ensure_ascii=False)
    else:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        console.print(Panel(f"结果已成功写入: {out_path}", title="Evaluation Completed", border_style="green"))

if __name__ == "__main__":
    main()
