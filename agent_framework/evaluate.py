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
from agent_framework.target_design_guidance import build_target_design_guidance

EVALUATOR_PROMPT_ADDITION = """
---------------------
[注意：你当前处于自动化评测模式。]
你将收到一份包含多个硬件指标要求的 target_spec.json 内容。
你的任务是：
1. 不要把仓库里预置的 probe 当作默认 benchmark。应根据当前 target_spec 和 profiling 目标，自主生成最小化 CUDA C++ 探针源码。
2. 使用 write_file 写入源码，再使用 compile_and_run_cuda_source 编译、运行，必要时开启 ncu profiling。
3. 如果 target_spec 中提供了 run 可执行文件路径，应将其视为动态算子目标，可用 bash_runner 调用 ncu 分析，而不是假设固定算子。
4. 从探针输出、ncu 日志或程序输出中推断、计算并交叉验证目标数值。
5. 一旦你确定了所有的指标值，你必须输出一份纯 JSON 格式的最终答案，必须被包裹在 Markdown 的 JSON 代码块中（即 ```json 和 ``` 之间），并且键名必须严格匹配要求。
6. 如果 compile_and_run_cuda_source 的 run_stdout 中已经明确打印出所有 targets 对应的 `name: value` 结果，你必须立刻停止继续试探，直接整理成最终 JSON 返回。
7. `run_stderr`、`profile_stdout`、`profile_stderr` 为空并不代表失败；如果所需指标已经在 run_stdout 中出现，就直接收尾。
8. 如果提供了有效的 run 可执行文件路径，在最终输出前你必须至少对它做一次分析：要么通过 ncu / bash_runner 分析它，要么明确判断它与当前目标无关且无法提供有效证据。不要在 run 仍可用且尚未分析前仅凭自生成 probe 结果提前结束。

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


def resolve_run_executable(spec_path: Path, raw_run: str) -> Path | None:
    if not raw_run:
        return None
    run_path = Path(raw_run).expanduser()
    if not run_path.is_absolute():
        run_path = (spec_path.parent / run_path).resolve()
    return run_path


def extract_tool_json(rendered_tool_output: str) -> dict:
    marker = "\n{"
    start = rendered_tool_output.find(marker)
    if start == -1:
        return {}
    try:
        return json.loads(rendered_tool_output[start + 1 :])
    except json.JSONDecodeError:
        return {}


def extract_numeric_targets_from_text(text: str, targets: list[str]) -> dict[str, float]:
    results: dict[str, float] = {}
    for target in targets:
        pattern = rf"(?im)^\s*{re.escape(target)}\s*:\s*([-+]?\d+(?:\.\d+)?)\s*$"
        match = re.search(pattern, text)
        if match:
            try:
                results[target] = float(match.group(1))
            except ValueError:
                continue
    return results


def fallback_extract_results_from_memory(memory: ConversationMemory, targets: list[str]) -> dict:
    merged_results: dict[str, float] = {}
    for message in reversed(memory.messages):
        if message.get("role") != "tool":
            continue
        content = message.get("content", "")
        if "[compile_and_run_cuda_source] status=ok" not in content:
            continue

        tool_payload = extract_tool_json(content)
        candidate_texts = [content]
        if tool_payload:
            for key in ("run_stdout", "profile_stdout", "run_stderr", "profile_stderr"):
                value = tool_payload.get(key, "")
                if isinstance(value, str) and value.strip():
                    candidate_texts.append(value)

        for candidate in candidate_texts:
            merged_results.update(extract_numeric_targets_from_text(candidate, targets))

        if all(target in merged_results for target in targets):
            return {target: merged_results[target] for target in targets}

    return {}


def run_target_was_analyzed(memory: ConversationMemory, run_executable: Path | None) -> bool:
    if not run_executable:
        return False
    run_texts = {str(run_executable), run_executable.as_posix(), run_executable.name}
    for message in memory.messages:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls", []) or []:
            function = tool_call.get("function", {})
            if function.get("name") != "bash_runner":
                continue
            arguments = function.get("arguments", "")
            if not isinstance(arguments, str):
                continue
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                continue
            command = parsed.get("command", "")
            if not isinstance(command, str):
                continue
            if "ncu" not in command.lower():
                continue
            if any(text in command for text in run_texts):
                return True
    return False

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

    api_key = os.getenv("API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-5.4")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        console.print(Panel("缺少 API_KEY，无法启动评测 Agent。", title="Config Error", border_style="red"))
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
        tool_registry=build_registry(
            console,
            auto_approve=True,
        ),
        memory=memory,
        console=console,
        max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "15")),
    )

    prompt = (
        f"请开始评测任务。以下是需要测量的硬件指标目标（targets）：\n"
        f"{json.dumps(targets, ensure_ascii=False, indent=2)}\n\n"
    )
    prompt += build_target_design_guidance(targets) + "\n\n"
    resolved_run_executable = resolve_run_executable(spec_path, run_executable) if run_executable else None
    run_is_available = bool(resolved_run_executable and resolved_run_executable.exists())
    if run_executable:
        prompt += (
            f"目标测试算子路径（run）：{run_executable}\n"
            f"解析后的绝对路径：{resolved_run_executable if resolved_run_executable else '无法解析'}\n"
            f"该路径当前是否存在：{'是' if run_is_available else '否'}\n"
            "如果 run 文件存在且可执行，你需要至少做一次基于它的 ncu / bash_runner 分析，并把得到的证据与自生成探针结果交叉验证。\n"
            "如果 run 文件不存在、不可执行或与目标无关，你可以放弃使用它，但应在最终答案前完成自己的 probe 测量。\n\n"
        )
    prompt += (
        "请根据目标自主生成并编译运行 CUDA 探针，必要时结合 ncu 和 run 可执行文件进行分析，推断上述数值，"
        "并确保在最终回复里输出标准的 JSON Block。"
    )

    console.print(Panel(f"评测目标加载完成，共 {len(targets)} 项。\n开始调用 Agent 核心...", title="Evaluator Started", border_style="cyan"))
    final_answer = engine.run(prompt)

    # Attempt to extract JSON from the final answer
    result_dict = extract_json(final_answer)
    if not result_dict:
        fallback_results = fallback_extract_results_from_memory(memory, targets)
        run_allows_fallback = (
            not run_executable
            or not run_is_available
            or run_target_was_analyzed(memory, resolved_run_executable)
        )
        if fallback_results and all(target in fallback_results for target in targets) and run_allows_fallback:
            result_dict = fallback_results
            console.print(
                Panel(
                    "Agent 未显式输出最终 JSON，但已从成功的工具输出中提取到全部目标结果，已自动写入 results.json。",
                    title="Fallback Extraction",
                    border_style="yellow",
                )
            )
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not result_dict:
        console.print("[bold red]Agent 未能返回符合格式的 JSON 结果。将保存原始输出作为错误日志。[/bold red]")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    "error": "Failed to extract JSON",
                    "raw_output": final_answer,
                    "fallback_partial": fallback_extract_results_from_memory(memory, targets),
                    "run_executable": str(resolved_run_executable) if resolved_run_executable else "",
                    "run_available": run_is_available,
                    "run_was_analyzed": run_target_was_analyzed(memory, resolved_run_executable),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    else:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        console.print(Panel(f"结果已成功写入: {out_path}", title="Evaluation Completed", border_style="green"))

if __name__ == "__main__":
    main()
