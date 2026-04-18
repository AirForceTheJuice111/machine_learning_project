import argparse
import json
import os
import re
from pathlib import Path
import time
from typing import Any

from rich.console import Console
from rich.panel import Panel

from agent_framework.core.engine import AgentEngine, CompletionCheckResult
from agent_framework.core.llm_client import OpenAILLMClient
from agent_framework.core.memory import ConversationMemory
from agent_framework.main import SYSTEM_PROMPT, build_registry
from agent_framework.target_design_guidance import build_target_design_guidance

EVALUATOR_PROMPT_ADDITION = """
---------------------
[注意：你当前处于自动化评测模式。]
你每次只会收到一个硬件指标 target。
你的任务是：
1. 必须为当前 target 单独设计并生成一个 micro-benchmark，禁止用一个泛化实验覆盖多个 target。
2. 自生成 CUDA 源码必须写入指定目录，再使用 compile_and_run_cuda_source 编译、运行。
3. 你必须先通过自生成程序测出当前 target 的真实数值，再对你自己生成并编译出的程序完成至少一次 ncu profiling。
4. 只要已经得到可信的目标数值，并完成了对该自生成程序的 ncu 分析，就可以结束，不要求额外修改代码做迭代。
5. 严禁使用外部 benchmark、第三方 benchmark、互联网下载资源，严禁把 target_spec 中可能出现的 run 外部可执行文件当作测量依据。
6. 最终答案必须是一个 JSON 代码块，且只返回当前 target 的结果。

示例格式：
```json
{
  "target": "dram_latency_cycles",
  "value": 440.0,
  "benchmark_source_path": "D:/.../generated_cuda/03_dram_latency_cycles/probe.cu",
  "used_ncu_analysis": true,
  "evidence_summary": "先用 pointer chasing 程序测出延迟，再对该程序执行 ncu profiling 并确认测量路径合理。"
}
```
在得出最终 JSON 前，不要轻易结束。
"""

def extract_json(text: str) -> dict[str, Any]:
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}

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


def fallback_extract_results_from_memory(memory: ConversationMemory, targets: list[str]) -> dict[str, float]:
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


def slugify_target(target: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", target.strip())
    slug = slug.strip("._")
    return slug or "target"


def build_target_prompt(
    *,
    target: str,
    target_workspace: Path,
) -> str:
    prompt = (
        "请开始单目标评测任务。\n"
        f"当前唯一目标指标：{target}\n"
        f"当前 target 的专用工作目录：{target_workspace}\n\n"
    )
    prompt += build_target_design_guidance([target]) + "\n\n"
    prompt += (
        f"你必须把当前 target 的 CUDA 源码和编译产物放在该目录下：{target_workspace}\n"
        "要求：\n"
        "1. 先生成 micro-benchmark 并编译运行，测出当前 target 的真实数值。\n"
        "2. 然后至少进行一轮 compile_and_run_cuda_source(profile_with_ncu=true) 对你自己生成的程序做 ncu profiling。\n"
        "3. 如果已经得到可信数值并完成 ncu 分析，就直接输出最终 JSON，不要求强制改代码迭代。\n"
        "4. 最终只输出当前 target 的 JSON 结果，不要输出其它 target。\n"
        "5. 不要依赖 target_spec 中可能存在的 run、外部可执行文件或第三方 benchmark。\n"
        "6. 上面的 target 设计信息是约束，不是代码模板；你必须自己决定具体 kernel、循环、数据布局、扫描范围和结果换算方式。\n"
    )
    return prompt


def build_engine(
    *,
    console: Console,
    llm_client: OpenAILLMClient,
    max_tool_output_chars: int,
    max_runtime_seconds: float,
    target: str,
    target_workspace: Path,
) -> AgentEngine:
    memory = ConversationMemory(
        system_prompt=SYSTEM_PROMPT + EVALUATOR_PROMPT_ADDITION,
        max_tool_output_chars=max_tool_output_chars,
    )
    return AgentEngine(
        llm_client=llm_client,
        tool_registry=build_registry(console, auto_approve=True),
        memory=memory,
        console=console,
        max_runtime_seconds=max_runtime_seconds,
        completion_checker=build_target_completion_checker(
            target=target,
            target_workspace=target_workspace,
        ),
    )


def _resolve_workspace_path(path_value: str, workspace: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidates = [
        (workspace / path).resolve(),
        (workspace.parent.parent / path).resolve(),
    ]
    workspace_root = workspace.resolve()
    for candidate in candidates:
        try:
            candidate.relative_to(workspace_root)
            return candidate
        except ValueError:
            continue
    return candidates[0]


def analyze_target_execution(memory: ConversationMemory, target_workspace: Path) -> dict[str, Any]:
    tool_results_by_id: dict[str, str] = {}
    for message in memory.messages:
        if message.get("role") == "tool":
            tool_results_by_id[message.get("tool_call_id", "")] = message.get("content", "")

    compile_runs = 0
    ncu_runs = 0
    iterative_compile_runs = 0
    successful_sources: list[str] = []
    pending_workspace_writes = 0
    last_successful_source: str | None = None
    last_successful_program_args: list[str] | None = None
    for message in memory.messages:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls", []) or []:
            function = tool_call.get("function", {})
            function_name = function.get("name")
            arguments = function.get("arguments", "")
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue

            if function_name == "write_file":
                file_path = parsed_arguments.get("path", "")
                if not isinstance(file_path, str):
                    continue
                resolved_write_path = _resolve_workspace_path(file_path, target_workspace)
                try:
                    resolved_write_path.relative_to(target_workspace.resolve())
                    pending_workspace_writes += 1
                except ValueError:
                    pass
                continue

            if function_name != "compile_and_run_cuda_source":
                continue
            source_path = parsed_arguments.get("source_path", "")
            if not isinstance(source_path, str):
                continue
            resolved_source = _resolve_workspace_path(source_path, target_workspace)
            try:
                resolved_source.relative_to(target_workspace.resolve())
            except ValueError:
                continue
            tool_result = tool_results_by_id.get(tool_call.get("id", ""), "")
            if "[compile_and_run_cuda_source] status=ok" not in tool_result:
                continue
            compile_runs += 1
            if bool(parsed_arguments.get("profile_with_ncu", False)):
                ncu_runs += 1
            resolved_source_str = str(resolved_source)
            successful_sources.append(resolved_source_str)
            program_args = parsed_arguments.get("program_args", [])
            if not isinstance(program_args, list):
                program_args = []
            if (
                last_successful_source is not None
                and (
                    pending_workspace_writes > 0
                    or resolved_source_str != last_successful_source
                    or program_args != (last_successful_program_args or [])
                )
            ):
                iterative_compile_runs += 1
            pending_workspace_writes = 0
            last_successful_source = resolved_source_str
            last_successful_program_args = [str(arg) for arg in program_args]

    return {
        "compile_runs": compile_runs,
        "ncu_runs": ncu_runs,
        "iterative_compile_runs": iterative_compile_runs,
        "successful_sources": successful_sources,
    }


def validate_target_result(
    *,
    target: str,
    result_dict: dict[str, Any],
    target_workspace: Path,
    execution_info: dict[str, Any],
) -> tuple[bool, str]:
    required_keys = {
        "target",
        "value",
        "benchmark_source_path",
        "used_ncu_analysis",
        "evidence_summary",
    }
    missing = required_keys - set(result_dict)
    if missing:
        return False, f"缺少必要字段: {sorted(missing)}"
    if result_dict["target"] != target:
        return False, f"target 字段不匹配: 期望 {target}，实际 {result_dict['target']}"
    if not isinstance(result_dict["value"], (int, float)):
        return False, "value 必须是数字"
    if not isinstance(result_dict["used_ncu_analysis"], bool) or not result_dict["used_ncu_analysis"]:
        return False, "used_ncu_analysis 必须为 true"
    if not isinstance(result_dict["evidence_summary"], str) or not result_dict["evidence_summary"].strip():
        return False, "evidence_summary 必须是非空字符串"
    benchmark_source_path = result_dict["benchmark_source_path"]
    if not isinstance(benchmark_source_path, str) or not benchmark_source_path.strip():
        return False, "benchmark_source_path 必须是非空字符串"
    resolved_source = _resolve_workspace_path(benchmark_source_path, target_workspace)
    try:
        resolved_source.relative_to(target_workspace.resolve())
    except ValueError:
        return False, f"benchmark_source_path 必须位于当前 target 工作目录内: {resolved_source}"
    if not resolved_source.exists():
        return False, f"benchmark_source_path 指向的文件不存在: {resolved_source}"
    if execution_info["compile_runs"] < 1:
        return False, "当前 target 尚未完成成功编译运行"
    if execution_info["ncu_runs"] < 1:
        return False, "当前 target 尚未完成至少一次带 ncu 的成功 profiling"
    return True, ""


def build_incomplete_feedback(
    *,
    target: str,
    execution_info: dict[str, Any],
    validation_error: str | None,
) -> str:
    lines = [f"当前 target `{target}` 尚未满足结束条件。"]
    if validation_error:
        lines.append(f"现有输出问题: {validation_error}")
    if execution_info["compile_runs"] < 1:
        lines.append("还没有成功完成当前 target 的编译运行与数值测量。")
    if execution_info["ncu_runs"] < 1:
        lines.append("还没有成功执行带 ncu 的 profiling。")
    if execution_info["compile_runs"] >= 1 and execution_info["ncu_runs"] >= 1:
        lines.append("实验记录已经足够，请立即整理并返回最终 JSON，不要继续空转。")
    else:
        lines.append("请继续调用工具先测出真实数值，再完成 ncu 分析，之后输出最终 JSON。")
    return " ".join(lines)


def build_target_completion_checker(
    *,
    target: str,
    target_workspace: Path,
):
    def checker(memory: ConversationMemory, assistant_content: str) -> CompletionCheckResult:
        execution_info = analyze_target_execution(memory, target_workspace)
        result_dict = extract_json(assistant_content)
        validation_error: str | None = None

        if result_dict:
            is_valid, validation_error = validate_target_result(
                target=target,
                result_dict=result_dict,
                target_workspace=target_workspace,
                execution_info=execution_info,
            )
            if is_valid:
                return CompletionCheckResult(
                    is_complete=True,
                    final_answer=assistant_content,
                )

        fallback_value = fallback_extract_results_from_memory(memory, [target])
        if (
            target in fallback_value
            and execution_info["compile_runs"] >= 1
            and execution_info["ncu_runs"] >= 1
            and execution_info["successful_sources"]
        ):
            synthesized_result = {
                "target": target,
                "value": fallback_value[target],
                "benchmark_source_path": execution_info["successful_sources"][-1],
                "used_ncu_analysis": True,
                "evidence_summary": "Agent 未显式收尾，但已完成当前 target 的真实测量与 ncu 分析，结果从有效工具输出中自动提取。",
            }
            return CompletionCheckResult(
                is_complete=True,
                final_answer=json.dumps(synthesized_result, ensure_ascii=False, indent=2),
            )

        feedback = build_incomplete_feedback(
            target=target,
            execution_info=execution_info,
            validation_error=validation_error if result_dict else None,
        )
        return CompletionCheckResult(
            is_complete=False,
            feedback=feedback,
        )

    return checker

def main() -> None:
    parser = argparse.ArgumentParser(description="Automated Hardware Probe Evaluator")
    parser.add_argument("--target-spec", type=str, default="/target/target_spec.json", help="Path to target_spec.json")
    parser.add_argument("--output", type=str, default="/workspace/output.json", help="Path to write final output file")
    parser.add_argument(
        "--skip-details-output",
        action="store_true",
        help="Do not emit the secondary details output file.",
    )
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
        return

    api_key = os.getenv("API_KEY")
    model = os.getenv("BASE_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5.4")
    base_url = os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL")

    if not api_key:
        console.print(Panel("缺少 API_KEY，无法启动评测 Agent。", title="Config Error", border_style="red"))
        return

    max_tool_output_chars = int(os.getenv("AGENT_MAX_TOOL_CHARS", "6000"))
    total_runtime_seconds = float(os.getenv("AGENT_MAX_TOTAL_RUNTIME_SECONDS", "2100"))
    llm_client = OpenAILLMClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
    )
    generated_cuda_dir = (spec_path.parent / "generated_cuda").resolve()
    generated_cuda_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            f"评测目标加载完成，共 {len(targets)} 项。\n"
            f"自生成 CUDA 根目录: {generated_cuda_dir}\n"
            f"总时间预算: {total_runtime_seconds / 60:.1f} 分钟\n"
            f"run 字段将被忽略: {run_executable or '未提供'}",
            title="Evaluator Started",
            border_style="cyan",
        )
    )

    aggregated_results: dict[str, float] = {}
    detailed_results: dict[str, Any] = {"targets": {}, "ignored_run": run_executable}
    evaluation_start_time = time.monotonic()
    evaluation_deadline = evaluation_start_time + total_runtime_seconds

    for index, target in enumerate(targets, start=1):
        remaining_total_seconds = evaluation_deadline - time.monotonic()
        if remaining_total_seconds <= 0:
            failure_payload = {
                "error": (
                    f"总测试时间已超过 {total_runtime_seconds:.1f}s，"
                    f"在完成 target `{target}` 前停止。"
                ),
                "target": target,
                "partial_results": aggregated_results,
                "ignored_run": run_executable,
                "elapsed_seconds": time.monotonic() - evaluation_start_time,
                "time_budget_seconds": total_runtime_seconds,
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(failure_payload, f, indent=2, ensure_ascii=False)
            console.print(
                Panel(
                    failure_payload["error"],
                    title="Evaluation Timed Out",
                    border_style="red",
                )
            )
            return
        target_workspace = generated_cuda_dir / f"{index:02d}_{slugify_target(target)}"
        target_workspace.mkdir(parents=True, exist_ok=True)
        console.print(
            Panel.fit(
                f"Target {index}/{len(targets)}: {target}\n"
                f"Workspace: {target_workspace}\n"
                f"Remaining total time: {remaining_total_seconds / 60:.1f} min",
                title="Per-Target Orchestrator",
                border_style="blue",
            )
        )

        engine = build_engine(
            console=console,
            llm_client=llm_client,
            max_tool_output_chars=max_tool_output_chars,
            max_runtime_seconds=remaining_total_seconds,
            target=target,
            target_workspace=target_workspace,
        )
        prompt = build_target_prompt(target=target, target_workspace=target_workspace)
        final_answer = engine.run(prompt)
        result_dict = extract_json(final_answer)
        execution_info = analyze_target_execution(engine.memory, target_workspace)

        if not result_dict:
            fallback_value = fallback_extract_results_from_memory(engine.memory, [target])
            if (
                target in fallback_value
                and execution_info["compile_runs"] >= 1
                and execution_info["ncu_runs"] >= 1
                and execution_info["successful_sources"]
            ):
                result_dict = {
                    "target": target,
                    "value": fallback_value[target],
                    "benchmark_source_path": execution_info["successful_sources"][-1],
                    "used_ncu_analysis": True,
                    "evidence_summary": "Agent 未显式收尾，已从成功的 benchmark 输出与 ncu 分析中自动提取当前 target 的结果。",
                }
                console.print(
                    Panel(
                        f"Target `{target}` 未显式输出最终 JSON，但已从成功的工具输出中提取结果。",
                        title="Fallback Extraction",
                        border_style="yellow",
                    )
                )

        is_valid, error_message = validate_target_result(
            target=target,
            result_dict=result_dict,
            target_workspace=target_workspace,
            execution_info=execution_info,
        )
        if not is_valid:
            failure_payload = {
                "error": f"Target `{target}` 的结果校验失败: {error_message}",
                "target": target,
                "raw_output": final_answer,
                "parsed_output": result_dict,
                "execution_info": execution_info,
                "partial_results": aggregated_results,
                "ignored_run": run_executable,
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(failure_payload, f, indent=2, ensure_ascii=False)
            console.print(
                Panel(
                    failure_payload["error"],
                    title="Evaluation Failed",
                    border_style="red",
                )
            )
            return

        aggregated_results[target] = float(result_dict["value"])
        detailed_results["targets"][target] = {
            "value": float(result_dict["value"]),
            "benchmark_source_path": result_dict["benchmark_source_path"],
            "used_ncu_analysis": result_dict["used_ncu_analysis"],
            "evidence_summary": result_dict["evidence_summary"],
            "execution_info": execution_info,
            "workspace": str(target_workspace),
            "raw_output": final_answer,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(aggregated_results, f, indent=2, ensure_ascii=False)
    details_path = None
    if not args.skip_details_output:
        details_path = out_path.with_name(f"{out_path.stem}.details.json")
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    completion_message = f"结果已成功写入: {out_path}"
    if details_path is not None:
        completion_message += f"\n详细结果已写入: {details_path}"
    console.print(
        Panel(
            completion_message,
            title="Evaluation Completed",
            border_style="green",
        )
    )

if __name__ == "__main__":
    main()
