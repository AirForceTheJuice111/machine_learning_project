import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from agent_framework.core.engine import AgentEngine, CompletionCheckResult
from agent_framework.core.llm_client import OpenAILLMClient
from agent_framework.core.memory import ConversationMemory
from agent_framework.main import SYSTEM_PROMPT, build_registry
from agent_framework.target_design_guidance import build_family_design_guidance, group_targets_by_family

EVALUATOR_PROMPT_ADDITION = """
---------------------
[注意：你当前处于自动化评测模式。]
你每次会收到一个 target family，而不是单独一个 target。
1. 为当前 family 生成一份共享的 CUDA micro-benchmark，并把源码写入指定目录。
2. 同一 family 的 benchmark 必须支持多个 mode 或参数路径，以便同一份 .cu / 同一 binary 复用测量多个 target。
3. 你必须先编译一次共享 benchmark，然后尽量通过 skip_compile=true 复用已有 binary，针对不同 mode 多次运行。
4. 你必须至少执行一次 compile_and_run_cuda_source(profile_with_ncu=true) 对该共享 benchmark 做 ncu profiling。
5. 每次运行时，请让程序输出机器可解析的结果行，格式为 `target_name: numeric_value`。
6. 严禁使用外部 benchmark、第三方 benchmark、互联网下载资源，严禁把 target_spec 中可能出现的 run 外部可执行文件当作测量依据。
7. 最终答案必须是一个 JSON 代码块，返回当前 family 中全部 target 的结果。
"""


def extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def extract_tool_json(rendered_tool_output: str) -> dict[str, Any]:
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


def write_json_output(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_failure_output(
    *,
    console: Console,
    out_path: Path,
    title: str,
    message: str,
    **extra_fields: Any,
) -> None:
    payload = {"error": message, **extra_fields}
    write_json_output(out_path, payload)
    console.print(Panel(message, title=title, border_style="red"))


def slugify_target(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    slug = slug.strip("._")
    return slug or "target"


def normalize_program_args(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def infer_mode_from_program_args(program_args: list[str], target: str) -> str:
    for index, arg in enumerate(program_args):
        arg_lower = str(arg).lower()
        if arg_lower in {"--mode", "-m", "mode"} and index + 1 < len(program_args):
            return str(program_args[index + 1])
        if str(arg) == target:
            return target
    if program_args:
        return "args:" + " ".join(program_args)
    return target


def build_family_prompt(*, family: str, family_targets: list[str], family_workspace: Path) -> str:
    prompt = (
        "请开始 family 级评测任务。\n"
        f"当前 family: {family}\n"
        f"当前 family 内的 targets: {json.dumps(family_targets, ensure_ascii=False)}\n"
        f"当前 family 的专用工作目录: {family_workspace}\n\n"
    )
    prompt += build_family_design_guidance(family, family_targets) + "\n\n"
    prompt += (
        f"你必须把当前 family 的 CUDA 源码和编译产物放在该目录下：{family_workspace}\n"
        "要求：\n"
        "1. 优先只生成一份共享 benchmark 源码，并让同一 binary 支持多个 mode 或参数路径。\n"
        "2. 先成功编译一次，再尽量通过 compile_and_run_cuda_source(skip_compile=true, program_args=[...]) 复用同一 binary 多次运行。\n"
        "3. 至少一次运行必须带 profile_with_ncu=true。\n"
        "4. 每次运行都应打印 `target_name: numeric_value`。\n"
        "5. 最终只输出当前 family 的 JSON 结果，必须一次性覆盖该 family 内所有 targets。\n"
    )
    return prompt


def build_engine(
    *,
    console: Console,
    llm_client: OpenAILLMClient,
    max_tool_output_chars: int,
    max_runtime_seconds: float,
    family: str,
    family_targets: list[str],
    family_workspace: Path,
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
        completion_checker=build_family_completion_checker(
            family=family,
            family_targets=family_targets,
            family_workspace=family_workspace,
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


def analyze_family_execution(
    memory: ConversationMemory,
    family_workspace: Path,
    family_targets: list[str],
) -> dict[str, Any]:
    tool_results_by_id: dict[str, str] = {}
    for message in memory.messages:
        if message.get("role") == "tool":
            tool_results_by_id[message.get("tool_call_id", "")] = message.get("content", "")

    successful_sources: list[str] = []
    successful_binaries: list[str] = []
    successful_invocations: list[dict[str, Any]] = []
    target_runs: dict[str, list[dict[str, Any]]] = {target: [] for target in family_targets}
    compile_executions = 0
    ncu_runs = 0

    for message in memory.messages:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls", []) or []:
            function = tool_call.get("function", {})
            if function.get("name") != "compile_and_run_cuda_source":
                continue
            try:
                parsed_arguments = json.loads(function.get("arguments", ""))
            except json.JSONDecodeError:
                continue

            source_path = parsed_arguments.get("source_path", "")
            if not isinstance(source_path, str):
                continue
            resolved_source = _resolve_workspace_path(source_path, family_workspace)
            try:
                resolved_source.relative_to(family_workspace.resolve())
            except ValueError:
                continue

            tool_result = tool_results_by_id.get(tool_call.get("id", ""), "")
            if "[compile_and_run_cuda_source] status=ok" not in tool_result:
                continue

            tool_payload = extract_tool_json(tool_result)
            binary_path_value = tool_payload.get("binary_path", parsed_arguments.get("binary_path", ""))
            resolved_binary = _resolve_workspace_path(binary_path_value, family_workspace) if isinstance(binary_path_value, str) and binary_path_value else resolved_source.with_suffix("")
            try:
                resolved_binary.relative_to(family_workspace.resolve())
            except ValueError:
                continue

            compile_executed = bool(tool_payload.get("compile_executed", not bool(parsed_arguments.get("skip_compile", False))))
            if compile_executed:
                compile_executions += 1
            if bool(parsed_arguments.get("profile_with_ncu", False)):
                ncu_runs += 1

            program_args = normalize_program_args(tool_payload.get("program_args", parsed_arguments.get("program_args", [])))
            invocation = {
                "source_path": str(resolved_source),
                "binary_path": str(resolved_binary),
                "program_args": program_args,
                "compile_executed": compile_executed,
                "profile_with_ncu": bool(parsed_arguments.get("profile_with_ncu", False)),
            }
            successful_invocations.append(invocation)
            successful_sources.append(str(resolved_source))
            successful_binaries.append(str(resolved_binary))

            merged_results: dict[str, float] = {}
            for key in ("run_stdout", "profile_stdout", "run_stderr", "profile_stderr"):
                value = tool_payload.get(key, "")
                if isinstance(value, str) and value.strip():
                    merged_results.update(extract_numeric_targets_from_text(value, family_targets))
            for target, value in merged_results.items():
                target_runs[target].append({**invocation, "value": value})

    unique_program_args = {json.dumps(item["program_args"], ensure_ascii=False) for item in successful_invocations}
    return {
        "compile_executions": compile_executions,
        "ncu_runs": ncu_runs,
        "successful_sources": successful_sources,
        "successful_binaries": successful_binaries,
        "successful_invocations": successful_invocations,
        "successful_invocation_count": len(successful_invocations),
        "reused_binary_runs": sum(1 for item in successful_invocations if not item["compile_executed"]),
        "unique_binary_paths": sorted(set(successful_binaries)),
        "unique_program_args_count": len(unique_program_args),
        "shared_source_reused": len(set(successful_sources)) == 1 if successful_sources else False,
        "target_runs": target_runs,
    }


def _validate_result_path(
    *,
    path_value: Any,
    family_workspace: Path,
    field_name: str,
) -> tuple[Path | None, str | None]:
    if not isinstance(path_value, str) or not path_value.strip():
        return None, f"{field_name} 必须是非空字符串"
    resolved_path = _resolve_workspace_path(path_value, family_workspace)
    try:
        resolved_path.relative_to(family_workspace.resolve())
    except ValueError:
        return None, f"{field_name} 必须位于当前 family 工作目录内: {resolved_path}"
    return resolved_path, None


def validate_family_result(
    *,
    family: str,
    family_targets: list[str],
    result_dict: dict[str, Any],
    family_workspace: Path,
    execution_info: dict[str, Any],
) -> tuple[bool, str]:
    required_keys = {"family", "benchmark_source_path", "binary_path", "targets"}
    missing = required_keys - set(result_dict)
    if missing:
        return False, f"缺少必要字段: {sorted(missing)}"
    if result_dict["family"] != family:
        return False, f"family 字段不匹配: 期望 {family}，实际 {result_dict['family']}"

    resolved_source, source_error = _validate_result_path(
        path_value=result_dict["benchmark_source_path"],
        family_workspace=family_workspace,
        field_name="benchmark_source_path",
    )
    if source_error:
        return False, source_error
    if resolved_source is not None and not resolved_source.exists():
        return False, f"benchmark_source_path 指向的文件不存在: {resolved_source}"

    resolved_binary, binary_error = _validate_result_path(
        path_value=result_dict["binary_path"],
        family_workspace=family_workspace,
        field_name="binary_path",
    )
    if binary_error:
        return False, binary_error
    if (
        resolved_binary is not None
        and execution_info["successful_binaries"]
        and str(resolved_binary) not in set(execution_info["successful_binaries"])
    ):
        return False, "binary_path 没有出现在成功执行记录中"

    targets_payload = result_dict["targets"]
    if not isinstance(targets_payload, dict):
        return False, "targets 必须是对象"
    missing_targets = [target for target in family_targets if target not in targets_payload]
    extra_targets = sorted(set(targets_payload) - set(family_targets))
    if missing_targets:
        return False, f"targets 缺少 family 内目标: {missing_targets}"
    if extra_targets:
        return False, f"targets 出现了不属于当前 family 的条目: {extra_targets}"

    if execution_info["compile_executions"] < 1:
        return False, "当前 family 尚未完成成功编译"
    if execution_info["ncu_runs"] < 1:
        return False, "当前 family 尚未完成至少一次带 ncu 的成功 profiling"

    if len(family_targets) > 1:
        if execution_info["successful_invocation_count"] < len(family_targets):
            return False, "当前 family 的成功运行次数不足，尚未覆盖全部 target 的独立测量"
        if execution_info["reused_binary_runs"] < 1:
            return False, "当前 family 尚未体现编译一次后复用 binary 多次运行"
        if not execution_info["shared_source_reused"]:
            return False, "当前 family 的成功运行没有复用同一份 benchmark 源码"
        if len(execution_info["unique_binary_paths"]) != 1:
            return False, "当前 family 没有稳定复用同一个 binary"
        if execution_info["unique_program_args_count"] < 2:
            return False, "当前 family 尚未体现多个 mode 或不同 program_args 的运行"

    for target in family_targets:
        payload = targets_payload[target]
        if not isinstance(payload, dict):
            return False, f"target `{target}` 的结果必须是对象"
        required_target_keys = {"value", "mode", "program_args", "used_ncu_analysis", "evidence_summary"}
        missing_target_keys = required_target_keys - set(payload)
        if missing_target_keys:
            return False, f"target `{target}` 缺少字段: {sorted(missing_target_keys)}"
        if not isinstance(payload["value"], (int, float)):
            return False, f"target `{target}` 的 value 必须是数字"
        if not isinstance(payload["mode"], str) or not payload["mode"].strip():
            return False, f"target `{target}` 的 mode 必须是非空字符串"
        if not isinstance(payload["program_args"], list):
            return False, f"target `{target}` 的 program_args 必须是字符串数组"
        program_args = normalize_program_args(payload["program_args"])
        if not isinstance(payload["used_ncu_analysis"], bool) or not payload["used_ncu_analysis"]:
            return False, f"target `{target}` 的 used_ncu_analysis 必须为 true"
        if not isinstance(payload["evidence_summary"], str) or not payload["evidence_summary"].strip():
            return False, f"target `{target}` 的 evidence_summary 必须是非空字符串"

        observed_runs = execution_info["target_runs"].get(target, [])
        if not observed_runs:
            return False, f"target `{target}` 没有在成功工具输出中出现机器可解析结果行"
        if not any(run["program_args"] == program_args for run in observed_runs):
            return False, f"target `{target}` 的 program_args 与实际成功运行记录不一致"

    return True, ""


def build_incomplete_feedback(
    *,
    family: str,
    family_targets: list[str],
    execution_info: dict[str, Any],
    validation_error: str | None,
) -> str:
    lines = [f"当前 family `{family}` 尚未满足结束条件。"]
    if validation_error:
        lines.append(f"现有输出问题: {validation_error}")
    if execution_info["compile_executions"] < 1:
        lines.append("还没有成功完成共享 benchmark 的编译。")
    if execution_info["ncu_runs"] < 1:
        lines.append("还没有成功执行带 ncu 的 profiling。")
    missing_targets = [target for target in family_targets if not execution_info["target_runs"].get(target)]
    if missing_targets:
        lines.append(f"以下 target 还没有机器可解析结果: {missing_targets}。")
    if len(family_targets) > 1 and execution_info["reused_binary_runs"] < 1:
        lines.append("请在首次成功编译后使用 skip_compile=true 复用同一 binary。")
    if len(family_targets) > 1 and execution_info["unique_program_args_count"] < 2:
        lines.append("请至少用两组不同的 mode 或 program_args 运行共享 benchmark。")
    if execution_info["compile_executions"] >= 1 and execution_info["ncu_runs"] >= 1 and not missing_targets:
        lines.append("实验记录接近完成，请整理并返回最终 family JSON。")
    else:
        lines.append("请继续补齐共享 benchmark 的多 mode 运行、结果打印与 ncu 分析，然后输出最终 JSON。")
    return " ".join(lines)


def synthesize_family_result_from_execution(
    *,
    family: str,
    family_targets: list[str],
    execution_info: dict[str, Any],
) -> dict[str, Any] | None:
    if execution_info["compile_executions"] < 1 or execution_info["ncu_runs"] < 1:
        return None
    if not execution_info["successful_sources"] or not execution_info["successful_binaries"]:
        return None

    targets_payload: dict[str, Any] = {}
    for target in family_targets:
        runs = execution_info["target_runs"].get(target, [])
        if not runs:
            return None
        latest_run = runs[-1]
        targets_payload[target] = {
            "value": float(latest_run["value"]),
            "mode": infer_mode_from_program_args(latest_run["program_args"], target),
            "program_args": latest_run["program_args"],
            "used_ncu_analysis": True,
            "evidence_summary": "Agent 未显式收尾，结果从共享 benchmark 的成功运行输出与 ncu 分析中自动提取。",
        }

    return {
        "family": family,
        "benchmark_source_path": execution_info["successful_sources"][-1],
        "binary_path": execution_info["successful_binaries"][-1],
        "targets": targets_payload,
    }


def build_family_completion_checker(*, family: str, family_targets: list[str], family_workspace: Path):
    def checker(memory: ConversationMemory, assistant_content: str) -> CompletionCheckResult:
        execution_info = analyze_family_execution(memory, family_workspace, family_targets)
        result_dict = extract_json(assistant_content)
        validation_error: str | None = None

        if result_dict:
            is_valid, validation_error = validate_family_result(
                family=family,
                family_targets=family_targets,
                result_dict=result_dict,
                family_workspace=family_workspace,
                execution_info=execution_info,
            )
            if is_valid:
                return CompletionCheckResult(is_complete=True, final_answer=assistant_content)

        synthesized_result = synthesize_family_result_from_execution(
            family=family,
            family_targets=family_targets,
            execution_info=execution_info,
        )
        if synthesized_result:
            is_valid, _ = validate_family_result(
                family=family,
                family_targets=family_targets,
                result_dict=synthesized_result,
                family_workspace=family_workspace,
                execution_info=execution_info,
            )
            if is_valid:
                return CompletionCheckResult(
                    is_complete=True,
                    final_answer=json.dumps(synthesized_result, ensure_ascii=False, indent=2),
                )

        feedback = build_incomplete_feedback(
            family=family,
            family_targets=family_targets,
            execution_info=execution_info,
            validation_error=validation_error if result_dict else None,
        )
        return CompletionCheckResult(is_complete=False, feedback=feedback)

    return checker


def is_engine_terminal_failure(message: str) -> bool:
    if not isinstance(message, str):
        return False
    prefixes = (
        "LLM 调用失败",
        "已达到时间上限",
        "执行被中止",
    )
    return message.startswith(prefixes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated Hardware Probe Evaluator")
    parser.add_argument("--target-spec", type=str, default="/target/target_spec.json", help="Path to target_spec.json")
    parser.add_argument("--output", type=str, default="/workspace/output.json", help="Path to write final output file")
    parser.add_argument("--skip-details-output", action="store_true", help="Do not emit the secondary details output file.")
    args = parser.parse_args()

    console = Console()
    spec_path = Path(args.target_spec).resolve()
    out_path = Path(args.output).resolve()
    project_root = Path(__file__).resolve().parent.parent
    generated_cuda_dir = (project_root / "generated_cuda").resolve()

    if not spec_path.exists():
        write_failure_output(
            console=console,
            out_path=out_path,
            title="File Not Found",
            message=f"Error: 找不到文件 {spec_path}",
            target_spec_path=str(spec_path),
        )
        return

    with open(spec_path, "r", encoding="utf-8") as f:
        try:
            target_spec = json.load(f)
        except json.JSONDecodeError as exc:
            write_failure_output(
                console=console,
                out_path=out_path,
                title="JSON Error",
                message=f"解析 {spec_path} 失败: {exc}",
                target_spec_path=str(spec_path),
            )
            return

    targets = target_spec.get("targets", [])
    run_executable = target_spec.get("run", "")
    if not isinstance(targets, list) or not targets:
        write_failure_output(
            console=console,
            out_path=out_path,
            title="Invalid Target Spec",
            message="target_spec.json 中没有找到非空的 targets 列表。",
            target_spec_path=str(spec_path),
        )
        return
    targets = [str(target) for target in targets]
    target_families = group_targets_by_family(targets)

    api_key = os.getenv("API_KEY")
    model = os.getenv("BASE_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5.4")
    base_url = os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not api_key:
        write_failure_output(
            console=console,
            out_path=out_path,
            title="Config Error",
            message="缺少 API_KEY，无法启动评测 Agent。",
            target_spec_path=str(spec_path),
        )
        return

    max_tool_output_chars = int(os.getenv("AGENT_MAX_TOOL_CHARS", "6000"))
    total_runtime_seconds = float(os.getenv("AGENT_MAX_TOTAL_RUNTIME_SECONDS", "2100"))
    llm_client = OpenAILLMClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
    )
    generated_cuda_dir.mkdir(parents=True, exist_ok=True)

    family_overview = "\n".join(
        f"- {family}: {', '.join(family_targets)}"
        for family, family_targets in target_families
    )
    console.print(
        Panel(
            f"评测目标加载完成，共 {len(targets)} 项，分成 {len(target_families)} 个 family。\n"
            f"自生成 CUDA 根目录: {generated_cuda_dir}\n"
            f"总时间预算: {total_runtime_seconds / 60:.1f} 分钟\n"
            f"run 字段将被忽略: {run_executable or '未提供'}\n"
            f"Family 划分:\n{family_overview}",
            title="Evaluator Started",
            border_style="cyan",
        )
    )

    aggregated_results: dict[str, float] = {}
    detailed_results: dict[str, Any] = {"families": {}, "targets": {}, "ignored_run": run_executable}
    evaluation_start_time = time.monotonic()
    evaluation_deadline = evaluation_start_time + total_runtime_seconds

    for index, (family, family_targets) in enumerate(target_families, start=1):
        remaining_total_seconds = evaluation_deadline - time.monotonic()
        if remaining_total_seconds <= 0:
            failure_payload = {
                "error": f"总测试时间已超过 {total_runtime_seconds:.1f}s，在完成 family `{family}` 前停止。",
                "family": family,
                "partial_results": aggregated_results,
                "ignored_run": run_executable,
                "elapsed_seconds": time.monotonic() - evaluation_start_time,
                "time_budget_seconds": total_runtime_seconds,
            }
            write_json_output(out_path, failure_payload)
            console.print(Panel(failure_payload["error"], title="Evaluation Timed Out", border_style="red"))
            return

        family_workspace = generated_cuda_dir / f"{index:02d}_{slugify_target(family)}"
        family_workspace.mkdir(parents=True, exist_ok=True)
        console.print(
            Panel.fit(
                f"Family {index}/{len(target_families)}: {family}\n"
                f"Targets: {', '.join(family_targets)}\n"
                f"Workspace: {family_workspace}\n"
                f"Remaining total time: {remaining_total_seconds / 60:.1f} min",
                title="Family Orchestrator",
                border_style="blue",
            )
        )

        engine = build_engine(
            console=console,
            llm_client=llm_client,
            max_tool_output_chars=max_tool_output_chars,
            max_runtime_seconds=remaining_total_seconds,
            family=family,
            family_targets=family_targets,
            family_workspace=family_workspace,
        )
        prompt = build_family_prompt(family=family, family_targets=family_targets, family_workspace=family_workspace)
        final_answer = engine.run(prompt)
        if is_engine_terminal_failure(final_answer):
            failure_payload = {
                "error": f"Family `{family}` 执行失败: {final_answer}",
                "family": family,
                "family_targets": family_targets,
                "raw_output": final_answer,
                "partial_results": aggregated_results,
                "ignored_run": run_executable,
            }
            write_json_output(out_path, failure_payload)
            console.print(Panel(failure_payload["error"], title="Evaluation Failed", border_style="red"))
            return
        result_dict = extract_json(final_answer)
        execution_info = analyze_family_execution(engine.memory, family_workspace, family_targets)

        if not result_dict:
            result_dict = synthesize_family_result_from_execution(
                family=family,
                family_targets=family_targets,
                execution_info=execution_info,
            ) or {}
            if result_dict:
                console.print(
                    Panel(
                        f"Family `{family}` 未显式输出最终 JSON，但已从共享 benchmark 的有效工具输出中自动提取结果。",
                        title="Fallback Extraction",
                        border_style="yellow",
                    )
                )

        is_valid, error_message = validate_family_result(
            family=family,
            family_targets=family_targets,
            result_dict=result_dict,
            family_workspace=family_workspace,
            execution_info=execution_info,
        )
        if not is_valid:
            failure_payload = {
                "error": f"Family `{family}` 的结果校验失败: {error_message}",
                "family": family,
                "family_targets": family_targets,
                "raw_output": final_answer,
                "parsed_output": result_dict,
                "execution_info": execution_info,
                "partial_results": aggregated_results,
                "ignored_run": run_executable,
            }
            write_json_output(out_path, failure_payload)
            console.print(Panel(failure_payload["error"], title="Evaluation Failed", border_style="red"))
            return

        family_target_details: dict[str, Any] = {}
        for target in family_targets:
            target_result = result_dict["targets"][target]
            value = float(target_result["value"])
            aggregated_results[target] = value
            target_detail = {
                "family": family,
                "value": value,
                "mode": target_result["mode"],
                "program_args": target_result["program_args"],
                "benchmark_source_path": result_dict["benchmark_source_path"],
                "binary_path": result_dict["binary_path"],
                "used_ncu_analysis": target_result["used_ncu_analysis"],
                "evidence_summary": target_result["evidence_summary"],
                "observed_runs": execution_info["target_runs"].get(target, []),
            }
            family_target_details[target] = target_detail
            detailed_results["targets"][target] = {**target_detail, "workspace": str(family_workspace), "raw_output": final_answer}

        detailed_results["families"][family] = {
            "targets": family_target_details,
            "benchmark_source_path": result_dict["benchmark_source_path"],
            "binary_path": result_dict["binary_path"],
            "execution_info": execution_info,
            "workspace": str(family_workspace),
            "raw_output": final_answer,
        }

    write_json_output(out_path, aggregated_results)
    if not args.skip_details_output:
        write_json_output(out_path.with_name(f"{out_path.stem}.details.json"), detailed_results)

    console.print(Panel(f"结果已成功写入: {out_path}", title="Evaluation Completed", border_style="green"))


if __name__ == "__main__":
    main()
