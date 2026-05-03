from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from agent_framework.core.engine import AgentEngine, CompletionCheckResult
from agent_framework.core.llm_client import OpenAILLMClient
from agent_framework.core.memory import ConversationMemory
from agent_framework.lora_candidate_templates import write_seed_candidate_templates
from agent_framework.lora_design_guidance import PHASE2_SYSTEM_PROMPT, build_phase2_prompt
from agent_framework.lora_harness import starter_optimized_lora_source
from agent_framework.tools.cuda_probe_tools import CompileAndRunCudaSourceTool
from agent_framework.tools.lora_candidate_tools import (
    EvaluateLoraCandidateTool,
    PromoteLoraCandidateTool,
)
from agent_framework.tools.system_tools import ReadFileTool, WriteFileTool
from agent_framework.tools.tool_registry import ToolRegistry


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


def build_phase2_registry(console: Console, *, project_root: Path) -> ToolRegistry:
    def approval_handler(prompt: str) -> bool:
        return True

    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(
        WriteFileTool(
            approval_handler=approval_handler,
            require_approval=False,
        )
    )
    registry.register(EvaluateLoraCandidateTool(default_project_root=project_root))
    registry.register(PromoteLoraCandidateTool(default_project_root=project_root))
    registry.register(CompileAndRunCudaSourceTool(default_project_root=project_root))
    return registry


def ensure_phase2_layout(project_root: Path) -> dict[str, Path]:
    workspace_root = project_root / "lora_workspace"
    candidates_dir = workspace_root / "candidates"
    logs_dir = workspace_root / "logs"
    best_dir = workspace_root / "best"
    build_dir = workspace_root / "build"
    for path in (workspace_root, candidates_dir, logs_dir, best_dir, build_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "workspace_root": workspace_root,
        "candidates_dir": candidates_dir,
        "logs_dir": logs_dir,
        "best_dir": best_dir,
        "build_dir": build_dir,
    }


def ensure_starter_candidate(project_root: Path, *, candidates_dir: Path) -> tuple[Path, dict[str, str]]:
    optimized_path = project_root / "optimized_lora.cu"
    if not optimized_path.exists() or not optimized_path.read_text(encoding="utf-8").strip():
        optimized_path.write_text(starter_optimized_lora_source(), encoding="utf-8")
    seed_candidate_path = candidates_dir / "seed_baseline.cu"
    if not seed_candidate_path.exists():
        seed_candidate_path.write_text(optimized_path.read_text(encoding="utf-8"), encoding="utf-8")
    seed_info = write_seed_candidate_templates(candidates_dir)
    return optimized_path, seed_info


def _last_assistant_has_tool_calls(memory: ConversationMemory) -> bool:
    for message in reversed(memory.messages):
        if message.get("role") == "assistant":
            return bool(message.get("tool_calls"))
    return False


def analyze_phase2_execution(memory: ConversationMemory) -> dict[str, Any]:
    tool_results_by_id: dict[str, str] = {}
    for message in memory.messages:
        if message.get("role") == "tool":
            tool_results_by_id[message.get("tool_call_id", "")] = message.get("content", "")

    candidate_reports: list[dict[str, Any]] = []
    promotions: list[dict[str, Any]] = []
    hardware_probe_runs = 0

    for message in memory.messages:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls", []) or []:
            function = tool_call.get("function", {})
            tool_name = function.get("name")
            tool_result = tool_results_by_id.get(tool_call.get("id", ""), "")
            if tool_name == "evaluate_lora_candidate" and "[evaluate_lora_candidate] status=ok" in tool_result:
                payload = extract_tool_json(tool_result)
                if payload:
                    candidate_reports.append(payload)
            elif tool_name == "promote_lora_candidate" and "[promote_lora_candidate] status=ok" in tool_result:
                payload = extract_tool_json(tool_result)
                if payload:
                    promotions.append(payload)
            elif tool_name == "compile_and_run_cuda_source" and "[compile_and_run_cuda_source] status=ok" in tool_result:
                hardware_probe_runs += 1

    compile_ok_reports = [report for report in candidate_reports if report.get("compile_ok")]
    passing_reports = [report for report in candidate_reports if report.get("correctness_passed")]
    full_reports = [report for report in passing_reports if str(report.get("shape_preset", "")).lower() == "full"]
    unique_candidates = {
        report.get("candidate_name") or Path(str(report.get("source_path", "candidate.cu"))).stem
        for report in candidate_reports
    }
    best_seen_report = choose_best_report(candidate_reports)

    return {
        "candidate_reports": candidate_reports,
        "evaluated_candidates": len(candidate_reports),
        "unique_candidates": len({name for name in unique_candidates if name}),
        "compile_ok_reports": len(compile_ok_reports),
        "passing_reports": len(passing_reports),
        "full_reports": len(full_reports),
        "promotions": len(promotions),
        "hardware_probe_runs": hardware_probe_runs,
        "best_seen_report": best_seen_report,
    }


def load_best_report(project_root: Path) -> dict[str, Any] | None:
    best_report_path = project_root / "lora_workspace" / "best" / "best_report.json"
    if not best_report_path.exists():
        return None
    try:
        return json.loads(best_report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def report_score(report: dict[str, Any]) -> float:
    value = report.get("score")
    if isinstance(value, (int, float)):
        return float(value)

    if not bool(report.get("compile_ok")):
        return -1_000_000_000.0
    if not bool(report.get("correctness_passed")):
        return -100_000_000.0

    shape_bonus = 1_000_000.0 if str(report.get("shape_preset", "")).lower() == "full" else 0.0
    mean_speedup = float(report.get("mean_speedup") or 0.0)
    min_speedup = float(report.get("min_speedup") or 0.0)
    mean_student_ms = float(report.get("mean_student_ms") or 1e9)
    return shape_bonus + mean_speedup * 10_000.0 + min_speedup * 1_000.0 - mean_student_ms


def choose_best_report(reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not reports:
        return None
    eligible_reports = [report for report in reports if report.get("compile_ok") and report.get("correctness_passed")]
    pool = eligible_reports or reports
    return max(pool, key=report_score)


def synthesize_phase2_summary(
    *,
    project_root: Path,
    execution_state: dict[str, Any],
    best_report: dict[str, Any] | None,
) -> dict[str, Any]:
    effective_best = best_report or execution_state.get("best_seen_report")
    best_candidate_path = None
    best_shape_preset = None
    correctness_passed = False
    mean_speedup = 0.0
    min_speedup = 0.0
    best_report_path = str((project_root / "lora_workspace" / "best" / "best_report.json").resolve())
    best_report_score = None
    if effective_best:
        best_candidate_path = effective_best.get("source_path")
        best_shape_preset = effective_best.get("shape_preset")
        correctness_passed = bool(effective_best.get("correctness_passed"))
        mean_speedup = float(effective_best.get("mean_speedup") or 0.0)
        min_speedup = float(effective_best.get("min_speedup") or 0.0)
        best_report_path = str(effective_best.get("report_path") or best_report_path)
        best_report_score = report_score(effective_best)

    return {
        "optimized_lora_path": str((project_root / "optimized_lora.cu").resolve()),
        "best_candidate_path": best_candidate_path or str((project_root / "optimized_lora.cu").resolve()),
        "evaluated_candidates": execution_state["evaluated_candidates"],
        "promotions": execution_state["promotions"],
        "best_shape_preset": best_shape_preset or "unknown",
        "correctness_passed": correctness_passed,
        "mean_speedup": mean_speedup,
        "min_speedup": min_speedup,
        "used_hardware_probes": execution_state["hardware_probe_runs"],
        "best_report_path": best_report_path,
        "best_report_score": best_report_score,
    }


def build_completion_feedback(*, project_root: Path, execution_state: dict[str, Any], best_report: dict[str, Any] | None) -> str:
    issues: list[str] = []
    optimized_path = project_root / "optimized_lora.cu"

    if not optimized_path.exists():
        issues.append("提交根目录下仍缺少 optimized_lora.cu")
    if execution_state["unique_candidates"] < 2:
        issues.append("当前还没有比较至少 2 个候选实现")
    if execution_state["promotions"] < 1:
        issues.append("当前还没有通过 promote_lora_candidate 固化 best 版本")
    if best_report is None:
        issues.append("当前还没有 best_report.json，说明 best 尚未完成正式评测记录")
    else:
        if not best_report.get("compile_ok"):
            issues.append("当前 best 最近一次评测未通过编译")
        if not best_report.get("correctness_passed"):
            issues.append("当前 best 最近一次评测未通过 correctness")
        if str(best_report.get("shape_preset", "")).lower() != "full":
            issues.append("当前 best 还没有完成 full 预设验证")

    if not issues:
        return (
            "工程状态已经满足结束条件。请不要继续调用工具，只输出最终总结 JSON，"
            "包含 optimized_lora_path、best_candidate_path、evaluated_candidates、promotions、"
            "best_shape_preset、correctness_passed、mean_speedup、min_speedup、used_hardware_probes、"
            "best_report_path、best_report_score。"
        )
    joined = "；".join(issues)
    return (
        f"phase2 尚未满足结束条件：{joined}。"
        "请继续按照候选搜索策略推进：评估基线、比较候选、必要时晋升 best，并在收尾前完成 full 验证。"
    )


def build_phase2_completion_checker(*, project_root: Path):
    optimized_path = project_root / "optimized_lora.cu"

    def checker(memory: ConversationMemory, assistant_content: str) -> CompletionCheckResult:
        execution_state = analyze_phase2_execution(memory)
        best_report = load_best_report(project_root)
        summary = extract_json(assistant_content)

        meets_engineering_requirements = (
            optimized_path.exists()
            and execution_state["unique_candidates"] >= 2
            and execution_state["promotions"] >= 1
            and best_report is not None
            and bool(best_report.get("compile_ok"))
            and bool(best_report.get("correctness_passed"))
            and str(best_report.get("shape_preset", "")).lower() == "full"
        )

        if meets_engineering_requirements and not _last_assistant_has_tool_calls(memory):
            required_summary_keys = {
                "optimized_lora_path",
                "best_candidate_path",
                "evaluated_candidates",
                "promotions",
                "best_shape_preset",
                "correctness_passed",
                "mean_speedup",
                "min_speedup",
                "used_hardware_probes",
                "best_report_path",
                "best_report_score",
            }
            if summary and required_summary_keys.issubset(summary):
                return CompletionCheckResult(is_complete=True, final_answer=assistant_content)
            feedback = (
                "工程状态已满足结束条件。请不要再调用工具，"
                "只输出最终总结 JSON，并补齐要求字段。"
            )
            return CompletionCheckResult(is_complete=False, feedback=feedback)

        feedback = build_completion_feedback(
            project_root=project_root,
            execution_state=execution_state,
            best_report=best_report,
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
    parser = argparse.ArgumentParser(description="Phase2 LoRA Optimization Agent")
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path.cwd()),
        help="Phase2 项目根目录；官方评测会在提交根目录运行。",
    )
    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        default=float(os.getenv("PHASE2_MAX_RUNTIME_SECONDS", "1740")),
        help="Phase2 Agent 总时间预算，默认 29 分钟。",
    )
    args = parser.parse_args()

    console = Console()
    project_root = Path(args.project_root).resolve()
    layout = ensure_phase2_layout(project_root)
    optimized_path, seed_info = ensure_starter_candidate(project_root, candidates_dir=layout["candidates_dir"])

    api_key = os.getenv("API_KEY")
    model = os.getenv("BASE_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5.4")
    base_url = os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not api_key:
        console.print(Panel("缺少 API_KEY，无法启动 phase2 Agent。", title="Configuration Error", border_style="red"))
        return

    llm_client = OpenAILLMClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
    )
    memory = ConversationMemory(
        system_prompt=PHASE2_SYSTEM_PROMPT,
        max_tool_output_chars=int(os.getenv("AGENT_MAX_TOOL_CHARS", "8000")),
    )
    engine = AgentEngine(
        llm_client=llm_client,
        tool_registry=build_phase2_registry(console, project_root=project_root),
        memory=memory,
        console=console,
        max_runtime_seconds=args.time_budget_seconds,
        completion_checker=build_phase2_completion_checker(project_root=project_root),
    )

    console.print(
        Panel(
            (
                f"Phase2 LoRA 优化已启动。\n"
                f"项目根目录: {project_root}\n"
                f"当前 baseline: {optimized_path}\n"
                f"候选目录: {layout['candidates_dir']}\n"
                f"模板清单: {seed_info['manifest_path']}\n"
                f"日志目录: {layout['logs_dir']}\n"
                f"时间预算: {args.time_budget_seconds / 60.0:.1f} 分钟"
            ),
            title="Phase2 Optimizer",
            border_style="cyan",
        )
    )

    prompt = build_phase2_prompt(
        project_root=project_root,
        time_budget_seconds=args.time_budget_seconds,
        seed_manifest_path=Path(seed_info["manifest_path"]),
    )
    final_answer = engine.run(prompt)
    summary_path = layout["logs_dir"] / "final_summary.json"
    execution_state = analyze_phase2_execution(engine.memory)
    best_report = load_best_report(project_root)

    if is_engine_terminal_failure(final_answer):
        failure_payload = {
            "error": final_answer,
            "optimized_lora_path": str((project_root / "optimized_lora.cu").resolve()),
            "best_report_path": str((layout["best_dir"] / "best_report.json").resolve()),
        }
        summary_path.write_text(json.dumps(failure_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(Panel(final_answer, title="Phase2 Failed", border_style="red"))
        return

    summary = extract_json(final_answer)
    if summary:
        synthesized = synthesize_phase2_summary(
            project_root=project_root,
            execution_state=execution_state,
            best_report=best_report,
        )
        merged_summary = {**synthesized, **summary}
        summary_path.write_text(json.dumps(merged_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        fallback_summary = synthesize_phase2_summary(
            project_root=project_root,
            execution_state=execution_state,
            best_report=best_report,
        )
        fallback_summary["raw_output"] = final_answer
        summary_path.write_text(json.dumps(fallback_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    console.print(Panel(f"phase2 已结束，当前 best 文件: {optimized_path}", title="Phase2 Completed", border_style="green"))


if __name__ == "__main__":
    main()
