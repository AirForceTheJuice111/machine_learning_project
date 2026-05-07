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
from agent_framework.lora_artifacts import (
    archive_best_artifacts,
    ensure_materialized_best_report,
    ensure_phase2_layout,
    ensure_starter_candidate,
    load_best_report,
)
from agent_framework.lora_design_guidance import PHASE2_SYSTEM_PROMPT, build_phase2_prompt
from agent_framework.lora_search_policy import (
    analyze_search_readiness,
    choose_best_full_report,
    choose_best_report,
    report_score,
)
from agent_framework.tools.cuda_probe_tools import CompileAndRunCudaSourceTool
from agent_framework.tools.lora_candidate_tools import (
    EvaluateLoraCandidateTool,
    GenerateCudaCandidateFromPromptTool,
    PromoteLoraCandidateTool,
    ReviseCandidateTool,
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


def build_phase2_registry(*, project_root: Path, llm_client: OpenAILLMClient) -> ToolRegistry:
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
    registry.register(
        GenerateCudaCandidateFromPromptTool(
            default_project_root=project_root,
            llm_client=llm_client,
        )
    )
    registry.register(
        ReviseCandidateTool(
            default_project_root=project_root,
            llm_client=llm_client,
        )
    )
    registry.register(PromoteLoraCandidateTool(default_project_root=project_root))

    if os.getenv("PHASE2_ENABLE_PHASE1_PROBE", "").strip().lower() in {"1", "true", "yes", "on"}:
        registry.register(CompileAndRunCudaSourceTool(default_project_root=project_root))

    return registry


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
    generation_events: list[dict[str, Any]] = []
    revision_events: list[dict[str, Any]] = []
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
            elif (
                tool_name == "generate_cuda_candidate_from_prompt"
                and "[generate_cuda_candidate_from_prompt] status=ok" in tool_result
            ):
                payload = extract_tool_json(tool_result)
                if payload:
                    generation_events.append(payload)
            elif tool_name == "revise_candidate" and "[revise_candidate] status=ok" in tool_result:
                payload = extract_tool_json(tool_result)
                if payload:
                    revision_events.append(payload)
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
    generation_prompt_ids = [
        str(event.get("prompt_id") or "").strip()
        for event in generation_events
        if str(event.get("prompt_id") or "").strip()
    ]

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
        "generation_events": generation_events,
        "generation_prompt_ids": generation_prompt_ids,
        "revision_events": revision_events,
    }


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
    fallback_full_report = choose_best_full_report(execution_state.get("candidate_reports", []))
    search_state = analyze_search_readiness(
        project_root=project_root,
        execution_state=execution_state,
        best_report=best_report,
    )

    if not optimized_path.exists():
        issues.append("提交根目录下仍缺少 optimized_lora.cu")
    if execution_state["unique_candidates"] < 2:
        issues.append("当前还没有比较至少 2 个候选实现")
    if execution_state["promotions"] < 1 and fallback_full_report is None:
        issues.append("当前还没有通过 promote_lora_candidate 固化 best 版本")
    if best_report is None:
        if fallback_full_report is None:
            issues.append("当前还没有 best_report.json，说明 best 尚未完成正式评测记录")
    else:
        if not best_report.get("compile_ok"):
            issues.append("当前 best 最近一次评测未通过编译")
        if not best_report.get("correctness_passed"):
            issues.append("当前 best 最近一次评测未通过 correctness")
        if str(best_report.get("shape_preset", "")).lower() != "full":
            issues.append("当前 best 还没有完成 full 预设验证")

    if not search_state["search_stop_ready"]:
        remaining_prompts = [
            prompt_id
            for prompt_id in search_state["available_prompt_ids"]
            if prompt_id not in search_state["attempted_prompt_ids"]
        ]
        if remaining_prompts:
            issues.append(
                "当前仍有未覆盖的 generation prompt："
                + ", ".join(remaining_prompts)
            )
        if search_state["best_speedup"] < search_state["target_speedup"]:
            issues.append(
                f"当前 best 的 mean_speedup={search_state['best_speedup']:.4f}，仍低于建议目标 {search_state['target_speedup']:.2f}"
            )
        if search_state["stalled_rounds"] < search_state["stalled_round_limit"]:
            issues.append(
                "连续无显著提升轮数仍不足结束阈值 "
                f"{search_state['stalled_round_limit']}（当前为 {search_state['stalled_rounds']}）"
            )

    if not issues:
        return (
            "工程状态已经满足结束条件。请不要继续调用工具，只输出最终总结 JSON，"
            "包含 optimized_lora_path、best_candidate_path、evaluated_candidates、promotions、"
            "best_shape_preset、correctness_passed、mean_speedup、min_speedup、used_hardware_probes、"
            "best_report_path、best_report_score。"
        )
    if fallback_full_report is not None and best_report is None:
        report_path = fallback_full_report.get("report_path", "")
        source_path = fallback_full_report.get("source_path", "")
        return (
            "当前已经存在通过 full 验证的最佳候选，但还没有固化到 best。"
            f"请立刻对 source_path={source_path}、report_path={report_path} 执行一次 promote_lora_candidate，"
            "然后停止继续搜索，只输出最终总结 JSON。"
        )
    if search_state["next_prompt_id"] is not None:
        next_step = (
            f"建议下一步优先调用 generate_cuda_candidate_from_prompt，直接传 "
            f"prompt_id={search_state['next_prompt_id']} 与新的 candidate_name；"
            "若编译失败或 correctness 失败，再调用 revise_candidate。"
        )
    else:
        next_step = "建议继续根据最新失败日志或性能瓶颈调用 revise_candidate，并在必要时补齐 full 验证。"
    joined = "；".join(issues)
    return f"phase2 尚未满足结束条件：{joined}。{next_step}"


def build_phase2_completion_checker(*, project_root: Path):
    optimized_path = project_root / "optimized_lora.cu"

    def checker(memory: ConversationMemory, assistant_content: str) -> CompletionCheckResult:
        execution_state = analyze_phase2_execution(memory)
        best_report = load_best_report(project_root)
        best_report, auto_promoted = ensure_materialized_best_report(
            project_root=project_root,
            execution_state=execution_state,
            best_report=best_report,
        )
        summary = extract_json(assistant_content)
        search_state = analyze_search_readiness(
            project_root=project_root,
            execution_state=execution_state,
            best_report=best_report,
        )

        meets_engineering_requirements = (
            optimized_path.exists()
            and execution_state["unique_candidates"] >= 2
            and (execution_state["promotions"] >= 1 or auto_promoted or best_report is not None)
            and best_report is not None
            and bool(best_report.get("compile_ok"))
            and bool(best_report.get("correctness_passed"))
            and str(best_report.get("shape_preset", "")).lower() == "full"
            and search_state["search_stop_ready"]
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


def classify_engine_terminal_failure(message: str) -> str:
    if not isinstance(message, str):
        return "unknown"
    if message.startswith("LLM 调用失败"):
        return "llm_error"
    if message.startswith("已达到时间上限"):
        return "timeout"
    if message.startswith("执行被中止"):
        return "aborted"
    return "unknown"


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
        tool_registry=build_phase2_registry(
            project_root=project_root,
            llm_client=llm_client,
        ),
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
                f"时间预算: {args.time_budget_seconds / 60.0:.1f} 分钟\n"
                "说明: 默认不再清空旧候选与日志，便于复盘与调试。"
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
    best_report, auto_promoted = ensure_materialized_best_report(
        project_root=project_root,
        execution_state=execution_state,
        best_report=best_report,
    )

    if is_engine_terminal_failure(final_answer):
        failure_kind = classify_engine_terminal_failure(final_answer)
        recovered_full_best = (
            best_report is not None
            and bool(best_report.get("compile_ok"))
            and bool(best_report.get("correctness_passed"))
            and str(best_report.get("shape_preset", "")).lower() == "full"
        )
        if recovered_full_best:
            best_report, archive_info = archive_best_artifacts(project_root=project_root, best_report=best_report)
            recovered_summary = synthesize_phase2_summary(
                project_root=project_root,
                execution_state=execution_state,
                best_report=best_report,
            )
            recovered_summary["raw_error"] = final_answer
            recovered_summary["recovered_from_timeout"] = True
            recovered_summary["auto_promoted"] = auto_promoted or bool(best_report.get("auto_promoted"))
            if archive_info is not None:
                recovered_summary.update(archive_info)
            summary_path.write_text(json.dumps(recovered_summary, indent=2, ensure_ascii=False), encoding="utf-8")
            if failure_kind == "timeout":
                recovered_title = "Phase2 Recovered"
                recovered_message = "达到时间上限，但系统已基于通过 full 验证的最佳候选自动固化 best 并写出 summary。"
            elif failure_kind == "llm_error":
                recovered_title = "Phase2 LLM Recovered"
                recovered_message = "LLM 调用失败，但系统已基于通过 full 验证的最佳候选自动固化 best 并写出 summary。"
            else:
                recovered_title = "Phase2 Recovered"
                recovered_message = "主循环提前结束，但系统已基于通过 full 验证的最佳候选自动固化 best 并写出 summary。"
            console.print(
                Panel(
                    recovered_message,
                    title=recovered_title,
                    border_style="yellow",
                )
            )
            return
        failure_payload = {
            "error": final_answer,
            "optimized_lora_path": str((project_root / "optimized_lora.cu").resolve()),
            "best_report_path": str((layout["best_dir"] / "best_report.json").resolve()),
        }
        summary_path.write_text(json.dumps(failure_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(Panel(final_answer, title="Phase2 Failed", border_style="red"))
        return

    summary = extract_json(final_answer)
    best_report, archive_info = archive_best_artifacts(project_root=project_root, best_report=best_report)
    if summary:
        synthesized = synthesize_phase2_summary(
            project_root=project_root,
            execution_state=execution_state,
            best_report=best_report,
        )
        merged_summary = {**synthesized, **summary}
        if archive_info is not None:
            merged_summary.update(archive_info)
        summary_path.write_text(json.dumps(merged_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        fallback_summary = synthesize_phase2_summary(
            project_root=project_root,
            execution_state=execution_state,
            best_report=best_report,
        )
        fallback_summary["raw_output"] = final_answer
        if archive_info is not None:
            fallback_summary.update(archive_info)
        summary_path.write_text(json.dumps(fallback_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    console.print(Panel(f"phase2 已结束，当前 best 文件: {optimized_path}", title="Phase2 Completed", border_style="green"))


if __name__ == "__main__":
    main()
