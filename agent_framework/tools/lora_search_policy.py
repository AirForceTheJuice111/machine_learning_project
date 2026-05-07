from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agent_framework.lora_harness import compute_report_score

DEFAULT_TARGET_SPEEDUP = 1.10
DEFAULT_STALLED_ROUNDS = 3
DEFAULT_IMPROVEMENT_THRESHOLD = 0.02


def report_score(report: dict[str, Any]) -> float:
    return compute_report_score(report)


def choose_best_report(reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not reports:
        return None
    eligible_reports = [report for report in reports if report.get("compile_ok") and report.get("correctness_passed")]
    pool = eligible_reports or reports
    return max(pool, key=report_score)


def choose_best_full_report(reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    full_reports = [
        report
        for report in reports
        if report.get("compile_ok")
        and report.get("correctness_passed")
        and str(report.get("shape_preset", "")).lower() == "full"
    ]
    return choose_best_report(full_reports) if full_reports else None


def load_generation_prompt_ids(project_root: Path) -> list[str]:
    manifest_path = project_root / "lora_workspace" / "candidates" / "seed_manifest.json"
    if not manifest_path.exists():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    prompt_entries = payload.get("generation_prompts")
    if not isinstance(prompt_entries, list):
        return []
    return [
        str(entry.get("id") or "").strip()
        for entry in prompt_entries
        if isinstance(entry, dict) and str(entry.get("id") or "").strip()
    ]


def choose_next_generation_prompt_id(*, available_prompt_ids: list[str], used_prompt_ids: list[str]) -> str | None:
    used = {prompt_id for prompt_id in used_prompt_ids if prompt_id}
    for prompt_id in available_prompt_ids:
        if prompt_id not in used:
            return prompt_id
    return None


def count_recent_stalled_rounds(reports: list[dict[str, Any]], *, improvement_threshold: float) -> int:
    eligible_reports = [report for report in reports if report.get("compile_ok") and report.get("correctness_passed")]
    if len(eligible_reports) < 2:
        return 0

    best_speedup = float(eligible_reports[0].get("mean_speedup") or 0.0)
    stalled_rounds = 0
    for report in eligible_reports[1:]:
        current_speedup = float(report.get("mean_speedup") or 0.0)
        required_speedup = best_speedup * (1.0 + improvement_threshold)
        if current_speedup > required_speedup:
            best_speedup = current_speedup
            stalled_rounds = 0
            continue
        best_speedup = max(best_speedup, current_speedup)
        stalled_rounds += 1
    return stalled_rounds


def phase2_speedup_target() -> float:
    return float(os.getenv("PHASE2_TARGET_SPEEDUP", str(DEFAULT_TARGET_SPEEDUP)))


def phase2_stalled_round_limit() -> int:
    return int(os.getenv("PHASE2_STALLED_ROUNDS", str(DEFAULT_STALLED_ROUNDS)))


def phase2_improvement_threshold() -> float:
    return float(os.getenv("PHASE2_IMPROVEMENT_THRESHOLD", str(DEFAULT_IMPROVEMENT_THRESHOLD)))


def analyze_search_readiness(*, project_root: Path, execution_state: dict[str, Any], best_report: dict[str, Any] | None) -> dict[str, Any]:
    available_prompt_ids = load_generation_prompt_ids(project_root)
    used_prompt_ids = execution_state.get("generation_prompt_ids", [])
    attempted_prompt_ids = sorted({prompt_id for prompt_id in used_prompt_ids if prompt_id})
    all_generation_prompts_tried = bool(available_prompt_ids) and all(
        prompt_id in attempted_prompt_ids for prompt_id in available_prompt_ids
    )
    best_speedup = float(
        (best_report or execution_state.get("best_seen_report") or {}).get("mean_speedup") or 0.0
    )
    target_speedup = phase2_speedup_target()
    stalled_round_limit = phase2_stalled_round_limit()
    improvement_threshold = phase2_improvement_threshold()
    stalled_rounds = count_recent_stalled_rounds(
        execution_state.get("candidate_reports", []),
        improvement_threshold=improvement_threshold,
    )
    search_stop_ready = (
        best_speedup >= target_speedup
        or all_generation_prompts_tried
        or stalled_rounds >= stalled_round_limit
    )
    return {
        "available_prompt_ids": available_prompt_ids,
        "attempted_prompt_ids": attempted_prompt_ids,
        "all_generation_prompts_tried": all_generation_prompts_tried,
        "next_prompt_id": choose_next_generation_prompt_id(
            available_prompt_ids=available_prompt_ids,
            used_prompt_ids=attempted_prompt_ids,
        ),
        "stalled_rounds": stalled_rounds,
        "stalled_round_limit": stalled_round_limit,
        "improvement_threshold": improvement_threshold,
        "best_speedup": best_speedup,
        "target_speedup": target_speedup,
        "search_stop_ready": search_stop_ready,
    }
