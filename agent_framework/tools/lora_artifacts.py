from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_framework.lora_candidate_templates import write_seed_candidate_templates
from agent_framework.lora_harness import starter_optimized_lora_source
from agent_framework.lora_search_policy import choose_best_full_report


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


def allocate_best_archive_id(best_dir: Path) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sequence = 1
    while True:
        archive_id = f"best_{timestamp}_{sequence:03d}"
        archive_source_path = best_dir / f"{archive_id}_optimized_lora.cu"
        archive_report_path = best_dir / f"{archive_id}_report.json"
        if not archive_source_path.exists() and not archive_report_path.exists():
            return archive_id
        sequence += 1


def update_best_manifest(*, best_dir: Path, archived_report: dict[str, Any]) -> str:
    manifest_path = best_dir / "manifest.json"
    manifest: dict[str, Any]
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    else:
        manifest = {}

    archives = manifest.get("archives")
    if not isinstance(archives, list):
        archives = []

    archive_id = str(archived_report.get("archive_id") or "").strip()
    archives = [
        entry
        for entry in archives
        if not isinstance(entry, dict) or str(entry.get("archive_id") or "").strip() != archive_id
    ]

    archive_entry = {
        "archive_id": archive_id,
        "archived_at": archived_report.get("archived_at"),
        "optimized_lora_path": archived_report.get("archive_source_path"),
        "report_path": archived_report.get("archive_report_path"),
        "source_candidate_path": archived_report.get("source_path"),
        "source_report_path": archived_report.get("source_report_path") or archived_report.get("report_path"),
        "shape_preset": archived_report.get("shape_preset"),
        "compile_ok": bool(archived_report.get("compile_ok")),
        "correctness_passed": bool(archived_report.get("correctness_passed")),
        "mean_speedup": float(archived_report.get("mean_speedup") or 0.0),
        "min_speedup": float(archived_report.get("min_speedup") or 0.0),
        "score": archived_report.get("score"),
    }
    archives.append(archive_entry)
    archives.sort(key=lambda entry: str(entry.get("archived_at") or ""))

    manifest_payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "archive_count": len(archives),
        "latest_archive_id": archive_id,
        "archives": archives,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(manifest_path.resolve())


def archive_best_artifacts(*, project_root: Path, best_report: dict[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    if best_report is None:
        return best_report, None

    best_dir = project_root / "lora_workspace" / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_source_path = best_dir / "optimized_lora_best.cu"
    best_report_path = best_dir / "best_report.json"
    optimized_path = project_root / "optimized_lora.cu"

    source_snapshot = best_source_path if best_source_path.exists() else optimized_path
    if not source_snapshot.exists() or not best_report_path.exists():
        return best_report, None

    archive_id = allocate_best_archive_id(best_dir)
    archive_source_path = best_dir / f"{archive_id}_optimized_lora.cu"
    archive_report_path = best_dir / f"{archive_id}_report.json"

    archived_report = dict(best_report)
    archived_report["archive_id"] = archive_id
    archived_report["archive_source_path"] = str(archive_source_path.resolve())
    archived_report["archive_report_path"] = str(archive_report_path.resolve())
    archived_report["archived_at"] = datetime.now().isoformat(timespec="seconds")

    shutil.copyfile(source_snapshot, archive_source_path)
    best_report_path.write_text(json.dumps(archived_report, indent=2, ensure_ascii=False), encoding="utf-8")
    archive_report_path.write_text(json.dumps(archived_report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_path = update_best_manifest(best_dir=best_dir, archived_report=archived_report)

    return archived_report, {
        "archive_id": archive_id,
        "archive_source_path": str(archive_source_path.resolve()),
        "archive_report_path": str(archive_report_path.resolve()),
        "archive_manifest_path": manifest_path,
    }


def ensure_starter_candidate(project_root: Path, *, candidates_dir: Path) -> tuple[Path, dict[str, str]]:
    optimized_path = project_root / "optimized_lora.cu"
    if not optimized_path.exists() or not optimized_path.read_text(encoding="utf-8").strip():
        optimized_path.write_text(starter_optimized_lora_source(), encoding="utf-8")
    seed_candidate_path = candidates_dir / "seed_baseline.cu"
    seed_candidate_path.write_text(optimized_path.read_text(encoding="utf-8"), encoding="utf-8")
    seed_info = write_seed_candidate_templates(candidates_dir)
    return optimized_path, seed_info


def load_best_report(project_root: Path) -> dict[str, Any] | None:
    best_report_path = project_root / "lora_workspace" / "best" / "best_report.json"
    if not best_report_path.exists():
        return None
    try:
        return json.loads(best_report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def materialize_best_from_report(project_root: Path, report: dict[str, Any]) -> dict[str, Any] | None:
    source_value = str(report.get("source_path") or "").strip()
    if not source_value:
        return None
    source_path = Path(source_value).resolve()
    if not source_path.exists():
        return None

    optimized_path = project_root / "optimized_lora.cu"
    best_dir = project_root / "lora_workspace" / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_source_path = best_dir / "optimized_lora_best.cu"
    best_report_path = best_dir / "best_report.json"

    shutil.copyfile(source_path, optimized_path)
    shutil.copyfile(source_path, best_source_path)

    materialized_report = dict(report)
    original_report_path = materialized_report.get("report_path")
    if original_report_path:
        materialized_report["source_report_path"] = str(original_report_path)
    materialized_report["report_path"] = str(best_report_path.resolve())
    materialized_report["auto_promoted"] = True
    best_report_path.write_text(json.dumps(materialized_report, indent=2, ensure_ascii=False), encoding="utf-8")
    return materialized_report


def ensure_materialized_best_report(*, project_root: Path, execution_state: dict[str, Any], best_report: dict[str, Any] | None) -> tuple[dict[str, Any] | None, bool]:
    if best_report is not None:
        return best_report, False
    fallback_report = choose_best_full_report(execution_state.get("candidate_reports", []))
    if fallback_report is None:
        return None, False
    materialized = materialize_best_from_report(project_root, fallback_report)
    return materialized, materialized is not None
