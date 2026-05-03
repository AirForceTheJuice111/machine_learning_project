from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from agent_framework.tools.tool_registry import ToolResult


def _slugify(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)
    cleaned = cleaned.strip("._-")
    return cleaned or "candidate"


@dataclass
class EvaluateLoraCandidateTool:
    default_project_root: Path
    default_timeout_seconds: int = 900

    name: str = "evaluate_lora_candidate"
    description: str = (
        "编译并评测单文件 optimized_lora CUDA 候选实现，返回正确性与 benchmark 结果。"
    )
    parameters_schema: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        self.parameters_schema = {
            "type": "object",
            "properties": {
                "source_path": {
                    "type": "string",
                    "description": "待评测的 .cu 候选文件路径。",
                },
                "candidate_name": {
                    "type": "string",
                    "description": "候选名称，用于命名日志文件。",
                },
                "shape_preset": {
                    "type": "string",
                    "enum": ["smoke", "quick", "full"],
                    "description": "预设形状集合。smoke 最快，full 最严格。",
                },
                "shape_values": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "可选，显式指定要评测的 d 值列表。",
                },
                "warmup": {
                    "type": "integer",
                    "description": "预热次数；默认按 preset 自动选择。",
                },
                "iters": {
                    "type": "integer",
                    "description": "计时轮数；默认按 preset 自动选择。",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "整个 harness 子进程超时，单位秒。",
                },
            },
            "required": ["source_path"],
            "additionalProperties": False,
        }

    def run(self, arguments: dict) -> ToolResult:
        source_path = self._resolve_within_project(arguments["source_path"])
        if source_path is None or source_path.suffix.lower() != ".cu":
            return ToolResult(status="rejected", content="source_path 必须是项目目录内的 .cu 文件。")
        if not source_path.exists():
            return ToolResult(status="error", content=f"候选源码不存在: {source_path}")

        candidate_name = _slugify(arguments.get("candidate_name") or source_path.stem)
        shape_preset = str(arguments.get("shape_preset", "quick"))
        timeout_seconds = int(arguments.get("timeout_seconds", self.default_timeout_seconds))
        report_dir = self.default_project_root / "lora_workspace" / "logs"
        build_root = self.default_project_root / "lora_workspace" / "build"
        report_path = report_dir / f"{candidate_name}_{int(time.time())}.json"

        command = [
            sys.executable,
            "-m",
            "agent_framework.lora_harness",
            "evaluate",
            "--source-path",
            str(source_path),
            "--build-root",
            str(build_root),
            "--shape-preset",
            shape_preset,
            "--report-path",
            str(report_path),
        ]

        shape_values = arguments.get("shape_values")
        if isinstance(shape_values, list) and shape_values:
            command.append("--shape-values")
            command.extend(str(int(value)) for value in shape_values)
        if arguments.get("warmup") is not None:
            command.extend(["--warmup", str(int(arguments["warmup"]))])
        if arguments.get("iters") is not None:
            command.extend(["--iters", str(int(arguments["iters"]))])

        try:
            completed = subprocess.run(
                command,
                cwd=str(self.default_project_root),
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=True,
            )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                status="timeout",
                content=(
                    f"LoRA 候选评测超时({timeout_seconds}s)。\n"
                    f"stdout:\n{exc.stdout or ''}\n"
                    f"stderr:\n{exc.stderr or ''}"
                ).strip(),
            )
        except subprocess.CalledProcessError as exc:
            return ToolResult(
                status="error",
                content=(
                    f"LoRA 候选评测进程失败，退出码: {exc.returncode}\n"
                    f"stdout:\n{exc.stdout or ''}\n"
                    f"stderr:\n{exc.stderr or ''}"
                ).strip(),
            )

        stdout = completed.stdout.strip()
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            return ToolResult(
                status="error",
                content=(
                    "LoRA harness 输出不是合法 JSON。\n"
                    f"stdout:\n{stdout}\n"
                    f"stderr:\n{completed.stderr.strip()}"
                ).strip(),
            )

        payload["candidate_name"] = candidate_name
        payload["stdout_truncated"] = False
        payload["decision_hint"] = (
            "eligible_for_promotion"
            if payload.get("compile_ok") and payload.get("correctness_passed")
            else "needs_fix_or_reject"
        )
        return ToolResult(status="ok", content=json.dumps(payload, indent=2, ensure_ascii=False))

    def _resolve_within_project(self, path_value: str) -> Path | None:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = (self.default_project_root / path).resolve()
        else:
            path = path.resolve()
        try:
            path.relative_to(self.default_project_root.resolve())
        except ValueError:
            return None
        return path


@dataclass
class PromoteLoraCandidateTool:
    default_project_root: Path

    name: str = "promote_lora_candidate"
    description: str = "将通过评测的候选 .cu 提升为提交根目录下的 optimized_lora.cu，并记录 best report。"
    parameters_schema: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        self.parameters_schema = {
            "type": "object",
            "properties": {
                "source_path": {
                    "type": "string",
                    "description": "待提升的候选 .cu 路径。",
                },
                "report_path": {
                    "type": "string",
                    "description": "对应 evaluate_lora_candidate 生成的 report JSON 路径。",
                },
                "reason": {
                    "type": "string",
                    "description": "可选，记录为何提升为 best。",
                },
            },
            "required": ["source_path"],
            "additionalProperties": False,
        }

    def run(self, arguments: dict) -> ToolResult:
        source_path = self._resolve_within_project(arguments["source_path"])
        if source_path is None or source_path.suffix.lower() != ".cu":
            return ToolResult(status="rejected", content="source_path 必须是项目目录内的 .cu 文件。")
        if not source_path.exists():
            return ToolResult(status="error", content=f"候选源码不存在: {source_path}")

        best_report_source = None
        report_value = arguments.get("report_path")
        if isinstance(report_value, str) and report_value.strip():
            report_path = self._resolve_within_project(report_value)
            if report_path is None or report_path.suffix.lower() != ".json":
                return ToolResult(status="rejected", content="report_path 必须是项目目录内的 .json 文件。")
            if not report_path.exists():
                return ToolResult(status="error", content=f"候选评测报告不存在: {report_path}")
            try:
                report_payload = json.loads(report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return ToolResult(status="error", content=f"候选评测报告不是合法 JSON: {report_path}")
            if not report_payload.get("compile_ok"):
                return ToolResult(status="rejected", content="禁止晋升未通过编译的候选。")
            if not report_payload.get("correctness_passed"):
                return ToolResult(status="rejected", content="禁止晋升未通过 correctness 的候选。")
            best_report_source = report_path

        optimized_path = self.default_project_root / "optimized_lora.cu"
        optimized_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, optimized_path)

        best_dir = self.default_project_root / "lora_workspace" / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        best_source_path = best_dir / "optimized_lora_best.cu"
        shutil.copyfile(source_path, best_source_path)

        best_report_path = None
        if best_report_source is not None:
            best_report_path = best_dir / "best_report.json"
            shutil.copyfile(best_report_source, best_report_path)

        payload = {
            "optimized_lora_path": str(optimized_path.resolve()),
            "best_source_snapshot": str(best_source_path.resolve()),
            "best_report_path": str(best_report_path.resolve()) if best_report_path else None,
            "reason": arguments.get("reason", ""),
            "validated_report": best_report_path is not None,
        }
        return ToolResult(status="ok", content=json.dumps(payload, indent=2, ensure_ascii=False))

    def _resolve_within_project(self, path_value: str) -> Path | None:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = (self.default_project_root / path_value).resolve()
        else:
            path = path.resolve()
        try:
            path.relative_to(self.default_project_root.resolve())
        except ValueError:
            return None
        return path
