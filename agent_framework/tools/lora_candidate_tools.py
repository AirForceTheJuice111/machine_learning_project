from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_framework.tools.tool_registry import ToolResult

if TYPE_CHECKING:
    from agent_framework.core.llm_client import OpenAILLMClient


def _slugify(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)
    cleaned = cleaned.strip("._-")
    return cleaned or "candidate"


def _candidate_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _resolve_workspace_root(workspace: str | Path) -> Path:
    workspace_root = Path(workspace).expanduser().resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "candidates").mkdir(parents=True, exist_ok=True)
    return workspace_root


def _extract_code_from_llm_response(response_text: str) -> str:
    text = response_text.strip()
    if not text:
        raise RuntimeError("LLM 返回了空内容，无法提取 CUDA 代码。")

    code_blocks = [
        match.group(1).strip()
        for match in re.finditer(r"```(?:[\w#+.-]+)?\s*\n(.*?)```", text, re.DOTALL)
        if match.group(1).strip()
    ]
    if not code_blocks:
        code_blocks = [text]

    preferred_blocks = [block for block in code_blocks if "torch::Tensor forward" in block]
    selected = max(preferred_blocks or code_blocks, key=len).strip()
    if not selected:
        raise RuntimeError("未能从 LLM 回复中提取出有效代码块。")
    return selected.rstrip() + "\n"


def _validate_generated_cuda_source(source_text: str) -> None:
    if "torch::Tensor forward" not in source_text:
        raise RuntimeError("生成的 CUDA 代码缺少 `torch::Tensor forward` 接口。")


def _write_candidate_source(*, workspace_root: Path, candidate_name: str, source_text: str) -> str:
    candidate_file_name = f"{_slugify(candidate_name)}_{_candidate_timestamp()}_{int(time.time() * 1000) % 1000:03d}.cu"
    candidate_path = workspace_root / "candidates" / candidate_file_name
    candidate_path.write_text(source_text, encoding="utf-8")
    return str(candidate_path.resolve())


def generate_cuda_candidate_from_prompt(
    prompt: str,
    candidate_name: str,
    llm_client: "OpenAILLMClient",
    workspace: str,
) -> str:
    """
    使用 LLM 根据给定 prompt 生成 CUDA 代码，保存到 {workspace}/candidates/ 下。
    返回生成文件的绝对路径。
    如果 LLM 返回空内容或代码中不包含 "torch::Tensor forward"，抛出异常。
    """

    workspace_root = _resolve_workspace_root(workspace)
    response_text = llm_client.chat(prompt).strip()
    source_text = _extract_code_from_llm_response(response_text)
    _validate_generated_cuda_source(source_text)
    return _write_candidate_source(
        workspace_root=workspace_root,
        candidate_name=candidate_name,
        source_text=source_text,
    )


def _build_revision_prompt(source_text: str, error_or_bottleneck: str) -> str:
    issue_summary = error_or_bottleneck.strip() or "Unknown issue"
    return f"""Here is a CUDA extension candidate for the LoRA forward pass.

It has the following issue:
{issue_summary}

Please fix or improve it and return the complete corrected single-file .cu code only.

Requirements:
- Keep the interface exactly as:
  torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B)
- Keep the file self-contained and compatible with torch.utils.cpp_extension.load
- Preserve PyTorch binding via PYBIND11_MODULE
- Do not return explanations outside the code block

Current CUDA code:
```cpp
{source_text}
```
"""


def revise_candidate(
    candidate_path: str,
    error_or_bottleneck: str,
    llm_client: "OpenAILLMClient",
    workspace: str,
    *,
    max_retries: int = 2,
) -> str:
    """
    读取 candidate_path 的当前 CUDA 代码，附上问题描述，
    请求 LLM 修复或改进，保存为新候选文件，返回路径。
    """

    source_path = Path(candidate_path).expanduser().resolve()
    if not source_path.exists():
        raise RuntimeError(f"待修复候选不存在: {source_path}")

    source_text = source_path.read_text(encoding="utf-8")
    revision_prompt = _build_revision_prompt(source_text, error_or_bottleneck)
    workspace_root = _resolve_workspace_root(workspace)

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response_text = llm_client.chat(revision_prompt).strip()
            revised_source = _extract_code_from_llm_response(response_text)
            _validate_generated_cuda_source(revised_source)
            return _write_candidate_source(
                workspace_root=workspace_root,
                candidate_name=f"{source_path.stem}_rev{attempt}",
                source_text=revised_source,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            revision_prompt = (
                f"{revision_prompt}\n\n"
                f"Previous revision attempt failed with this validation error:\n{exc}\n"
                "Please return the full corrected .cu file again."
            )

    raise RuntimeError(f"候选修复在 {max_retries} 次尝试后仍失败: {last_error}")


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
class GenerateCudaCandidateFromPromptTool:
    default_project_root: Path
    llm_client: "OpenAILLMClient"

    name: str = "generate_cuda_candidate_from_prompt"
    description: str = "调用 LLM 根据给定 generation prompt 生成新的 LoRA CUDA 候选文件。"
    parameters_schema: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        self.parameters_schema = {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "用于生成 CUDA 候选的完整提示词文本。",
                },
                "candidate_name": {
                    "type": "string",
                    "description": "候选名，用于生成文件名。",
                },
                "prompt_id": {
                    "type": "string",
                    "description": "可选，记录本次使用的 generation prompt ID。",
                },
                "prompt_description": {
                    "type": "string",
                    "description": "可选，对本次 generation prompt 的简短说明。",
                },
            },
            "required": ["prompt", "candidate_name"],
            "additionalProperties": False,
        }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        prompt = str(arguments["prompt"])
        candidate_name = str(arguments["candidate_name"])
        prompt_id = str(arguments.get("prompt_id") or "").strip()
        prompt_description = str(arguments.get("prompt_description") or "").strip()
        workspace_root = self.default_project_root / "lora_workspace"

        try:
            candidate_path = generate_cuda_candidate_from_prompt(
                prompt=prompt,
                candidate_name=candidate_name,
                llm_client=self.llm_client,
                workspace=str(workspace_root),
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(status="error", content=f"生成 CUDA 候选失败: {exc}")

        payload = {
            "candidate_path": candidate_path,
            "candidate_name": Path(candidate_path).stem,
            "prompt_id": prompt_id,
            "prompt_description": prompt_description,
            "workspace": str(workspace_root.resolve()),
        }
        return ToolResult(status="ok", content=json.dumps(payload, indent=2, ensure_ascii=False))


@dataclass
class ReviseCandidateTool:
    default_project_root: Path
    llm_client: "OpenAILLMClient"

    name: str = "revise_candidate"
    description: str = "根据编译错误、correctness 失败或性能瓶颈描述，让 LLM 修复并生成新的候选文件。"
    parameters_schema: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        self.parameters_schema = {
            "type": "object",
            "properties": {
                "candidate_path": {
                    "type": "string",
                    "description": "待修复候选文件路径。",
                },
                "error_or_bottleneck": {
                    "type": "string",
                    "description": "编译错误、correctness 失败原因或性能瓶颈说明。",
                },
                "max_retries": {
                    "type": "integer",
                    "description": "最多重试次数，默认 2。",
                },
            },
            "required": ["candidate_path", "error_or_bottleneck"],
            "additionalProperties": False,
        }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        candidate_path = self._resolve_within_project(str(arguments["candidate_path"]))
        if candidate_path is None or candidate_path.suffix.lower() != ".cu":
            return ToolResult(status="rejected", content="candidate_path 必须是项目目录内的 .cu 文件。")
        if not candidate_path.exists():
            return ToolResult(status="error", content=f"待修复候选不存在: {candidate_path}")

        max_retries = int(arguments.get("max_retries", 2))
        workspace_root = self.default_project_root / "lora_workspace"

        try:
            revised_candidate_path = revise_candidate(
                candidate_path=str(candidate_path),
                error_or_bottleneck=str(arguments["error_or_bottleneck"]),
                llm_client=self.llm_client,
                workspace=str(workspace_root),
                max_retries=max_retries,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(status="error", content=f"修复 CUDA 候选失败: {exc}")

        payload = {
            "source_candidate_path": str(candidate_path.resolve()),
            "revised_candidate_path": revised_candidate_path,
            "candidate_name": Path(revised_candidate_path).stem,
            "max_retries": max_retries,
        }
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
