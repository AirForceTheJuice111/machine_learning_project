from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from agent_framework.tools.tool_registry import ToolResult


@dataclass
class CompileAndRunCudaSourceTool:
    """Compile and optionally profile arbitrary agent-generated CUDA source."""

    default_project_root: Path
    default_compile_timeout: int = 120
    default_run_timeout: int = 120
    default_profile_timeout: int = 180

    name: str = "compile_and_run_cuda_source"
    description: str = (
        "编译 Agent 动态生成的 CUDA C++ 源文件，并可选择直接运行或通过 ncu 采集指标。"
    )
    parameters_schema: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        self.parameters_schema = {
            "type": "object",
            "properties": {
                "source_path": {
                    "type": "string",
                    "description": "要编译的 CUDA 源文件路径。通常由 Agent 先通过 write_file 生成。",
                },
                "binary_path": {
                    "type": "string",
                    "description": "输出二进制路径，默认与源文件同名去掉 .cu 后缀。",
                },
                "nvcc_path": {
                    "type": "string",
                    "description": "nvcc 可执行文件路径，默认直接使用 nvcc。",
                },
                "extra_nvcc_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "附加 nvcc 编译参数列表，例如 ['-lineinfo']。",
                },
                "program_args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "运行被编译程序时传递的参数列表。",
                },
                "compile_timeout": {
                    "type": "integer",
                    "description": "编译超时时间，单位秒，默认 120。",
                },
                "run_timeout": {
                    "type": "integer",
                    "description": "程序运行超时时间，单位秒，默认 120。",
                },
                "profile_with_ncu": {
                    "type": "boolean",
                    "description": "是否在编译后使用 ncu 对该二进制进行 profiling。",
                },
                "ncu_path": {
                    "type": "string",
                    "description": "ncu 可执行文件路径，默认直接使用 ncu。",
                },
                "ncu_metrics": {
                    "type": "string",
                    "description": "可选，逗号分隔的 ncu metrics 列表。",
                },
                "ncu_set": {
                    "type": "string",
                    "description": "可选，使用 ncu 预设集合，例如 full。若给出 metrics，则优先使用 metrics。",
                },
                "profile_timeout": {
                    "type": "integer",
                    "description": "ncu profiling 超时时间，单位秒，默认 180。",
                },
            },
            "required": ["source_path"],
            "additionalProperties": False,
        }

    def run(self, arguments: dict) -> ToolResult:
        project_root = self.default_project_root
        source_path = Path(arguments["source_path"]).expanduser()
        binary_path = Path(arguments.get("binary_path", source_path.with_suffix(""))).expanduser()
        nvcc_path = arguments.get("nvcc_path", "nvcc")
        compile_timeout = int(arguments.get("compile_timeout", self.default_compile_timeout))
        run_timeout = int(arguments.get("run_timeout", self.default_run_timeout))
        profile_timeout = int(arguments.get("profile_timeout", self.default_profile_timeout))
        extra_nvcc_flags = list(arguments.get("extra_nvcc_flags", []))
        program_args = list(arguments.get("program_args", []))
        profile_with_ncu = bool(arguments.get("profile_with_ncu", False))
        ncu_path = arguments.get("ncu_path", "ncu")
        ncu_metrics = arguments.get("ncu_metrics", "")
        ncu_set = arguments.get("ncu_set", "")

        if not source_path.exists():
            return ToolResult(status="error", content=f"CUDA 源文件不存在: {source_path}")

        compile_result = self._compile_source(
            nvcc_path=nvcc_path,
            source_path=source_path,
            binary_path=binary_path,
            extra_nvcc_flags=extra_nvcc_flags,
            timeout=compile_timeout,
        )
        if isinstance(compile_result, ToolResult):
            return compile_result

        run_result = self._run_binary(
            binary_path=binary_path,
            program_args=program_args,
            timeout=run_timeout,
        )
        if isinstance(run_result, ToolResult):
            return run_result
        run_stdout, run_stderr = run_result

        profile_stdout = ""
        profile_stderr = ""
        if profile_with_ncu:
            profile_result = self._profile_with_ncu(
                ncu_path=ncu_path,
                binary_path=binary_path,
                program_args=program_args,
                ncu_metrics=ncu_metrics,
                ncu_set=ncu_set,
                timeout=profile_timeout,
            )
            if isinstance(profile_result, ToolResult):
                return profile_result
            profile_stdout, profile_stderr = profile_result

        return ToolResult(
            status="ok",
            content=json.dumps(
                {
                    "source_path": str(source_path),
                    "binary_path": str(binary_path),
                    "compile_stderr": compile_result,
                    "run_stdout": run_stdout,
                    "run_stderr": run_stderr,
                    "profile_stdout": profile_stdout,
                    "profile_stderr": profile_stderr,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

    def _compile_source(
        self,
        *,
        nvcc_path: str,
        source_path: Path,
        binary_path: Path,
        extra_nvcc_flags: list[str],
        timeout: int,
    ) -> ToolResult | str:
        binary_path.parent.mkdir(parents=True, exist_ok=True)
        command = [nvcc_path, "-O3", "-std=c++17", *extra_nvcc_flags, str(source_path), "-o", str(binary_path)]
        try:
            completed = subprocess.run(
                command,
                cwd=str(self.default_project_root),
                text=True,
                capture_output=True,
                timeout=timeout,
                check=True,
            )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                status="timeout",
                content=(
                    f"动态 CUDA 源文件编译超时({timeout}s)。\n"
                    f"stdout:\n{exc.stdout or ''}\n"
                    f"stderr:\n{exc.stderr or ''}"
                ).strip(),
            )
        except subprocess.CalledProcessError as exc:
            return ToolResult(
                status="error",
                content=(
                    f"动态 CUDA 源文件编译失败，退出码: {exc.returncode}\n"
                    f"stdout:\n{exc.stdout or ''}\n"
                    f"stderr:\n{exc.stderr or ''}"
                ).strip(),
            )
        except FileNotFoundError:
            return ToolResult(status="error", content=f"找不到 nvcc: {nvcc_path}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(status="error", content=f"编译动态 CUDA 源文件时发生系统异常: {exc}")
        return completed.stderr.strip()

    def _run_binary(
        self,
        *,
        binary_path: Path,
        program_args: list[str],
        timeout: int,
    ) -> ToolResult | tuple[str, str]:
        try:
            completed = subprocess.run(
                [str(binary_path), *program_args],
                cwd=str(self.default_project_root),
                text=True,
                capture_output=True,
                timeout=timeout,
                check=True,
            )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                status="timeout",
                content=(
                    f"动态 CUDA 程序运行超时({timeout}s)。\n"
                    f"stdout:\n{exc.stdout or ''}\n"
                    f"stderr:\n{exc.stderr or ''}"
                ).strip(),
            )
        except subprocess.CalledProcessError as exc:
            return ToolResult(
                status="error",
                content=(
                    f"动态 CUDA 程序运行失败，退出码: {exc.returncode}\n"
                    f"stdout:\n{exc.stdout or ''}\n"
                    f"stderr:\n{exc.stderr or ''}"
                ).strip(),
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(status="error", content=f"运行动态 CUDA 程序时发生系统异常: {exc}")
        return completed.stdout, completed.stderr

    def _profile_with_ncu(
        self,
        *,
        ncu_path: str,
        binary_path: Path,
        program_args: list[str],
        ncu_metrics: str,
        ncu_set: str,
        timeout: int,
    ) -> ToolResult | tuple[str, str]:
        command = [ncu_path]
        if ncu_metrics:
            command.extend(["--metrics", ncu_metrics])
        elif ncu_set:
            command.extend(["--set", ncu_set])
        else:
            command.extend(["--set", "full"])
        command.append(str(binary_path))
        command.extend(program_args)
        try:
            completed = subprocess.run(
                command,
                cwd=str(self.default_project_root),
                text=True,
                capture_output=True,
                timeout=timeout,
                check=True,
            )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                status="timeout",
                content=(
                    f"ncu profiling 超时({timeout}s)。\n"
                    f"stdout:\n{exc.stdout or ''}\n"
                    f"stderr:\n{exc.stderr or ''}"
                ).strip(),
            )
        except subprocess.CalledProcessError as exc:
            return ToolResult(
                status="error",
                content=(
                    f"ncu profiling 失败，退出码: {exc.returncode}\n"
                    f"stdout:\n{exc.stdout or ''}\n"
                    f"stderr:\n{exc.stderr or ''}"
                ).strip(),
            )
        except FileNotFoundError:
            return ToolResult(status="error", content=f"找不到 ncu: {ncu_path}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(status="error", content=f"执行 ncu profiling 时发生系统异常: {exc}")
        return completed.stdout, completed.stderr
