from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from agent_framework.tools.tool_registry import ToolResult


ApprovalHandler = Callable[[str], bool]


def default_approval_handler(prompt: str) -> bool:
    answer = input(f"{prompt} [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


@dataclass
class BashRunnerTool:
    """Executes shell commands with timeout and safety interception."""

    approval_handler: ApprovalHandler = default_approval_handler
    default_timeout: int = 30

    name: str = "bash_runner"
    description: str = (
        "在本地 shell 中执行受限的诊断命令。"
        "严禁下载、安装、克隆或调用外部 benchmark。"
    )
    parameters_schema: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        self.parameters_schema = {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的 shell 命令。",
                },
                "cwd": {
                    "type": "string",
                    "description": "命令执行目录，可选。",
                },
                "timeout": {
                    "type": "integer",
                    "description": "超时时间，单位秒。默认 30。",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        }

    def run(self, arguments: dict) -> ToolResult:
        command = arguments["command"]
        cwd = arguments.get("cwd")
        timeout = int(arguments.get("timeout", self.default_timeout))

        blocked_reason = self._get_policy_block_reason(command)
        if blocked_reason:
            return ToolResult(
                status="rejected",
                content=f"命令被策略拒绝: {blocked_reason}\ncommand: {command}",
            )

        if self._is_high_risk_command(command):
            approved = self.approval_handler(
                f"检测到高风险 shell 命令，是否继续执行？\n{command}"
            )
            if not approved:
                return ToolResult(
                    status="rejected",
                    content=f"用户拒绝执行高风险命令: {command}",
                )

        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=True,
            )
        except subprocess.TimeoutExpired as exc:
            partial_stdout = exc.stdout or ""
            partial_stderr = exc.stderr or ""
            content = (
                f"命令执行超时({timeout}s)，请检查是否写了死循环或阻塞逻辑。\n"
                f"stdout:\n{partial_stdout}\n"
                f"stderr:\n{partial_stderr}"
            ).strip()
            return ToolResult(status="timeout", content=content)
        except subprocess.CalledProcessError as exc:
            content = (
                f"命令执行失败，退出码: {exc.returncode}\n"
                f"stdout:\n{exc.stdout or ''}\n"
                f"stderr:\n{exc.stderr or ''}"
            ).strip()
            return ToolResult(status="error", content=content)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                status="error",
                content=f"命令执行时发生系统异常: {exc}",
            )

        content = (
            f"命令执行成功，退出码: {completed.returncode}\n"
            f"stdout:\n{completed.stdout or ''}\n"
            f"stderr:\n{completed.stderr or ''}"
        ).strip()
        return ToolResult(status="ok", content=content)

    @staticmethod
    def _is_high_risk_command(command: str) -> bool:
        risky_tokens = [
            "rm -rf",
            "mkfs",
            "shutdown",
            "reboot",
            "poweroff",
            "dd ",
            "chmod -r",
            "chown -r",
            "git reset --hard",
            "curl ",
            "wget ",
            "sudo ",
            ">|",
            " > ",
        ]
        lowered = command.lower()
        return any(token in lowered for token in risky_tokens)

    @staticmethod
    def _get_policy_block_reason(command: str) -> str | None:
        lowered = command.lower()
        blocked_tokens = {
            "curl ": "禁止下载外部资源或第三方 benchmark",
            "wget ": "禁止下载外部资源或第三方 benchmark",
            "invoke-webrequest": "禁止通过 PowerShell 下载外部资源或 benchmark",
            "start-bitstransfer": "禁止通过 PowerShell 下载外部资源或 benchmark",
            "certutil ": "禁止通过 certutil 下载外部资源或 benchmark",
            "git clone": "禁止克隆外部仓库或 benchmark",
            "pip install": "禁止在 Agent 运行中联网安装第三方依赖或 benchmark",
            "python -m pip": "禁止在 Agent 运行中联网安装第三方依赖或 benchmark",
            "winget ": "禁止通过包管理器安装外部工具或 benchmark",
            "choco ": "禁止通过包管理器安装外部工具或 benchmark",
        }
        for token, reason in blocked_tokens.items():
            if token in lowered:
                return reason
        return None


@dataclass
class ReadFileTool:
    """Reads a local text file."""

    name: str = "read_file"
    description: str = "读取本地文本文件内容。"
    parameters_schema: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        self.parameters_schema = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "要读取的文件路径。",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        }

    def run(self, arguments: dict) -> ToolResult:
        path = Path(arguments["path"]).expanduser()
        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ToolResult(status="error", content=f"文件不存在: {path}")
        except UnicodeDecodeError:
            return ToolResult(status="error", content=f"文件不是 UTF-8 文本或包含不可解码内容: {path}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(status="error", content=f"读取文件失败: {exc}")

        return ToolResult(status="ok", content=content)


@dataclass
class WriteFileTool:
    """Writes content to a local file, optionally requiring human confirmation."""

    approval_handler: ApprovalHandler = default_approval_handler
    require_approval: bool = True

    name: str = "write_file"
    description: str = "写入本地文件。可根据运行模式配置是否需要人工确认。"
    parameters_schema: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        self.parameters_schema = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "要写入的文件路径。",
                },
                "content": {
                    "type": "string",
                    "description": "要写入的文本内容。",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        }

    def run(self, arguments: dict) -> ToolResult:
        path = Path(arguments["path"]).expanduser()
        content = arguments["content"]

        if self.require_approval:
            approved = self.approval_handler(f"即将写入文件 {path}，是否继续？")
            if not approved:
                return ToolResult(
                    status="rejected",
                    content=f"用户拒绝写入文件: {path}",
                )

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(status="error", content=f"写入文件失败: {exc}")

        return ToolResult(status="ok", content=f"文件写入成功: {path}")
