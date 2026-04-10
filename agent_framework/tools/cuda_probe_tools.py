from __future__ import annotations

import csv
import io
import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from agent_framework.tools.tool_registry import ToolResult


@dataclass
class ParsedFrequencyResult:
    metric: str
    value_mhz: float
    raw_csv: str

    def to_json(
        self,
        compile_stderr: str = "",
        run_stderr: str = "",
    ) -> str:
        return json.dumps(
            {
                "metric": self.metric,
                "value_mhz": self.value_mhz,
                "raw_csv": self.raw_csv,
                "compile_stderr": compile_stderr,
                "run_stderr": run_stderr,
            },
            ensure_ascii=False,
            indent=2,
        )


@dataclass
class ParsedShmemResult:
    no_conflict_cycles: float
    conflict_cycles: float
    conflict_penalty_cycles: float
    raw_csv: str

    def to_json(
        self,
        compile_stderr: str = "",
        run_stderr: str = "",
    ) -> str:
        return json.dumps(
            {
                "metric": "shared_memory_bank_conflict_penalty",
                "no_conflict_stride_1_cycles": self.no_conflict_cycles,
                "conflict_stride_32_cycles": self.conflict_cycles,
                "conflict_penalty_cycles": self.conflict_penalty_cycles,
                "raw_csv": self.raw_csv,
                "compile_stderr": compile_stderr,
                "run_stderr": run_stderr,
            },
            ensure_ascii=False,
            indent=2,
        )


@dataclass
class ParsedLatencySweepResult:
    samples: list[dict[str, float]]
    min_latency_cycles: float
    max_latency_cycles: float
    raw_csv: str

    def to_json(
        self,
        compile_stderr: str = "",
        run_stderr: str = "",
    ) -> str:
        return json.dumps(
            {
                "metric": "memory_latency_sweep",
                "samples": self.samples,
                "sample_count": len(self.samples),
                "min_latency_cycles": self.min_latency_cycles,
                "max_latency_cycles": self.max_latency_cycles,
                "raw_csv": self.raw_csv,
                "compile_stderr": compile_stderr,
                "run_stderr": run_stderr,
            },
            ensure_ascii=False,
            indent=2,
        )


def parse_probe_frequency_csv(stdout: str) -> ParsedFrequencyResult:
    text = stdout.strip()
    if not text:
        raise ValueError("频率探针没有输出任何 stdout。")

    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if reader.fieldnames != ["Metric", "Value"]:
        raise ValueError(f"频率探针 CSV 表头不符合预期: {reader.fieldnames}")
    if len(rows) != 1:
        raise ValueError(f"频率探针 CSV 数据行数量异常: {len(rows)}")

    row = rows[0]
    metric = (row.get("Metric") or "").strip()
    raw_value = (row.get("Value") or "").strip()
    if metric != "Actual_Boost_Frequency_MHz":
        raise ValueError(f"频率探针 Metric 字段异常: {metric}")

    try:
        value_mhz = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"频率探针 Value 不是合法数字: {raw_value}") from exc

    if not math.isfinite(value_mhz) or value_mhz <= 0.0:
        raise ValueError(f"频率探针结果非法，必须为正实数: {value_mhz}")

    # Reject absurd outputs so the agent does not anchor on nonsense.
    if value_mhz < 100.0 or value_mhz > 10000.0:
        raise ValueError(f"频率探针结果超出合理范围: {value_mhz} MHz")

    return ParsedFrequencyResult(
        metric=metric,
        value_mhz=value_mhz,
        raw_csv=text,
    )


def parse_probe_shmem_csv(stdout: str) -> ParsedShmemResult:
    text = stdout.strip()
    if not text:
        raise ValueError("共享内存探针没有输出任何 stdout。")

    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if reader.fieldnames != ["Test_Type", "Latency_Cycles"]:
        raise ValueError(f"共享内存探针 CSV 表头不符合预期: {reader.fieldnames}")
    if len(rows) != 2:
        raise ValueError(f"共享内存探针 CSV 数据行数量异常: {len(rows)}")

    parsed_rows: dict[str, float] = {}
    for row in rows:
        test_type = (row.get("Test_Type") or "").strip()
        raw_value = (row.get("Latency_Cycles") or "").strip()
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"共享内存探针 Latency_Cycles 不是合法数字: {raw_value}") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"共享内存探针结果非法，必须为正实数: {value}")
        parsed_rows[test_type] = value

    expected_keys = {"No_Conflict_Stride_1", "32_Way_Conflict_Stride_32"}
    if set(parsed_rows) != expected_keys:
        raise ValueError(f"共享内存探针 Test_Type 字段异常: {sorted(parsed_rows)}")

    no_conflict_cycles = parsed_rows["No_Conflict_Stride_1"]
    conflict_cycles = parsed_rows["32_Way_Conflict_Stride_32"]
    if conflict_cycles < no_conflict_cycles:
        raise ValueError(
            "共享内存探针结果不符合预期：严重冲突延迟小于无冲突延迟。"
        )

    return ParsedShmemResult(
        no_conflict_cycles=no_conflict_cycles,
        conflict_cycles=conflict_cycles,
        conflict_penalty_cycles=conflict_cycles - no_conflict_cycles,
        raw_csv=text,
    )


def parse_probe_latency_csv(stdout: str) -> ParsedLatencySweepResult:
    text = stdout.strip()
    if not text:
        raise ValueError("缓存延迟探针没有输出任何 stdout。")

    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if reader.fieldnames != ["Size_KB", "Latency_Cycles"]:
        raise ValueError(f"缓存延迟探针 CSV 表头不符合预期: {reader.fieldnames}")
    if not rows:
        raise ValueError("缓存延迟探针没有输出任何数据行。")

    samples: list[dict[str, float]] = []
    previous_size_kb = -1.0
    min_latency_cycles = math.inf
    max_latency_cycles = -math.inf

    for row in rows:
        raw_size_kb = (row.get("Size_KB") or "").strip()
        raw_latency = (row.get("Latency_Cycles") or "").strip()
        try:
            size_kb = float(raw_size_kb)
        except ValueError as exc:
            raise ValueError(f"缓存延迟探针 Size_KB 不是合法数字: {raw_size_kb}") from exc
        try:
            latency_cycles = float(raw_latency)
        except ValueError as exc:
            raise ValueError(
                f"缓存延迟探针 Latency_Cycles 不是合法数字: {raw_latency}"
            ) from exc

        if not math.isfinite(size_kb) or size_kb <= 0.0:
            raise ValueError(f"缓存延迟探针 Size_KB 非法: {size_kb}")
        if not math.isfinite(latency_cycles) or latency_cycles <= 0.0:
            raise ValueError(f"缓存延迟探针 Latency_Cycles 非法: {latency_cycles}")
        if size_kb <= previous_size_kb:
            raise ValueError("缓存延迟探针 Size_KB 必须严格递增。")

        samples.append(
            {
                "size_kb": size_kb,
                "latency_cycles": latency_cycles,
            }
        )
        previous_size_kb = size_kb
        min_latency_cycles = min(min_latency_cycles, latency_cycles)
        max_latency_cycles = max(max_latency_cycles, latency_cycles)

    return ParsedLatencySweepResult(
        samples=samples,
        min_latency_cycles=min_latency_cycles,
        max_latency_cycles=max_latency_cycles,
        raw_csv=text,
    )


def _compile_cuda_probe(
    *,
    nvcc_path: str,
    source_path: Path,
    binary_path: Path,
    project_root: Path,
    timeout: int,
    probe_name: str,
) -> ToolResult | str:
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        nvcc_path,
        "-O3",
        "-std=c++17",
        str(source_path),
        "-o",
        str(binary_path),
    ]

    try:
        completed = subprocess.run(
            command,
            cwd=str(project_root),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=True,
        )
    except subprocess.TimeoutExpired as exc:
        return ToolResult(
            status="timeout",
            content=(
                f"{probe_name} 编译超时({timeout}s)。\n"
                f"stdout:\n{exc.stdout or ''}\n"
                f"stderr:\n{exc.stderr or ''}"
            ).strip(),
        )
    except subprocess.CalledProcessError as exc:
        return ToolResult(
            status="error",
            content=(
                f"{probe_name} 编译失败，退出码: {exc.returncode}\n"
                f"stdout:\n{exc.stdout or ''}\n"
                f"stderr:\n{exc.stderr or ''}"
            ).strip(),
        )
    except FileNotFoundError:
        return ToolResult(
            status="error",
            content=f"找不到 nvcc，可通过 nvcc_path 指定编译器路径: {nvcc_path}",
        )
    except Exception as exc:  # noqa: BLE001
        return ToolResult(
            status="error",
            content=f"调用 nvcc 编译 {probe_name} 时发生系统异常: {exc}",
        )

    return completed.stderr.strip()


def _run_cuda_probe(
    *,
    binary_path: Path,
    project_root: Path,
    timeout: int,
    probe_name: str,
) -> ToolResult | tuple[str, str]:
    try:
        completed = subprocess.run(
            [str(binary_path)],
            cwd=str(project_root),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=True,
        )
    except subprocess.TimeoutExpired as exc:
        return ToolResult(
            status="timeout",
            content=(
                f"{probe_name} 运行超时({timeout}s)。\n"
                f"stdout:\n{exc.stdout or ''}\n"
                f"stderr:\n{exc.stderr or ''}"
            ).strip(),
        )
    except subprocess.CalledProcessError as exc:
        return ToolResult(
            status="error",
            content=(
                f"{probe_name} 运行失败，退出码: {exc.returncode}\n"
                f"stdout:\n{exc.stdout or ''}\n"
                f"stderr:\n{exc.stderr or ''}"
            ).strip(),
        )
    except Exception as exc:  # noqa: BLE001
        return ToolResult(
            status="error",
            content=f"运行 {probe_name} 时发生系统异常: {exc}",
        )

    return completed.stdout, completed.stderr


@dataclass
class RunProbeFrequencyTool:
    """Compile, run, and parse the frequency probe."""

    default_project_root: Path
    default_compile_timeout: int = 120
    default_run_timeout: int = 60

    name: str = "run_probe_frequency"
    description: str = (
        "编译并运行 CUDA 频率探针 probe_frequency.cu，"
        "解析 CSV 输出，返回结构化的真实 Boost 频率结果。"
    )
    parameters_schema: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        self.parameters_schema = {
            "type": "object",
            "properties": {
                "source_path": {
                    "type": "string",
                    "description": "频率探针源文件路径，默认使用项目根目录下的 probe_frequency.cu。",
                },
                "binary_path": {
                    "type": "string",
                    "description": "输出二进制路径，默认与源文件同名去掉 .cu 后缀。",
                },
                "compile_timeout": {
                    "type": "integer",
                    "description": "编译超时时间，单位秒，默认 120。",
                },
                "run_timeout": {
                    "type": "integer",
                    "description": "运行超时时间，单位秒，默认 60。",
                },
                "nvcc_path": {
                    "type": "string",
                    "description": "nvcc 可执行文件路径，默认直接使用 nvcc。",
                },
            },
            "required": [],
            "additionalProperties": False,
        }

    def run(self, arguments: dict) -> ToolResult:
        project_root = self.default_project_root
        source_path = Path(arguments.get("source_path", project_root / "probe_frequency.cu")).expanduser()
        binary_path = Path(arguments.get("binary_path", source_path.with_suffix(""))).expanduser()
        compile_timeout = int(arguments.get("compile_timeout", self.default_compile_timeout))
        run_timeout = int(arguments.get("run_timeout", self.default_run_timeout))
        nvcc_path = arguments.get("nvcc_path", "nvcc")

        if not source_path.exists():
            return ToolResult(
                status="error",
                content=f"频率探针源文件不存在: {source_path}",
            )

        compile_result = self._compile_probe(
            nvcc_path=nvcc_path,
            source_path=source_path,
            binary_path=binary_path,
            timeout=compile_timeout,
        )
        if isinstance(compile_result, ToolResult):
            return compile_result
        compile_stderr = compile_result

        run_result = self._run_probe(binary_path=binary_path, timeout=run_timeout)
        if isinstance(run_result, ToolResult):
            return run_result

        stdout, stderr = run_result
        try:
            parsed = parse_probe_frequency_csv(stdout)
        except ValueError as exc:
            return ToolResult(
                status="error",
                content=(
                    f"频率探针输出解析失败: {exc}\n"
                    f"stdout:\n{stdout}\n"
                    f"stderr:\n{stderr}"
                ).strip(),
            )

        return ToolResult(
            status="ok",
            content=parsed.to_json(
                compile_stderr=compile_stderr,
                run_stderr=stderr,
            ),
        )

    def _compile_probe(
        self,
        nvcc_path: str,
        source_path: Path,
        binary_path: Path,
        timeout: int,
    ) -> ToolResult | str:
        return _compile_cuda_probe(
            nvcc_path=nvcc_path,
            source_path=source_path,
            binary_path=binary_path,
            project_root=self.default_project_root,
            timeout=timeout,
            probe_name="频率探针",
        )

    def _run_probe(
        self,
        binary_path: Path,
        timeout: int,
    ) -> ToolResult | tuple[str, str]:
        return _run_cuda_probe(
            binary_path=binary_path,
            project_root=self.default_project_root,
            timeout=timeout,
            probe_name="频率探针",
        )


@dataclass
class RunProbeShmemTool:
    """Compile, run, and parse the shared-memory bank-conflict probe."""

    default_project_root: Path
    default_compile_timeout: int = 120
    default_run_timeout: int = 60

    name: str = "run_probe_shmem"
    description: str = (
        "编译并运行 CUDA 共享内存 bank conflict 探针 probe_shmem.cu，"
        "解析 CSV 输出，返回结构化的无冲突、冲突和冲突惩罚结果。"
    )
    parameters_schema: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        self.parameters_schema = {
            "type": "object",
            "properties": {
                "source_path": {
                    "type": "string",
                    "description": "共享内存探针源文件路径，默认使用项目根目录下的 probe_shmem.cu。",
                },
                "binary_path": {
                    "type": "string",
                    "description": "输出二进制路径，默认与源文件同名去掉 .cu 后缀。",
                },
                "compile_timeout": {
                    "type": "integer",
                    "description": "编译超时时间，单位秒，默认 120。",
                },
                "run_timeout": {
                    "type": "integer",
                    "description": "运行超时时间，单位秒，默认 60。",
                },
                "nvcc_path": {
                    "type": "string",
                    "description": "nvcc 可执行文件路径，默认直接使用 nvcc。",
                },
            },
            "required": [],
            "additionalProperties": False,
        }

    def run(self, arguments: dict) -> ToolResult:
        project_root = self.default_project_root
        source_path = Path(arguments.get("source_path", project_root / "probe_shmem.cu")).expanduser()
        binary_path = Path(arguments.get("binary_path", source_path.with_suffix(""))).expanduser()
        compile_timeout = int(arguments.get("compile_timeout", self.default_compile_timeout))
        run_timeout = int(arguments.get("run_timeout", self.default_run_timeout))
        nvcc_path = arguments.get("nvcc_path", "nvcc")

        if not source_path.exists():
            return ToolResult(
                status="error",
                content=f"共享内存探针源文件不存在: {source_path}",
            )

        compile_result = self._compile_probe(
            nvcc_path=nvcc_path,
            source_path=source_path,
            binary_path=binary_path,
            timeout=compile_timeout,
        )
        if isinstance(compile_result, ToolResult):
            return compile_result
        compile_stderr = compile_result

        run_result = self._run_probe(binary_path=binary_path, timeout=run_timeout)
        if isinstance(run_result, ToolResult):
            return run_result

        stdout, stderr = run_result
        try:
            parsed = parse_probe_shmem_csv(stdout)
        except ValueError as exc:
            return ToolResult(
                status="error",
                content=(
                    f"共享内存探针输出解析失败: {exc}\n"
                    f"stdout:\n{stdout}\n"
                    f"stderr:\n{stderr}"
                ).strip(),
            )

        return ToolResult(
            status="ok",
            content=parsed.to_json(
                compile_stderr=compile_stderr,
                run_stderr=stderr,
            ),
        )

    def _compile_probe(
        self,
        nvcc_path: str,
        source_path: Path,
        binary_path: Path,
        timeout: int,
    ) -> ToolResult | str:
        return _compile_cuda_probe(
            nvcc_path=nvcc_path,
            source_path=source_path,
            binary_path=binary_path,
            project_root=self.default_project_root,
            timeout=timeout,
            probe_name="共享内存探针",
        )

    def _run_probe(
        self,
        binary_path: Path,
        timeout: int,
    ) -> ToolResult | tuple[str, str]:
        return _run_cuda_probe(
            binary_path=binary_path,
            project_root=self.default_project_root,
            timeout=timeout,
            probe_name="共享内存探针",
        )


@dataclass
class RunProbeLatencyTool:
    """Compile, run, and parse the memory-latency sweep probe."""

    default_project_root: Path
    default_compile_timeout: int = 120
    default_run_timeout: int = 120

    name: str = "run_probe_latency"
    description: str = (
        "编译并运行 CUDA 缓存延迟探针 probe_latency.cu，"
        "解析 CSV 输出，返回结构化的 working set 与延迟扫频结果。"
    )
    parameters_schema: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        self.parameters_schema = {
            "type": "object",
            "properties": {
                "source_path": {
                    "type": "string",
                    "description": "缓存延迟探针源文件路径，默认使用项目根目录下的 probe_latency.cu。",
                },
                "binary_path": {
                    "type": "string",
                    "description": "输出二进制路径，默认与源文件同名去掉 .cu 后缀。",
                },
                "compile_timeout": {
                    "type": "integer",
                    "description": "编译超时时间，单位秒，默认 120。",
                },
                "run_timeout": {
                    "type": "integer",
                    "description": "运行超时时间，单位秒，默认 120。",
                },
                "nvcc_path": {
                    "type": "string",
                    "description": "nvcc 可执行文件路径，默认直接使用 nvcc。",
                },
            },
            "required": [],
            "additionalProperties": False,
        }

    def run(self, arguments: dict) -> ToolResult:
        project_root = self.default_project_root
        source_path = Path(arguments.get("source_path", project_root / "probe_latency.cu")).expanduser()
        binary_path = Path(arguments.get("binary_path", source_path.with_suffix(""))).expanduser()
        compile_timeout = int(arguments.get("compile_timeout", self.default_compile_timeout))
        run_timeout = int(arguments.get("run_timeout", self.default_run_timeout))
        nvcc_path = arguments.get("nvcc_path", "nvcc")

        if not source_path.exists():
            return ToolResult(
                status="error",
                content=f"缓存延迟探针源文件不存在: {source_path}",
            )

        compile_result = self._compile_probe(
            nvcc_path=nvcc_path,
            source_path=source_path,
            binary_path=binary_path,
            timeout=compile_timeout,
        )
        if isinstance(compile_result, ToolResult):
            return compile_result
        compile_stderr = compile_result

        run_result = self._run_probe(binary_path=binary_path, timeout=run_timeout)
        if isinstance(run_result, ToolResult):
            return run_result

        stdout, stderr = run_result
        try:
            parsed = parse_probe_latency_csv(stdout)
        except ValueError as exc:
            return ToolResult(
                status="error",
                content=(
                    f"缓存延迟探针输出解析失败: {exc}\n"
                    f"stdout:\n{stdout}\n"
                    f"stderr:\n{stderr}"
                ).strip(),
            )

        return ToolResult(
            status="ok",
            content=parsed.to_json(
                compile_stderr=compile_stderr,
                run_stderr=stderr,
            ),
        )

    def _compile_probe(
        self,
        nvcc_path: str,
        source_path: Path,
        binary_path: Path,
        timeout: int,
    ) -> ToolResult | str:
        return _compile_cuda_probe(
            nvcc_path=nvcc_path,
            source_path=source_path,
            binary_path=binary_path,
            project_root=self.default_project_root,
            timeout=timeout,
            probe_name="缓存延迟探针",
        )

    def _run_probe(
        self,
        binary_path: Path,
        timeout: int,
    ) -> ToolResult | tuple[str, str]:
        return _run_cuda_probe(
            binary_path=binary_path,
            project_root=self.default_project_root,
            timeout=timeout,
            probe_name="缓存延迟探针",
        )
