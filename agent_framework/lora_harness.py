from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_RANK = 16
DEFAULT_RTOL = 1e-4
DEFAULT_ATOL = 1e-4
DEFAULT_QUICK_SHAPES = [3584, 4096]
DEFAULT_FULL_SHAPES = [3584, 4096, 4608]
DEFAULT_SMOKE_SHAPES = [3584]


def starter_optimized_lora_source() -> str:
    return r"""#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")

namespace {

void validate_inputs(const torch::Tensor& W,
                     const torch::Tensor& X,
                     const torch::Tensor& A,
                     const torch::Tensor& B) {
    CHECK_CUDA(W);
    CHECK_CUDA(X);
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    CHECK_CONTIGUOUS(W);
    CHECK_CONTIGUOUS(X);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(B);
    CHECK_FLOAT32(W);
    CHECK_FLOAT32(X);
    CHECK_FLOAT32(A);
    CHECK_FLOAT32(B);
    TORCH_CHECK(W.dim() == 2, "W must be 2D");
    TORCH_CHECK(X.dim() == 2, "X must be 2D");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(W.size(0) == W.size(1), "W must be square");
    TORCH_CHECK(X.size(0) == X.size(1), "X must be square");
    TORCH_CHECK(A.size(0) == W.size(0), "A rows must match d");
    TORCH_CHECK(B.size(0) == W.size(0), "B rows must match d");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must share rank r");
    TORCH_CHECK(W.size(1) == X.size(0), "W and X inner dims must match");
    TORCH_CHECK(A.size(1) > 0, "rank r must be positive");
    TORCH_CHECK(W.device() == X.device(), "W and X must be on the same CUDA device");
    TORCH_CHECK(W.device() == A.device(), "W and A must be on the same CUDA device");
    TORCH_CHECK(W.device() == B.device(), "W and B must be on the same CUDA device");
}

}  // namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    validate_inputs(W, X, A, B);

    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    // Safe baseline: rely on ATen GEMM path first, then let the Agent
    // iteratively replace pieces with custom CUDA kernels.
    auto BT = Bc.transpose(0, 1).contiguous();
    auto BTX = at::mm(BT, Xc);
    auto WX = at::mm(Wc, Xc);
    auto ABTX = at::mm(Ac, BTX);
    return (WX + ABTX).contiguous();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward (baseline)");
}
"""


def resolve_shape_values(shape_preset: str, shape_values: list[int] | None) -> list[int]:
    if shape_values:
        return sorted({int(value) for value in shape_values if int(value) > 0})
    preset = shape_preset.lower().strip()
    if preset == "smoke":
        return list(DEFAULT_SMOKE_SHAPES)
    if preset == "full":
        return list(DEFAULT_FULL_SHAPES)
    return list(DEFAULT_QUICK_SHAPES)


def default_warmup_iters(shape_preset: str) -> tuple[int, int]:
    preset = shape_preset.lower().strip()
    if preset == "smoke":
        return 1, 3
    if preset == "full":
        return 5, 15
    return 3, 8


def _ensure_torch_available():
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime env dependent
        raise RuntimeError(
            "缺少 torch，无法评测 LoRA 候选实现。请在具备 PyTorch + CUDA 的环境中运行 phase2。"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境未检测到可用 CUDA，无法评测 optimized_lora.cu。")
    return torch


def generate_synthetic_inputs(d: int, *, rank: int = DEFAULT_RANK, seed: int = 0):
    torch = _ensure_torch_available()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + d)
    kwargs = {
        "device": "cuda",
        "dtype": torch.float32,
        "generator": generator,
    }
    W = torch.randn((d, d), **kwargs).contiguous()
    X = torch.randn((d, d), **kwargs).contiguous()
    A = torch.randn((d, rank), **kwargs).contiguous()
    B = torch.randn((d, rank), **kwargs).contiguous()
    return W, X, A, B


def reference_impl(W, X, A, B):
    torch = _ensure_torch_available()
    with torch.no_grad():
        return W @ X + A @ (B.transpose(0, 1).contiguous() @ X)


def build_module(cu_path: str, *, build_root: Path):
    torch = _ensure_torch_available()
    from torch.utils.cpp_extension import load  # type: ignore

    source_path = Path(cu_path).resolve()
    source_text = source_path.read_text(encoding="utf-8")
    digest = hashlib.sha1(source_text.encode("utf-8")).hexdigest()[:12]
    module_name = f"optimized_lora_ext_{digest}"
    build_dir = build_root / digest
    build_dir.mkdir(parents=True, exist_ok=True)
    extra_cuda_cflags = ["-O3"]
    use_fast_math = os.getenv("LORA_HARNESS_USE_FAST_MATH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if use_fast_math:
        extra_cuda_cflags.append("--use_fast_math")

    module = load(
        name=module_name,
        sources=[str(source_path)],
        verbose=False,
        with_cuda=True,
        extra_cflags=["-O3"],
        extra_cuda_cflags=extra_cuda_cflags,
        build_directory=str(build_dir),
    )
    return module, module_name


def check_correctness(y, y_ref):
    torch = _ensure_torch_available()
    diff = (y - y_ref).float()
    max_abs_err = diff.abs().max().item()
    rel_l2_err = (diff.norm() / (y_ref.float().norm() + 1e-12)).item()
    passed = torch.allclose(y, y_ref, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
    return passed, max_abs_err, rel_l2_err


def benchmark(fn, W, X, A, B, *, warmup: int, iters: int) -> float:
    torch = _ensure_torch_available()
    for _ in range(warmup):
        _ = fn(W, X, A, B)
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = fn(W, X, A, B)
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))

    times.sort()
    return times[len(times) // 2]


def evaluate_candidate(
    *,
    source_path: Path,
    build_root: Path,
    shape_preset: str,
    shape_values: list[int] | None = None,
    warmup: int | None = None,
    iters: int | None = None,
    rank: int = DEFAULT_RANK,
    report_path: Path | None = None,
) -> dict[str, Any]:
    if not source_path.exists():
        raise FileNotFoundError(f"候选源码不存在: {source_path}")
    resolved_shapes = resolve_shape_values(shape_preset, shape_values)
    default_warmup, default_iters = default_warmup_iters(shape_preset)
    warmup = default_warmup if warmup is None else int(warmup)
    iters = default_iters if iters is None else int(iters)

    result: dict[str, Any] = {
        "source_path": str(source_path.resolve()),
        "shape_preset": shape_preset,
        "shape_values": resolved_shapes,
        "warmup": warmup,
        "iters": iters,
        "rank": rank,
        "compile_ok": False,
        "correctness_passed": False,
        "max_abs_err": None,
        "rel_l2_err": None,
        "mean_student_ms": None,
        "mean_torch_ms": None,
        "mean_speedup": 0.0,
        "min_speedup": 0.0,
        "shape_results": [],
        "module_name": None,
        "score": None,
        "score_breakdown": None,
        "best_eligible": False,
    }

    try:
        module, module_name = build_module(str(source_path), build_root=build_root)
        result["compile_ok"] = True
        result["module_name"] = module_name
    except Exception as exc:  # noqa: BLE001
        result["compile_error"] = str(exc)
        return _finalize_report(result, report_path)

    max_abs_err = 0.0
    max_rel_l2_err = 0.0
    student_times: list[float] = []
    torch_times: list[float] = []
    speedups: list[float] = []
    all_shapes_passed = True

    for d in resolved_shapes:
        try:
            W, X, A, B = generate_synthetic_inputs(d, rank=rank, seed=2026)
            with _ensure_torch_available().no_grad():
                y_student = module.forward(W, X, A, B)
                y_ref = reference_impl(W, X, A, B)
            passed, shape_max_abs_err, shape_rel_l2_err = check_correctness(y_student, y_ref)
            shape_entry: dict[str, Any] = {
                "d": d,
                "correct": passed,
                "max_abs_err": shape_max_abs_err,
                "rel_l2_err": shape_rel_l2_err,
            }
            max_abs_err = max(max_abs_err, float(shape_max_abs_err))
            max_rel_l2_err = max(max_rel_l2_err, float(shape_rel_l2_err))

            if passed:
                student_ms = benchmark(module.forward, W, X, A, B, warmup=warmup, iters=iters)
                torch_ms = benchmark(reference_impl, W, X, A, B, warmup=warmup, iters=iters)
                speedup = (torch_ms / student_ms) if student_ms > 0 else 0.0
                student_times.append(student_ms)
                torch_times.append(torch_ms)
                speedups.append(speedup)
                shape_entry.update(
                    {
                        "student_median_ms": student_ms,
                        "torch_median_ms": torch_ms,
                        "speedup": speedup,
                    }
                )
            else:
                all_shapes_passed = False
                shape_entry.update(
                    {
                        "student_median_ms": None,
                        "torch_median_ms": None,
                        "speedup": 0.0,
                    }
                )
            result["shape_results"].append(shape_entry)
        except Exception as exc:  # noqa: BLE001
            all_shapes_passed = False
            result["shape_results"].append(
                {
                    "d": d,
                    "correct": False,
                    "runtime_error": str(exc),
                    "student_median_ms": None,
                    "torch_median_ms": None,
                    "speedup": 0.0,
                }
            )

    result["correctness_passed"] = all_shapes_passed and len(result["shape_results"]) == len(resolved_shapes)
    result["max_abs_err"] = max_abs_err
    result["rel_l2_err"] = max_rel_l2_err
    result["mean_student_ms"] = mean(student_times) if student_times else None
    result["mean_torch_ms"] = mean(torch_times) if torch_times else None
    result["mean_speedup"] = mean(speedups) if speedups else 0.0
    result["min_speedup"] = min(speedups) if speedups else 0.0
    result["best_eligible"] = bool(result["compile_ok"] and result["correctness_passed"])
    result["score"] = compute_report_score(result)
    result["score_breakdown"] = build_score_breakdown(result)
    return _finalize_report(result, report_path)


def compute_report_score(report: dict[str, Any]) -> float:
    if not bool(report.get("compile_ok")):
        return -1_000_000_000.0
    if not bool(report.get("correctness_passed")):
        return -100_000_000.0

    shape_bonus = 1_000_000.0 if str(report.get("shape_preset", "")).lower() == "full" else 0.0
    mean_speedup = float(report.get("mean_speedup") or 0.0)
    min_speedup = float(report.get("min_speedup") or 0.0)
    mean_student_ms = float(report.get("mean_student_ms") or 1e9)
    return shape_bonus + mean_speedup * 10_000.0 + min_speedup * 1_000.0 - mean_student_ms


def build_score_breakdown(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "compile_ok": bool(report.get("compile_ok")),
        "correctness_passed": bool(report.get("correctness_passed")),
        "shape_preset": report.get("shape_preset"),
        "mean_speedup": float(report.get("mean_speedup") or 0.0),
        "min_speedup": float(report.get("min_speedup") or 0.0),
        "mean_student_ms": report.get("mean_student_ms"),
    }


def _finalize_report(result: dict[str, Any], report_path: Path | None) -> dict[str, Any]:
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        result = {**result, "report_path": str(report_path.resolve())}
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA candidate harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a candidate optimized_lora.cu")
    evaluate_parser.add_argument("--source-path", required=True)
    evaluate_parser.add_argument("--build-root", required=True)
    evaluate_parser.add_argument("--shape-preset", default="quick", choices=["smoke", "quick", "full"])
    evaluate_parser.add_argument("--shape-values", nargs="*", type=int, default=None)
    evaluate_parser.add_argument("--warmup", type=int, default=None)
    evaluate_parser.add_argument("--iters", type=int, default=None)
    evaluate_parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    evaluate_parser.add_argument("--report-path", type=str, default=None)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "evaluate":
        payload = evaluate_candidate(
            source_path=Path(args.source_path),
            build_root=Path(args.build_root),
            shape_preset=args.shape_preset,
            shape_values=args.shape_values,
            warmup=args.warmup,
            iters=args.iters,
            rank=args.rank,
            report_path=Path(args.report_path) if args.report_path else None,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
