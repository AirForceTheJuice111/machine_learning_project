from __future__ import annotations

import json
from pathlib import Path


def _common_prelude() -> str:
    return r"""#include <torch/extension.h>
#include <ATen/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
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
"""


def build_seed_template_catalog() -> list[dict[str, str]]:
    prelude = _common_prelude()
    return [
        {
            "file_name": "seed_aten_mm.cu",
            "strategy": "baseline_aten_mm",
            "description": "最稳的 correctness-first 基线。保留 W@X 和低秩路径都走 ATen mm。",
            "content": prelude
            + r"""
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    validate_inputs(W, X, A, B);
    at::cuda::OptionalCUDAGuard device_guard(device_of(W));
    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    auto BT = Bc.transpose(0, 1).contiguous();
    auto BTX = at::mm(BT, Xc);
    auto WX = at::mm(Wc, Xc);
    auto ABTX = at::mm(Ac, BTX);
    return (WX + ABTX).contiguous();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward baseline");
}
""",
        },
        {
            "file_name": "seed_addmm_rank16.cu",
            "strategy": "low_rank_addmm",
            "description": "仍让 W@X 走 ATen mm，但将 A @ (B^T X) 合并进 addmm，适合作为第一批低风险优化候选。",
            "content": prelude
            + r"""
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    validate_inputs(W, X, A, B);
    at::cuda::OptionalCUDAGuard device_guard(device_of(W));
    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    auto BT = Bc.transpose(0, 1).contiguous();
    auto BTX = at::mm(BT, Xc);          // [r, d]
    auto WX = at::mm(Wc, Xc);           // [d, d]
    return at::addmm(WX, Ac, BTX, 1.0, 1.0).contiguous();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward addmm rank16");
}
""",
        },
        {
            "file_name": "seed_lowrank_epilogue.cu",
            "strategy": "custom_lowrank_epilogue",
            "description": "保留 W@X 走 ATen mm，仅对 rank=16 的低秩更新与最终加和写自定义 CUDA kernel，适合第二阶段尝试。",
            "content": prelude
            + r"""
namespace {

__global__ void add_low_rank_epilogue_kernel(
    const float* __restrict__ WX,
    const float* __restrict__ A,
    const float* __restrict__ BTX,
    float* __restrict__ Y,
    int d,
    int r) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= d || col >= d) {
        return;
    }

    float acc = WX[row * d + col];
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        if (k < r) {
            acc += A[row * r + k] * BTX[k * d + col];
        }
    }
    Y[row * d + col] = acc;
}

}  // namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    validate_inputs(W, X, A, B);
    at::cuda::OptionalCUDAGuard device_guard(device_of(W));
    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    const int64_t d = Wc.size(0);
    const int64_t r = Ac.size(1);
    auto BT = Bc.transpose(0, 1).contiguous();
    auto BTX = at::mm(BT, Xc);          // [r, d]
    auto WX = at::mm(Wc, Xc);           // [d, d]
    auto Y = at::empty_like(WX);

    constexpr int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid(static_cast<unsigned int>((d + TILE - 1) / TILE),
              static_cast<unsigned int>((d + TILE - 1) / TILE));
    cudaStream_t stream = at::cuda::getDefaultCUDAStream(W.device().index()).stream();
    add_low_rank_epilogue_kernel<<<grid, block, 0, stream>>>(
        WX.data_ptr<float>(),
        Ac.data_ptr<float>(),
        BTX.data_ptr<float>(),
        Y.data_ptr<float>(),
        static_cast<int>(d),
        static_cast<int>(r));
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "add_low_rank_epilogue_kernel launch failed: ", cudaGetErrorString(err));
    return Y.contiguous();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward custom low-rank epilogue");
}
""",
        },
    ]


def write_seed_candidate_templates(candidates_dir: Path) -> dict[str, str]:
    candidates_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    for item in build_seed_template_catalog():
        path = candidates_dir / item["file_name"]
        if not path.exists():
            path.write_text(item["content"], encoding="utf-8")
        manifest.append(
            {
                "file_name": item["file_name"],
                "path": str(path.resolve()),
                "strategy": item["strategy"],
                "description": item["description"],
            }
        )

    manifest_path = candidates_dir / "seed_manifest.json"
    manifest_path.write_text(json.dumps({"templates": manifest}, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "template_count": str(len(manifest)),
    }
