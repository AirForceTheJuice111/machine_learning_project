from __future__ import annotations

import json
from pathlib import Path

FUSED_TILED_FP32_PROMPT = """You are writing a single-file CUDA extension for the LoRA forward pass:
Y = W @ X + A @ (B^T @ X)

Target:
- W(d,d), X(d,d), A(d,16), B(d,16), Y(d,d), all float32 on CUDA.
- d is in [3584, 4608].
- The goal is a genuinely fused CUDA implementation, not a composition of ATen matmul/addmm calls.

Required interface:
- Implement exactly:
  torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B)
- Export it with PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
- Return one complete self-contained .cu file only

Hard requirements:
- Do NOT use at::mm, at::matmul, at::addmm, or cuBLAS wrappers for the main computation.
- Use float32 loads, float32 accumulation, and float32 output.
- Use shared-memory tiling for W, X, and any reusable B/X data needed by the fused computation.
- Use register tiling: each thread should accumulate a small output micro-tile in registers.
- Keep intermediate data on chip whenever possible. Do not materialize WX or BTX in global memory.
- Make global-memory accesses coalesced and avoid obvious shared-memory bank conflicts.
- Handle all boundary tiles correctly.

Recommended design:
1. Launch a 2D thread-block grid over Y tiles.
2. For each output tile, cooperatively load tiles of W and X into shared memory and accumulate the WX term into registers.
3. In the same kernel, compute the low-rank contribution without writing BTX to global memory:
   - iterate over the K dimension in tiles
   - accumulate partial dot products for the 16 rank channels
   - reuse those partial values to update the same output-register tile
4. Store the final Y tile once.

Implementation guidance:
- A reasonable starting point is block tiles like 64x64 or 128x64 with thread-level micro-tiles such as 4x4 or 8x4.
- Specialize aggressively for rank=16.
- Use #pragma unroll on the rank loop where helpful.
- Prefer explicit kernels and indexing over template metaprogramming.
- Add a brief comment for the block tile shape and register tile shape.

Return the full .cu file only.
"""

FUSED_REGISTER_TILED_PROMPT = """Write a single-file CUDA extension for:
Y = W @ X + A @ (B^T @ X)

Primary goal:
- Produce a tightly fused kernel that uses hierarchical tiling:
  - global memory -> shared memory
  - shared memory -> registers
  - register micro-tile accumulation for Y

Constraints:
- Single self-contained .cu file
- forward(W, X, A, B) signature
- PYBIND11_MODULE binding
- float32 only
- no extra source/header files

Performance requirements:
- Do not decompose the operator into separate ATen matrix multiplications.
- Avoid global-memory materialization of intermediate WX or BTX.
- Each thread should compute multiple output elements in registers.
- Reuse X tile data across both the WX path and the low-rank path.
- Use shared memory for block tiles and structure accesses to avoid bank conflicts.

Suggested structure:
- One main fused kernel for the general case
- Optional small helper kernels only for edge handling or layout transforms if absolutely necessary
- Explicitly unroll the rank-16 dimension

Correctness requirements:
- Must pass torch.allclose(rtol=1e-4, atol=1e-4)
- Must handle arbitrary d in [3584, 4608]
- Must avoid undefined behavior on boundary tiles

If the full fusion becomes too complex, prefer a partially fused kernel that still avoids writing the biggest intermediate tensors to HBM.

Return the complete .cu file only.
"""

TENSORCORE_WMMA_PROMPT = """Write a single-file CUDA extension for the fused LoRA forward pass on Ampere:
Y = W @ X + A @ (B^T @ X)

Goal:
- Explore an aggressive fused design using Tensor Cores / WMMA where useful, while keeping the final output numerically close enough to pass torch.allclose(rtol=1e-4, atol=1e-4).

Requirements:
- Single self-contained .cu file
- forward(W, X, A, B)
- PYBIND11_MODULE binding
- Keep the output tensor in float32
- Use float32 accumulation
- Avoid writing large intermediates such as WX or BTX to global memory

Design direction:
- Tile the output in WMMA-friendly shapes
- Reuse shared-memory tiles across the WX path and the low-rank path
- Treat rank=16 as a first-class specialization
- Keep boundary handling explicit and safe

Important:
- Favor a correct, compilable aggressive candidate over pseudo-code.
- If mixed-precision input staging is used, keep accumulation and output in float32.
- Do not rely on external files or helper libraries beyond standard CUDA / PyTorch extension headers.

Return the full .cu file only.
"""

LOWRANK_OUTER_FALLBACK_PROMPT = """Generate a fallback single-file CUDA extension for:
Y = W @ X + A @ (B^T @ X)

Use this direction only if aggressive fused kernels keep failing.

Goal:
- Still reduce memory traffic relative to a naive ATen composition, but allow a more conservative structure.
- Prefer partial fusion and explicit CUDA kernels over pure at::mm + at::addmm composition.

Requirements:
- Single self-contained .cu file
- forward(W, X, A, B)
- PYBIND11_MODULE binding
- float32 only
- no extra source/header files

Preferred fallback shape:
- Keep at most one large intermediate in global memory
- Make the final low-rank update a custom CUDA kernel
- Specialize the rank-16 path with unrolled accumulation

Return the full .cu file only.
"""


def _common_prelude() -> str:
    return r"""#include <torch/extension.h>
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


def build_generation_prompt_catalog() -> list[dict[str, str]]:
    return [
        {
            "id": "fused_tiled_fp32",
            "description": "Single-kernel fused FP32 design with shared-memory tiling and register micro-tiles",
            "risk_level": "high",
            "recommended_when": "First aggressive candidate after measuring the baseline",
            "prompt": FUSED_TILED_FP32_PROMPT,
        },
        {
            "id": "lowrank_outer_fallback",
            "description": "Fallback partial-fusion path when aggressive fused kernels repeatedly fail",
            "risk_level": "medium",
            "recommended_when": "Use early as a recovery path after the first fused candidate fails correctness",
            "prompt": LOWRANK_OUTER_FALLBACK_PROMPT,
        },
        {
            "id": "fused_register_tiled",
            "description": "Hierarchical fused kernel with shared-memory tiles and thread-level register tiling",
            "risk_level": "high",
            "recommended_when": "Try after a stable fallback exists, or when the first fused kernel compiles but still underutilizes on-chip storage",
            "prompt": FUSED_REGISTER_TILED_PROMPT,
        },
        {
            "id": "tensorcore_wmma",
            "description": "Aggressive fused Tensor Core / WMMA candidate with float32 accumulation",
            "risk_level": "high",
            "recommended_when": "When fused FP32 kernels are correct but speedup is still not strong enough",
            "prompt": TENSORCORE_WMMA_PROMPT,
        },
    ]


def resolve_generation_prompt(prompt_id: str) -> dict[str, str]:
    normalized_id = str(prompt_id).strip()
    for item in build_generation_prompt_catalog():
        if item["id"] == normalized_id:
            return item
    available = ", ".join(item["id"] for item in build_generation_prompt_catalog())
    raise KeyError(f"未知 generation prompt id: {normalized_id}. 可选值: {available}")


def get_generation_prompts(manifest_path: str | Path) -> list[dict[str, str]]:
    resolved_path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    prompts = payload.get("generation_prompts", [])
    if not isinstance(prompts, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in prompts:
        if not isinstance(item, dict):
            continue
        prompt_id = str(item.get("id") or "").strip()
        description = str(item.get("description") or "").strip()
        try:
            catalog_item = resolve_generation_prompt(prompt_id)
        except KeyError:
            catalog_item = None
        prompt = catalog_item["prompt"] if catalog_item is not None else str(item.get("prompt") or "").strip()
        if prompt_id and prompt:
            normalized.append(
                {
                    "id": prompt_id,
                    "description": description,
                    "prompt": prompt,
                }
            )
    return normalized


def write_seed_candidate_templates(candidates_dir: Path) -> dict[str, str]:
    candidates_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    for item in build_seed_template_catalog():
        path = candidates_dir / item["file_name"]
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
    manifest_payload = {
        "templates": manifest,
        "generation_prompts": [
            {
                "id": item["id"],
                "description": item["description"],
                "risk_level": item["risk_level"],
                "recommended_when": item["recommended_when"],
            }
            for item in build_generation_prompt_catalog()
        ],
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "template_count": str(len(manifest)),
        "generation_prompt_count": str(len(manifest_payload["generation_prompts"])),
    }
