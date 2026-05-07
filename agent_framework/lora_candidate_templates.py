from __future__ import annotations

import json
from pathlib import Path

FUSED_V1_PROMPT = """You are a CUDA optimization expert. Your task is to write a single-file CUDA kernel that implements the LoRA forward pass:
Y = W @ X + A @ (B^T @ X)
where:
- W (d x d), X (d x d), A (d x 16), B (d x 16), Y (d x d), all float32.
- d is a multiple of 16, within [3584, 4608].

**Core constraints**:
- Output ONE self-contained .cu file with the interface:
  torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B)
- The file must compile with torch.utils.cpp_extension.load.
- Do NOT use any external headers except <torch/extension.h> and standard CUDA/C++ headers.
- The kernel must produce exactly Y = WX + A(B^T X) with numerical error < 1e-4.

**Optimization strategy**:
Fuse the three stages into a single kernel launch. For each output tile (tileSize = 64 or 128):
1. Load a tile of W and a tile of X into shared memory, compute the WX contribution into registers.
2. Immediately compute the LoRA contribution for that same tile:
   - For each rank k in 0..15, load column k of A and the corresponding row of (B^T X) (obtained via a small tile of B and X), compute the outer product and add to the same register accumulator.
3. Write the final tile of Y to global memory.

Use cooperative groups or standard threadblock synchronization. No intermediate global memory for BTX or A*BTX. The kernel must handle arbitrary d inside the given range, with a simple launch grid (e.g., dim3 grid(ceil(d/tileSize), ceil(d/tileSize)), block(tileSize, tileSize)).
Provide the complete .cu file. Include proper PyTorch binding via PYBIND11_MODULE.
"""

LOWRANK_OUTER_PROMPT = """You are a CUDA performance engineer. Generate a single-file CUDA extension for the LoRA forward pass:
Y = W @ X + A @ (B^T @ X)   with rank r=16, all tensors float32, d in [3584, 4608].

**Requirements**:
- Function signature: torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B)
- The .cu file must be self-contained, without extra .cuh/.h files.
- Use proper PyTorch extension bindings (PYBIND11_MODULE).

**Kernel design**:
- Use an output-tile parallelization: each thread block computes a TILE_X × TILE_Y (e.g., 64×64) block of Y.
- Decompose A(B^T X) into sum of 16 outer products:
    for k in 0..15:  Y += A[:,k] * (B[:,k]^T @ X)
- In the kernel:
  1. Load the necessary tile of W and tile of X into shared memory, compute WX contribution into registers.
  2. For each k, load the required segment of A[:,k] and compute the partial product with the already-loaded X tile (using B’s corresponding row). Do this using shared memory for A_column_k and a small buffer for the B^T X intermediate.
  3. Accumulate all contributions into registers, then write the final tile to Y.

- Choose tile sizes so that shared memory per block stays under 48 KB (Ampere). Use float4 reads/writes for coalesced access.
- Add comments explaining tile choices and synchronization.
Output the complete .cu code.
"""

WMMA_TENSORCORE_PROMPT = """You are a CUDA Tensor Core specialist. Write a self-contained .cu file that implements the LoRA forward pass on Ampere (SM 8.0, RTX 3090):
Y = W @ X + A @ (B^T @ X),  d in [3584,4608], r=16, all inputs/outputs float32.

**Interface**:
torch::Tensor forward(torch::Tensor W, torch::Tensor X, torch::Tensor A, torch::Tensor B)

**Optimization targets**:
- Use nvcuda::wmma for 16x16x16 matrix multiply-add operations on Tensor Cores.
- Perform WX and LoRA contribution using WMMA with FP16 inputs and FP32 accumulation.
- Fuse the two stages in one kernel: compute a 16x16 tile of Y, then immediately add the LoRA contribution (A(B^T X)) to the same tile without writing to global memory.

**Implementation plan**:
1. Load 16x16 tiles of W and X (converted to half) into WMMA fragments, compute with mma_sync.
2. For the LoRA part, for each rank k (0..15):
   - Load the 16-element column of A and the corresponding row of (B^T X) (computed on-the-fly from a 16x16 tile of B and X using another WMMA call or manual half-precision multiply-add).
   - Accumulate the outer product using the same WMMA fragment.
3. Store the final FP32 tile to Y.

**Constraints**:
- Must compile with torch.utils.cpp_extension.load.
- Use only <torch/extension.h>, standard CUDA headers, and <cuda_fp16.h>, <mma.h>.
- Output must pass torch.allclose(rtol=1e-4, atol=1e-4) against the float32 reference.
- Ensure proper threadblock and grid sizes for arbitrary d (pad or handle boundary tiles correctly).

Provide the full .cu file.
"""


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


def build_generation_prompt_catalog() -> list[dict[str, str]]:
    return [
        {
            "id": "fused_v1",
            "description": "Fused LoRA single kernel, tile size 64 or 128",
            "prompt": FUSED_V1_PROMPT,
        },
        {
            "id": "lowrank_outer",
            "description": "Tiled low-rank outer product fused kernel",
            "prompt": LOWRANK_OUTER_PROMPT,
        },
        {
            "id": "wmma_tensorcore",
            "description": "Tensor Core WMMA fused kernel (FP16/FP32)",
            "prompt": WMMA_TENSORCORE_PROMPT,
        },
    ]


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
        prompt = str(item.get("prompt") or "").strip()
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
    manifest_payload = {
        "templates": manifest,
        "generation_prompts": build_generation_prompt_catalog(),
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "template_count": str(len(manifest)),
        "generation_prompt_count": str(len(manifest_payload["generation_prompts"])),
    }
