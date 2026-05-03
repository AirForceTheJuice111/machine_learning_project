#include <torch/extension.h>
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
}

}  // namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    validate_inputs(W, X, A, B);
    auto BTX = torch::matmul(B.transpose(0, 1).contiguous(), X);
    auto Y1 = torch::matmul(W, X);
    auto Y2 = torch::matmul(A, BTX);
    return Y1 + Y2;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward (baseline)");
}
