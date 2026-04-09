#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace {

constexpr int kWarpSize = 32;
constexpr int kNoConflictStride = 1;
constexpr int kConflictStride = 32;
constexpr int kIterations = 1 << 15;
constexpr int kSharedElements = kWarpSize * kConflictStride;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA_ERROR,%s\n", cudaGetErrorString(err__)); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

__global__ void probe_shmem_kernel(
    int stride,
    unsigned long long* average_cycles,
    int* sinks) {
    if (blockIdx.x != 0 || threadIdx.x >= kWarpSize) {
        return;
    }

    const int lane = threadIdx.x;
    const int access_index = lane * stride;

    // volatile forces each shared-memory read to remain a real load instead of
    // being cached in a register by the optimizer.
    volatile __shared__ int shmem[kSharedElements];

    for (int i = lane; i < kSharedElements; i += kWarpSize) {
        shmem[i] = i + 1;
    }
    __syncthreads();

    int sum = 0;

    // Align the warp before timing so all lanes issue the measured load loop
    // together. With a single warp, this isolates bank conflict cost cleanly.
    __syncthreads();
    const unsigned long long start = clock64();

    #pragma unroll 1
    for (int iter = 0; iter < kIterations; ++iter) {
        sum += shmem[access_index];
    }

    const unsigned long long stop = clock64();
    __syncthreads();

    sinks[lane] = sum;

    if (lane == 0) {
        average_cycles[0] = (stop - start) / static_cast<unsigned long long>(kIterations);
    }
}

int run_probe(
    int stride,
    unsigned long long* d_cycles,
    int* d_sinks,
    unsigned long long* out_cycles) {
    unsigned long long h_cycles = 0;
    int h_sinks[kWarpSize] = {};

    probe_shmem_kernel<<<1, kWarpSize>>>(stride, d_cycles, d_sinks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        &h_cycles,
        d_cycles,
        sizeof(unsigned long long),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        h_sinks,
        d_sinks,
        sizeof(h_sinks),
        cudaMemcpyDeviceToHost));

    // Consume one sink on the host side so the compiler cannot treat the stores
    // as provably irrelevant.
    if (h_sinks[0] == -1) {
        std::fprintf(stderr, "UNREACHABLE,%d\n", h_sinks[0]);
    }

    out_cycles[0] = h_cycles;
    return 0;
}

}  // namespace

int main() {
    unsigned long long* d_cycles = nullptr;
    int* d_sinks = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cycles), sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sinks), sizeof(int) * kWarpSize));

    unsigned long long no_conflict_cycles = 0;
    unsigned long long conflict_cycles = 0;
    if (run_probe(kNoConflictStride, d_cycles, d_sinks, &no_conflict_cycles) != 0) {
        return 1;
    }
    if (run_probe(kConflictStride, d_cycles, d_sinks, &conflict_cycles) != 0) {
        return 1;
    }

    std::printf("Test_Type,Latency_Cycles\n");
    std::printf("No_Conflict_Stride_1,%llu\n", no_conflict_cycles);
    std::printf("32_Way_Conflict_Stride_32,%llu\n", conflict_cycles);

    CUDA_CHECK(cudaFree(d_cycles));
    CUDA_CHECK(cudaFree(d_sinks));
    return 0;
}
