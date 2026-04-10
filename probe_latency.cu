#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

namespace {

constexpr std::size_t kMinBytes = 4ULL * 1024ULL;
constexpr std::size_t kMaxBytes = 80ULL * 1024ULL * 1024ULL;
constexpr int kStrideBytes = 128;
constexpr int kWarmupSteps = 2048;
constexpr int kMeasureSteps = 1 << 20;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA_ERROR,%s\n", cudaGetErrorString(err__)); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

std::vector<std::size_t> build_working_set_sizes() {
    std::vector<std::size_t> sizes;
    for (std::size_t bytes = kMinBytes; bytes <= kMaxBytes; bytes <<= 1) {
        sizes.push_back(bytes);
    }
    if (sizes.empty() || sizes.back() != kMaxBytes) {
        sizes.push_back(kMaxBytes);
    }
    return sizes;
}

std::vector<int> build_pointer_chasing_ring(std::size_t working_set_bytes) {
    const int stride_elems = kStrideBytes / static_cast<int>(sizeof(int));
    const std::size_t total_elems = working_set_bytes / sizeof(int);
    const std::size_t node_count = total_elems / stride_elems;

    std::vector<int> data(total_elems, 0);
    std::vector<int> order(node_count);
    std::iota(order.begin(), order.end(), 0);

    // Fixed seed keeps runs reproducible while still defeating simple prefetch.
    std::mt19937 rng(0xC0FFEEu + static_cast<unsigned>(working_set_bytes));
    std::shuffle(order.begin(), order.end(), rng);

    for (std::size_t i = 0; i < node_count; ++i) {
        const int current = order[i] * stride_elems;
        const int next = order[(i + 1) % node_count] * stride_elems;
        data[current] = next;
    }
    return data;
}

__global__ void pointer_chase_kernel(
    const int* chain,
    int start_idx,
    int warmup_steps,
    int measure_steps,
    unsigned long long* total_cycles,
    int* sink) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int idx = start_idx;

    // Warm cache state into a stable condition before timing.
    for (int i = 0; i < warmup_steps; ++i) {
        idx = chain[idx];
    }

    __threadfence();
    asm volatile("" ::: "memory");
    const unsigned long long start = clock64();
    for (int i = 0; i < measure_steps; ++i) {
        idx = chain[idx];
    }
    const unsigned long long stop = clock64();
    asm volatile("" : "+r"(idx) :: "memory");

    total_cycles[0] = stop - start;
    sink[0] = idx;
}

}  // namespace

int main() {
    int* d_chain = nullptr;
    unsigned long long* d_cycles = nullptr;
    int* d_sink = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cycles), sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sink), sizeof(int)));

    std::vector<std::size_t> sizes = build_working_set_sizes();
    std::printf("Size_KB,Latency_Cycles\n");

    for (std::size_t working_set_bytes : sizes) {
        std::vector<int> h_chain = build_pointer_chasing_ring(working_set_bytes);

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_chain), h_chain.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(
            d_chain,
            h_chain.data(),
            h_chain.size() * sizeof(int),
            cudaMemcpyHostToDevice));

        pointer_chase_kernel<<<1, 1>>>(
            d_chain,
            0,
            kWarmupSteps,
            kMeasureSteps,
            d_cycles,
            d_sink);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned long long total_cycles = 0;
        int sink_value = 0;
        CUDA_CHECK(cudaMemcpy(
            &total_cycles,
            d_cycles,
            sizeof(unsigned long long),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            &sink_value,
            d_sink,
            sizeof(int),
            cudaMemcpyDeviceToHost));

        // Consume sink_value so the compiler cannot prove the chase is dead code.
        if (sink_value < 0) {
            std::fprintf(stderr, "UNREACHABLE,%d\n", sink_value);
        }

        const double latency_cycles =
            static_cast<double>(total_cycles) / static_cast<double>(kMeasureSteps);
        const std::size_t size_kb = working_set_bytes / 1024ULL;
        std::printf("%zu,%.2f\n", size_kb, latency_cycles);

        CUDA_CHECK(cudaFree(d_chain));
        d_chain = nullptr;
    }

    CUDA_CHECK(cudaFree(d_cycles));
    CUDA_CHECK(cudaFree(d_sink));
    return 0;
}
