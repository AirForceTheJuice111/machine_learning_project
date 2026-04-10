#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdint>

namespace {

constexpr int kWarmupIterations = 1 << 24;
constexpr int kMeasureIterations = 1 << 27;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA_ERROR,%s\n", cudaGetErrorString(err__)); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

__global__ void warmup_kernel(float* sink) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    float x0 = 1.0f;
    float x1 = 2.0f;
    float x2 = 3.0f;
    float x3 = 4.0f;
    float x4 = 5.0f;
    float x5 = 6.0f;
    float x6 = 7.0f;
    float x7 = 8.0f;

    // Independent FMA chains keep a single thread busy long enough to pull the
    // GPU out of idle and into a stable higher-performance state.
    for (int i = 0; i < kWarmupIterations; ++i) {
        x0 = fmaf(x0, 1.000001f, 0.000001f);
        x1 = fmaf(x1, 1.000002f, 0.000002f);
        x2 = fmaf(x2, 1.000003f, 0.000003f);
        x3 = fmaf(x3, 1.000004f, 0.000004f);
        x4 = fmaf(x4, 1.000005f, 0.000005f);
        x5 = fmaf(x5, 1.000006f, 0.000006f);
        x6 = fmaf(x6, 1.000007f, 0.000007f);
        x7 = fmaf(x7, 1.000008f, 0.000008f);
    }

    sink[0] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}

__global__ void measure_frequency_kernel(
    unsigned long long* elapsed_cycles,
    float* sink) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    float x0 = 1.0f;
    float x1 = 2.0f;
    float x2 = 3.0f;
    float x3 = 4.0f;
    float x4 = 5.0f;
    float x5 = 6.0f;
    float x6 = 7.0f;
    float x7 = 8.0f;

    __threadfence();
    asm volatile("" ::: "memory");
    const unsigned long long start = clock64();

    // The loop body is deliberately compact and arithmetic-heavy. A long run
    // time makes kernel launch overhead negligible in the host-side timing.
    for (int i = 0; i < kMeasureIterations; ++i) {
        x0 = fmaf(x0, 1.000001f, 0.000001f);
        x1 = fmaf(x1, 1.000002f, 0.000002f);
        x2 = fmaf(x2, 1.000003f, 0.000003f);
        x3 = fmaf(x3, 1.000004f, 0.000004f);
        x4 = fmaf(x4, 1.000005f, 0.000005f);
        x5 = fmaf(x5, 1.000006f, 0.000006f);
        x6 = fmaf(x6, 1.000007f, 0.000007f);
        x7 = fmaf(x7, 1.000008f, 0.000008f);
    }

    const unsigned long long stop = clock64();
    asm volatile("" ::: "memory");

    elapsed_cycles[0] = stop - start;
    sink[0] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}

}  // namespace

int main() {
    unsigned long long* d_elapsed_cycles = nullptr;
    float* d_sink = nullptr;
    unsigned long long h_elapsed_cycles = 0;
    float h_sink = 0.0f;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_elapsed_cycles), sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sink), sizeof(float)));

    warmup_kernel<<<1, 1>>>(d_sink);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto host_start = std::chrono::high_resolution_clock::now();
    measure_frequency_kernel<<<1, 1>>>(d_elapsed_cycles, d_sink);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto host_stop = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(
        &h_elapsed_cycles,
        d_elapsed_cycles,
        sizeof(unsigned long long),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        &h_sink,
        d_sink,
        sizeof(float),
        cudaMemcpyDeviceToHost));

    // Consume the final value so the compiler cannot discard the compute loop.
    if (h_sink == -1.0f) {
        std::fprintf(stderr, "UNREACHABLE,%.9f\n", h_sink);
    }

    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
            host_stop - host_start)
            .count();
    const double frequency_mhz =
        static_cast<double>(h_elapsed_cycles) / elapsed_us;

    std::printf("Metric,Value\n");
    std::printf("Actual_Boost_Frequency_MHz,%.3f\n", frequency_mhz);

    CUDA_CHECK(cudaFree(d_elapsed_cycles));
    CUDA_CHECK(cudaFree(d_sink));
    return 0;
}
