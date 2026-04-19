#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Kernel for DRAM read bandwidth test - optimized for maximum bandwidth
__global__ void dram_read_kernel(const float* __restrict__ src, float* __restrict__ dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Use multiple accumulators to hide latency
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    
    for (size_t i = idx; i < n - 3; i += stride * 4) {
        sum0 += src[i];
        sum1 += src[i + stride];
        sum2 += src[i + stride * 2];
        sum3 += src[i + stride * 3];
    }
    
    // Prevent optimization
    float total = sum0 + sum1 + sum2 + sum3;
    if (total > 1e30f) {
        dst[0] = total;
    }
}

// Kernel for DRAM write bandwidth test - optimized for maximum bandwidth
__global__ void dram_write_kernel(float* __restrict__ dst, size_t n, float value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Vectorized write pattern
    for (size_t i = idx; i < n; i += stride) {
        dst[i] = value;
    }
}

void measure_sm_count() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("launch__sm_count: %d\n", prop.multiProcessorCount);
}

void measure_gpu_frequency() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    int clock_rate_khz = 0;
    cudaError_t err = cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, device);
    
    if (err == cudaSuccess && clock_rate_khz > 0) {
        printf("device__attribute_max_gpu_frequency_khz: %d\n", clock_rate_khz);
    } else {
        printf("device__attribute_max_gpu_frequency_khz: 0\n");
    }
}

void measure_mem_frequency() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    int mem_clock_rate_khz = 0;
    cudaError_t err = cudaDeviceGetAttribute(&mem_clock_rate_khz, cudaDevAttrMemoryClockRate, device);
    
    if (err == cudaSuccess && mem_clock_rate_khz > 0) {
        printf("device__attribute_max_mem_frequency_khz: %d\n", mem_clock_rate_khz);
    } else {
        printf("device__attribute_max_mem_frequency_khz: 0\n");
    }
}

void measure_bus_width() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("device__attribute_fb_bus_width: %d\n", prop.memoryBusWidth);
}

void measure_dram_read_bandwidth_ncu(size_t data_size_mb) {
    size_t data_size = data_size_mb * 1024 * 1024;
    size_t num_elements = data_size / sizeof(float);
    
    float *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, data_size));
    CUDA_CHECK(cudaMalloc(&d_dst, sizeof(float)));
    
    CUDA_CHECK(cudaMemset(d_src, 1, data_size));
    
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    int block_size = 256;
    int grid_size = prop.multiProcessorCount * 32;
    
    // Single kernel launch for NCU profiling
    dram_read_kernel<<<grid_size, block_size>>>(d_src, d_dst, num_elements);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
}

void measure_dram_write_bandwidth_ncu(size_t data_size_mb) {
    size_t data_size = data_size_mb * 1024 * 1024;
    size_t num_elements = data_size / sizeof(float);
    
    float *d_dst;
    CUDA_CHECK(cudaMalloc(&d_dst, data_size));
    
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    int block_size = 256;
    int grid_size = prop.multiProcessorCount * 32;
    
    // Single kernel launch for NCU profiling
    dram_write_kernel<<<grid_size, block_size>>>(d_dst, num_elements, 1.0f);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_dst));
}

void measure_dram_read_bandwidth_timing(size_t data_size_mb, int iterations) {
    size_t data_size = data_size_mb * 1024 * 1024;
    size_t num_elements = data_size / sizeof(float);
    
    float *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, data_size));
    CUDA_CHECK(cudaMalloc(&d_dst, sizeof(float)));
    
    CUDA_CHECK(cudaMemset(d_src, 1, data_size));
    
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    int block_size = 256;
    int grid_size = prop.multiProcessorCount * 32;
    
    // Warmup
    dram_read_kernel<<<grid_size, block_size>>>(d_src, d_dst, num_elements);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    float total_time_ms = 0.0f;
    
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        
        dram_read_kernel<<<grid_size, block_size>>>(d_src, d_dst, num_elements);
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        total_time_ms += elapsed_ms;
    }
    
    float avg_time_s = (total_time_ms / iterations) / 1000.0f;
    double bandwidth_bytes_per_sec = (double)data_size / avg_time_s;
    
    printf("dram__bytes_read.sum.per_second: %.0f\n", bandwidth_bytes_per_sec);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
}

void measure_dram_write_bandwidth_timing(size_t data_size_mb, int iterations) {
    size_t data_size = data_size_mb * 1024 * 1024;
    size_t num_elements = data_size / sizeof(float);
    
    float *d_dst;
    CUDA_CHECK(cudaMalloc(&d_dst, data_size));
    
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    int block_size = 256;
    int grid_size = prop.multiProcessorCount * 32;
    
    // Warmup
    dram_write_kernel<<<grid_size, block_size>>>(d_dst, num_elements, 1.0f);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    float total_time_ms = 0.0f;
    
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        
        dram_write_kernel<<<grid_size, block_size>>>(d_dst, num_elements, 1.0f);
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        total_time_ms += elapsed_ms;
    }
    
    float avg_time_s = (total_time_ms / iterations) / 1000.0f;
    double bandwidth_bytes_per_sec = (double)data_size / avg_time_s;
    
    printf("dram__bytes_write.sum.per_second: %.0f\n", bandwidth_bytes_per_sec);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_dst));
}

void print_usage(const char* prog_name) {
    printf("Usage: %s <mode> [options]\n", prog_name);
    printf("Modes:\n");
    printf("  sm_count              - Query SM count\n");
    printf("  gpu_freq              - Query max GPU frequency (kHz)\n");
    printf("  mem_freq              - Query max memory frequency (kHz)\n");
    printf("  bus_width             - Query memory bus width (bits)\n");
    printf("  dram_read_ncu [mb]    - DRAM read kernel for NCU profiling\n");
    printf("  dram_write_ncu [mb]   - DRAM write kernel for NCU profiling\n");
    printf("  dram_read_timing [mb] - DRAM read bandwidth via timing\n");
    printf("  dram_write_timing [mb] - DRAM write bandwidth via timing\n");
    printf("  all                   - Run all measurements\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    const char* mode = argv[1];
    
    if (strcmp(mode, "sm_count") == 0) {
        measure_sm_count();
    }
    else if (strcmp(mode, "gpu_freq") == 0) {
        measure_gpu_frequency();
    }
    else if (strcmp(mode, "mem_freq") == 0) {
        measure_mem_frequency();
    }
    else if (strcmp(mode, "bus_width") == 0) {
        measure_bus_width();
    }
    else if (strcmp(mode, "dram_read_ncu") == 0) {
        size_t size_mb = (argc > 2) ? atoi(argv[2]) : 512;
        measure_dram_read_bandwidth_ncu(size_mb);
    }
    else if (strcmp(mode, "dram_write_ncu") == 0) {
        size_t size_mb = (argc > 2) ? atoi(argv[2]) : 512;
        measure_dram_write_bandwidth_ncu(size_mb);
    }
    else if (strcmp(mode, "dram_read_timing") == 0) {
        size_t size_mb = (argc > 2) ? atoi(argv[2]) : 512;
        measure_dram_read_bandwidth_timing(size_mb, 10);
    }
    else if (strcmp(mode, "dram_write_timing") == 0) {
        size_t size_mb = (argc > 2) ? atoi(argv[2]) : 512;
        measure_dram_write_bandwidth_timing(size_mb, 10);
    }
    else if (strcmp(mode, "all") == 0) {
        measure_sm_count();
        measure_gpu_frequency();
        measure_mem_frequency();
        measure_bus_width();
        measure_dram_read_bandwidth_timing(512, 10);
        measure_dram_write_bandwidth_timing(512, 10);
    }
    else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        print_usage(argv[0]);
        return 1;
    }
    
    return 0;
}