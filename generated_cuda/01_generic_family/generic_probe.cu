#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Mode 1: 查询SM数量
void measure_sm_count() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    int sm_count;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    
    printf("launch__sm_count: %d\n", sm_count);
}

// Mode 2: 测量DRAM读取带宽
__global__ void dram_read_kernel(const float* __restrict__ src, float* __restrict__ dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    float sum = 0.0f;
    for (size_t i = idx; i < n; i += stride) {
        sum += src[i];
    }
    
    // 防止编译器优化掉读取
    if (sum != 0.0f) {
        dst[idx] = sum;
    }
}

void measure_dram_read_bandwidth() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    // 分配大缓冲区以避免缓存影响
    const size_t data_size = 512 * 1024 * 1024; // 512 MB
    const size_t n = data_size / sizeof(float);
    
    float *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, data_size));
    CUDA_CHECK(cudaMalloc(&d_dst, data_size));
    
    // 初始化数据
    CUDA_CHECK(cudaMemset(d_src, 1, data_size));
    CUDA_CHECK(cudaMemset(d_dst, 0, data_size));
    
    // 创建事件用于计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // 配置kernel
    int block_size = 256;
    int grid_size = 1024;
    
    // 预热
    dram_read_kernel<<<grid_size, block_size>>>(d_src, d_dst, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 多次测量取平均
    const int iterations = 10;
    float total_time_ms = 0.0f;
    
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        dram_read_kernel<<<grid_size, block_size>>>(d_src, d_dst, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        total_time_ms += elapsed_ms;
    }
    
    float avg_time_s = (total_time_ms / iterations) / 1000.0f;
    double bytes_per_second = (double)data_size / avg_time_s;
    
    printf("dram__bytes_read.sum.per_second: %.0f\n", bytes_per_second);
    
    // 清理
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
}

// Mode 3: 测量DRAM写入带宽
__global__ void dram_write_kernel(float* dst, size_t n, float value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        dst[i] = value;
    }
}

void measure_dram_write_bandwidth() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    const size_t data_size = 512 * 1024 * 1024; // 512 MB
    const size_t n = data_size / sizeof(float);
    
    float *d_dst;
    CUDA_CHECK(cudaMalloc(&d_dst, data_size));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    int block_size = 256;
    int grid_size = 1024;
    
    // 预热
    dram_write_kernel<<<grid_size, block_size>>>(d_dst, n, 1.0f);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    const int iterations = 10;
    float total_time_ms = 0.0f;
    
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        dram_write_kernel<<<grid_size, block_size>>>(d_dst, n, 1.0f);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        total_time_ms += elapsed_ms;
    }
    
    float avg_time_s = (total_time_ms / iterations) / 1000.0f;
    double bytes_per_second = (double)data_size / avg_time_s;
    
    printf("dram__bytes_write.sum.per_second: %.0f\n", bytes_per_second);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_dst));
}

// Mode 4: 查询GPU最大频率
void measure_max_gpu_frequency() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    int clock_rate_khz;
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, device));
    
    printf("device__attribute_max_gpu_frequency_khz: %d\n", clock_rate_khz);
}

// Mode 5: 查询内存最大频率
void measure_max_mem_frequency() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    int mem_clock_khz;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device));
    
    printf("device__attribute_max_mem_frequency_khz: %d\n", mem_clock_khz);
}

// Mode 6: 查询显存总线位宽
void measure_fb_bus_width() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    int bus_width;
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, device));
    
    printf("device__attribute_fb_bus_width: %d\n", bus_width);
}

// Mode 7: 运行所有测量
void measure_all() {
    measure_sm_count();
    measure_max_gpu_frequency();
    measure_max_mem_frequency();
    measure_fb_bus_width();
    measure_dram_read_bandwidth();
    measure_dram_write_bandwidth();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <mode>\n", argv[0]);
        printf("Modes:\n");
        printf("  sm_count      - Measure SM count\n");
        printf("  dram_read     - Measure DRAM read bandwidth\n");
        printf("  dram_write    - Measure DRAM write bandwidth\n");
        printf("  gpu_freq      - Measure max GPU frequency\n");
        printf("  mem_freq      - Measure max memory frequency\n");
        printf("  bus_width     - Measure framebuffer bus width\n");
        printf("  all           - Run all measurements\n");
        return 1;
    }
    
    const char* mode = argv[1];
    
    if (strcmp(mode, "sm_count") == 0) {
        measure_sm_count();
    } else if (strcmp(mode, "dram_read") == 0) {
        measure_dram_read_bandwidth();
    } else if (strcmp(mode, "dram_write") == 0) {
        measure_dram_write_bandwidth();
    } else if (strcmp(mode, "gpu_freq") == 0) {
        measure_max_gpu_frequency();
    } else if (strcmp(mode, "mem_freq") == 0) {
        measure_max_mem_frequency();
    } else if (strcmp(mode, "bus_width") == 0) {
        measure_fb_bus_width();
    } else if (strcmp(mode, "all") == 0) {
        measure_all();
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        return 1;
    }
    
    return 0;
}