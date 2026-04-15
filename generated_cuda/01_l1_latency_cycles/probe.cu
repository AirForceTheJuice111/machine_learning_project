#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>

// Improved L1 latency measurement
// Use clock64() for better precision and subtract loop overhead

__global__ void l1_latency_kernel(uint32_t* data, uint32_t* result, 
                                   uint32_t chase_count, uint32_t warmup_count) {
    // Start from index 0
    uint32_t idx = 0;
    
    // Warmup phase: ensure data is in L1 cache
    #pragma unroll 1
    for (uint32_t i = 0; i < warmup_count; i++) {
        idx = data[idx];
    }
    
    // Synchronize before measurement
    __syncwarp();
    
    // Measurement phase with pointer chasing
    long long start = clock64();
    
    #pragma unroll 1
    for (uint32_t i = 0; i < chase_count; i++) {
        idx = data[idx];
    }
    
    long long end = clock64();
    
    // Store result (prevent optimization)
    result[0] = idx;
    result[1] = (uint32_t)(end - start);
}

// Kernel to measure loop overhead (no memory access)
__global__ void loop_overhead_kernel(uint32_t* result, uint32_t count) {
    uint32_t idx = 0;
    
    long long start = clock64();
    
    #pragma unroll 1
    for (uint32_t i = 0; i < count; i++) {
        idx = (idx + 1) % 1024;  // Simple arithmetic, no memory access
    }
    
    long long end = clock64();
    
    result[0] = idx;
    result[1] = (uint32_t)(end - start);
}

int main() {
    // L1 cache size is typically 32KB-128KB depending on GPU
    // Use a small working set: 4KB (1024 uint32_t elements)
    // This ensures high L1 hit rate
    
    const uint32_t WORKING_SET_SIZE = 1024;  // 4KB
    const uint32_t WARMUP_ITERATIONS = 10000;
    const uint32_t MEASUREMENT_ITERATIONS = 1000000;
    
    // Allocate host memory
    uint32_t* h_data = (uint32_t*)malloc(WORKING_SET_SIZE * sizeof(uint32_t));
    uint32_t* h_result = (uint32_t*)malloc(2 * sizeof(uint32_t));
    
    // Create a pointer chasing pattern with stride
    // Use a simple sequential pattern first
    for (uint32_t i = 0; i < WORKING_SET_SIZE; i++) {
        h_data[i] = (i + 1) % WORKING_SET_SIZE;
    }
    
    // Allocate device memory
    uint32_t* d_data;
    uint32_t* d_result;
    cudaMalloc(&d_data, WORKING_SET_SIZE * sizeof(uint32_t));
    cudaMalloc(&d_result, 2 * sizeof(uint32_t));
    
    // Copy data to device
    cudaMemcpy(d_data, h_data, WORKING_SET_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Measure L1 latency with pointer chasing
    l1_latency_kernel<<<1, 1>>>(d_data, d_result, MEASUREMENT_ITERATIONS, WARMUP_ITERATIONS);
    cudaDeviceSynchronize();
    cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    long long total_cycles_with_mem = h_result[1];
    
    // Measure loop overhead
    loop_overhead_kernel<<<1, 1>>>(d_result, MEASUREMENT_ITERATIONS);
    cudaDeviceSynchronize();
    cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    long long total_cycles_overhead = h_result[1];
    
    // Calculate latency
    long long memory_cycles = total_cycles_with_mem - total_cycles_overhead;
    double avg_latency = (double)memory_cycles / MEASUREMENT_ITERATIONS;
    
    printf("L1 Latency Measurement Results:\n");
    printf("Working set size: %u elements (%u KB)\n", WORKING_SET_SIZE, (WORKING_SET_SIZE * 4) / 1024);
    printf("Warmup iterations: %u\n", WARMUP_ITERATIONS);
    printf("Measurement iterations: %u\n", MEASUREMENT_ITERATIONS);
    printf("Total cycles (with memory): %lld\n", total_cycles_with_mem);
    printf("Total cycles (overhead): %lld\n", total_cycles_overhead);
    printf("Memory access cycles: %lld\n", memory_cycles);
    printf("Average L1 latency: %.2f cycles\n", avg_latency);
    
    // Cleanup
    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);
    free(h_result);
    
    return 0;
}