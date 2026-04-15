#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Initialize pointer chasing array with random permutation
__global__ void init_pointers(int* ptr, int length, unsigned int seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        // Initialize sequential indices
        for (int i = 0; i < length; i++) {
            ptr[i] = i;
        }
        
        // Fisher-Yates shuffle for random permutation
        unsigned int r_state = seed;
        for (int i = length - 1; i > 0; i--) {
            r_state = r_state * 1103515245 + 12345;
            int j = r_state % (i + 1);
            if (j < 0) j += (i + 1);
            
            // Swap ptr[i] and ptr[j]
            int temp = ptr[i];
            ptr[i] = ptr[j];
            ptr[j] = temp;
        }
    }
}

// Pointer chasing kernel to measure L2 latency
__global__ void pointer_chase_kernel(int* __restrict__ ptr, int length, 
                                     int iterations, unsigned long long* total_cycles) {
    int idx = 0;  // Start from first element
    
    // Warmup iterations to ensure data is in cache
    #pragma unroll 1
    for (int i = 0; i < 100; i++) {
        idx = ptr[idx];
    }
    
    // Synchronize before measurement
    __syncthreads();
    
    // Start timing
    unsigned long long start = clock64();
    
    // Main measurement loop
    #pragma unroll 1
    for (int i = 0; i < iterations; i++) {
        idx = ptr[idx];
    }
    
    // End timing
    unsigned long long end = clock64();
    
    // Store result (prevent compiler optimization)
    if (threadIdx.x == 0) {
        *total_cycles = end - start;
        ptr[0] = idx;  // Use the result to prevent optimization
    }
}

int main() {
    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("Device: %s\n", prop.name);
    printf("L2 Cache Size: %d bytes\n", prop.l2CacheSize);
    
    // Calculate working set size
    // Target: 50-80% of L2 cache size to ensure L2 residency
    // Avoid L1 (typically 32KB-128KB) and DRAM
    int l2_size = prop.l2CacheSize;
    int working_set_size = (int)(l2_size * 0.6);  // 60% of L2
    
    // Ensure minimum size to avoid L1 (at least 256KB)
    if (working_set_size < 256 * 1024) {
        working_set_size = 256 * 1024;
    }
    
    // Number of int pointers
    int N = working_set_size / sizeof(int);
    
    printf("Working set size: %d bytes (%d elements)\n", working_set_size, N);
    printf("Target: %.1f%% of L2 cache\n", 100.0 * working_set_size / l2_size);
    
    // Allocate device memory
    int* d_ptr = nullptr;
    unsigned long long* d_cycles = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));
    
    // Initialize with random permutation
    unsigned int seed = (unsigned int)time(nullptr);
    init_pointers<<<1, 1>>>(d_ptr, N, seed);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Test parameters
    int iterations = 10000;
    int num_runs = 5;
    
    printf("\nRunning pointer chasing benchmark...\n");
    printf("Iterations per run: %d\n", iterations);
    printf("Number of runs: %d\n\n", num_runs);
    
    double total_avg_latency = 0.0;
    
    for (int run = 0; run < num_runs; run++) {
        // Clear cycles counter
        CUDA_CHECK(cudaMemset(d_cycles, 0, sizeof(unsigned long long)));
        
        // Launch kernel
        pointer_chase_kernel<<<1, 1>>>(d_ptr, N, iterations, d_cycles);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Get result
        unsigned long long cycles;
        CUDA_CHECK(cudaMemcpy(&cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        
        // Calculate average latency per access
        double avg_latency = (double)cycles / iterations;
        total_avg_latency += avg_latency;
        
        printf("Run %d: %llu total cycles, %.2f cycles/access\n", 
               run + 1, cycles, avg_latency);
    }
    
    double final_avg = total_avg_latency / num_runs;
    printf("\n========================================\n");
    printf("Average L2 Latency: %.2f cycles\n", final_avg);
    printf("========================================\n");
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_ptr));
    CUDA_CHECK(cudaFree(d_cycles));
    
    return 0;
}