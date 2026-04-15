#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Pointer chasing kernel - serial access to avoid parallel hiding latency
__global__ void pointer_chase_kernel(volatile unsigned long long* ptr_chain, 
                                       unsigned long long* result,
                                       unsigned long long num_elements,
                                       unsigned long long iterations) {
    // Single thread performs the chase
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long current = 0;
        
        // Warm up
        for (unsigned long long i = 0; i < num_elements; i++) {
            current = ptr_chain[current];
        }
        
        // Actual measurement
        unsigned long long start_time, end_time;
        
        // Use CUDA clock for timing
        start_time = clock64();
        
        for (unsigned long long iter = 0; iter < iterations; iter++) {
            for (unsigned long long i = 0; i < num_elements; i++) {
                current = ptr_chain[current];
            }
        }
        
        end_time = clock64();
        
        // Store results
        result[0] = end_time - start_time;
        result[1] = current; // Prevent optimization
        result[2] = iterations * num_elements;
    }
}

// Initialize pointer chain with random permutation to defeat prefetching
void initialize_pointer_chain(unsigned long long* h_chain, unsigned long long num_elements) {
    // Create sequential indices
    for (unsigned long long i = 0; i < num_elements; i++) {
        h_chain[i] = i;
    }
    
    // Fisher-Yates shuffle for random permutation
    srand(42); // Fixed seed for reproducibility
    for (unsigned long long i = num_elements - 1; i > 0; i--) {
        unsigned long long j = rand() % (i + 1);
        unsigned long long temp = h_chain[i];
        h_chain[i] = h_chain[j];
        h_chain[j] = temp;
    }
}

int main() {
    // Configuration - use large working set to exceed cache
    // L2 cache on modern GPUs is typically 4-40 MB
    // Use 256 MB to ensure DRAM access
    const unsigned long long working_set_mb = 256;
    const unsigned long long num_elements = (working_set_mb * 1024 * 1024) / sizeof(unsigned long long);
    const unsigned long long iterations = 10; // Multiple iterations for stable measurement
    
    printf("DRAM Latency Measurement via Pointer Chasing\n");
    printf("Working set size: %llu MB\n", working_set_mb);
    printf("Number of elements: %llu\n", num_elements);
    printf("Iterations: %llu\n", iterations);
    
    // Allocate host memory
    unsigned long long* h_chain = (unsigned long long*)malloc(num_elements * sizeof(unsigned long long));
    unsigned long long* h_result = (unsigned long long*)malloc(3 * sizeof(unsigned long long));
    
    if (!h_chain || !h_result) {
        fprintf(stderr, "Host allocation failed\n");
        return 1;
    }
    
    // Initialize pointer chain
    printf("Initializing pointer chain...\n");
    initialize_pointer_chain(h_chain, num_elements);
    
    // Allocate device memory
    unsigned long long* d_chain;
    unsigned long long* d_result;
    
    CUDA_CHECK(cudaMalloc(&d_chain, num_elements * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_result, 3 * sizeof(unsigned long long)));
    
    // Copy to device
    printf("Copying to device...\n");
    CUDA_CHECK(cudaMemcpy(d_chain, h_chain, num_elements * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    
    // Launch kernel
    printf("Launching pointer chase kernel...\n");
    pointer_chase_kernel<<<1, 1>>>(d_chain, d_result, num_elements, iterations);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_result, d_result, 3 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    
    // Calculate latency
    unsigned long long total_cycles = h_result[0];
    unsigned long long total_accesses = h_result[2];
    double avg_latency_cycles = (double)total_cycles / (double)total_accesses;
    
    printf("\n=== Results ===\n");
    printf("Total cycles: %llu\n", total_cycles);
    printf("Total accesses: %llu\n", total_accesses);
    printf("Average DRAM latency: %.2f cycles\n", avg_latency_cycles);
    
    // Clean up
    CUDA_CHECK(cudaFree(d_chain));
    CUDA_CHECK(cudaFree(d_result));
    free(h_chain);
    free(h_result);
    
    return 0;
}