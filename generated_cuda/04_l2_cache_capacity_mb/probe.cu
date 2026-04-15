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

// Kernel: pointer chasing to measure latency
__global__ void pointer_chasing_kernel(unsigned int* next, unsigned long long* timer, int num_elements, int iterations) {
    unsigned int idx = 0;
    
    // Warm up
    for (int i = 0; i < 10; i++) {
        idx = next[idx];
    }
    
    // Start timing
    unsigned long long start = clock64();
    
    for (int i = 0; i < iterations; i++) {
        idx = next[idx];
    }
    
    unsigned long long end = clock64();
    
    timer[0] = end - start;
    timer[1] = idx; // Prevent optimization
}

// Build a random permutation for pointer chasing
void build_pointer_chain(unsigned int* h_next, int num_elements) {
    // Create a random permutation
    unsigned int* perm = (unsigned int*)malloc(num_elements * sizeof(unsigned int));
    for (int i = 0; i < num_elements; i++) {
        perm[i] = i;
    }
    
    // Fisher-Yates shuffle
    srand(42);
    for (int i = num_elements - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        unsigned int tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }
    
    // Build chain
    for (int i = 0; i < num_elements - 1; i++) {
        h_next[perm[i]] = perm[i + 1];
    }
    h_next[perm[num_elements - 1]] = perm[0]; // Close the loop
    
    free(perm);
}

int main() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("Device: %s\n", prop.name);
    printf("\n");
    
    // Test different working set sizes (in KB)
    int test_sizes_kb[] = {
        64, 128, 192, 256, 320, 384, 448, 512, 
        576, 640, 704, 768, 832, 896, 960, 1024,
        1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536,
        1600, 1664, 1728, 1792, 1856, 1920, 1984, 2048,
        2112, 2176, 2240, 2304, 2368, 2432, 2560, 3072, 4096
    };
    int num_tests = sizeof(test_sizes_kb) / sizeof(test_sizes_kb[0]);
    
    int iterations = 100000; // Number of pointer chasing iterations
    
    printf("Working Set Size (KB), Working Set Size (MB), Avg Latency (cycles)\n");
    printf("====================================================================\n");
    
    double latencies[100]; // Store latencies for analysis
    double sizes_mb[100];
    
    for (int t = 0; t < num_tests; t++) {
        int size_kb = test_sizes_kb[t];
        size_t size_bytes = (size_t)size_kb * 1024;
        int num_elements = size_bytes / sizeof(unsigned int);
        
        // Allocate host memory
        unsigned int* h_next = (unsigned int*)malloc(size_bytes);
        unsigned long long* h_timer = (unsigned long long*)malloc(2 * sizeof(unsigned long long));
        
        // Build pointer chain
        build_pointer_chain(h_next, num_elements);
        
        // Allocate device memory
        unsigned int* d_next;
        unsigned long long* d_timer;
        CUDA_CHECK(cudaMalloc(&d_next, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_timer, 2 * sizeof(unsigned long long)));
        
        // Copy to device
        CUDA_CHECK(cudaMemcpy(d_next, h_next, size_bytes, cudaMemcpyHostToDevice));
        
        // Launch kernel
        pointer_chasing_kernel<<<1, 1>>>(d_next, d_timer, num_elements, iterations);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy result back
        CUDA_CHECK(cudaMemcpy(h_timer, d_timer, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        
        double avg_latency_cycles = (double)h_timer[0] / iterations;
        
        double size_mb = size_kb / 1024.0;
        
        latencies[t] = avg_latency_cycles;
        sizes_mb[t] = size_mb;
        
        printf("%8d KB, %10.3f MB, %12.2f\n", 
               size_kb, size_mb, avg_latency_cycles);
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_next));
        CUDA_CHECK(cudaFree(d_timer));
        free(h_next);
        free(h_timer);
    }
    
    printf("\n====================================================================\n");
    
    // Improved cliff detection using derivative analysis:
    // Find where the rate of latency increase changes significantly
    
    // Skip first point (L1 cache), find stable L2 region
    double l2_stable_sum = 0;
    int l2_stable_count = 0;
    
    // L2 stable region: from index 5 (320KB) to where latency increases
    for (int i = 5; i < num_tests; i++) {
        if (latencies[i] < 210) { // L2 cache latency threshold
            l2_stable_sum += latencies[i];
            l2_stable_count++;
        } else {
            break;
        }
    }
    
    double l2_stable_latency = l2_stable_sum / l2_stable_count;
    printf("L2 stable latency: %.2f cycles (from %d measurements)\n", l2_stable_latency, l2_stable_count);
    
    // Find the cliff using derivative: where latency starts to increase
    double cliff_size_mb = 0;
    double max_derivative = 0;
    int cliff_index = -1;
    
    for (int i = 10; i < num_tests - 1; i++) {
        // Calculate derivative (rate of change)
        double derivative = (latencies[i+1] - latencies[i-1]) / (sizes_mb[i+1] - sizes_mb[i-1]);
        
        // Find where derivative is maximum (steepest increase)
        if (derivative > max_derivative) {
            max_derivative = derivative;
            cliff_index = i;
        }
    }
    
    if (cliff_index > 0 && max_derivative > 5.0) { // Threshold for significant change
        cliff_size_mb = sizes_mb[cliff_index];
        printf("Cliff detected at %.3f MB (derivative: %.2f cycles/MB)\n", 
               cliff_size_mb, max_derivative);
    } else {
        // Alternative method: find where latency exceeds L2 stable by 2%
        double threshold = l2_stable_latency * 1.02;
        for (int i = 5; i < num_tests; i++) {
            if (latencies[i] > threshold) {
                cliff_size_mb = sizes_mb[i-1];
                printf("Cliff detected at %.3f MB (latency: %.2f > threshold %.2f)\n", 
                       sizes_mb[i], latencies[i], threshold);
                break;
            }
        }
    }
    
    if (cliff_size_mb == 0) {
        printf("Warning: No clear cliff detected. Using fallback estimation.\n");
        cliff_size_mb = 2.0; // Conservative estimate
    }
    
    printf("\nFinal Result:\n");
    printf("L2 Cache Capacity: %.3f MB\n", cliff_size_mb);
    
    return 0;
}