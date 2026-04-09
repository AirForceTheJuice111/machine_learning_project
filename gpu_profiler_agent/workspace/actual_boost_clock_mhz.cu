#include <cuda_runtime.h>
#include <stdio.h>

// A kernel designed to be extremely compute-bound (FP32 or Tensor Core bound)
__global__ void boost_clock_kernel(int iterations) {
    float a = 1.0f;
    float b = 1.0001f;
    float c = 0.0f;
    
    // Very dense FP32 loop to keep SMs busy and trigger boost clocks
    for (int i = 0; i < iterations; ++i) {
        c = fmaf(a, b, c);
        c = fmaf(a, b, c);
        c = fmaf(a, b, c);
        c = fmaf(a, b, c);
        c = fmaf(a, b, c);
        c = fmaf(a, b, c);
        c = fmaf(a, b, c);
        c = fmaf(a, b, c);
    }
    
    // Prevent compiler from optimizing away the loop
    if (c == -1.0f) {
        printf("%f", c);
    }
}

int main() {
    int iterations = 1000000;
    
    // Launch enough blocks and threads to fill the GPU
    int threadsPerBlock = 256;
    int numBlocks = 1024; // Typically enough to occupy all SMs
    
    boost_clock_kernel<<<numBlocks, threadsPerBlock>>>(iterations);
    cudaDeviceSynchronize();
    
    return 0;
}