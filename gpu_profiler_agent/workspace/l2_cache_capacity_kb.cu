#include <cuda_runtime.h>
#include <stdio.h>

// Pointer chasing kernel to measure memory latency hierarchy
__global__ void pointer_chasing_kernel(int* arr, int num_elements, int stride) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Only one thread does the chasing to avoid bandwidth saturation
    if (tid == 0) {
        int current_idx = 0;
        // Chase pointers sequentially
        for (int i = 0; i < num_elements; ++i) {
            current_idx = arr[current_idx];
        }
        
        // Prevent compiler optimization
        if (current_idx == -1) {
            printf("%d", current_idx);
        }
    }
}

int main() {
    int array_size = 41943040;
    int stride = 64;
    int num_elements = array_size / sizeof(int);
    
    int* h_arr = (int*)malloc(array_size);
    if (!h_arr) return -1;
    
    // Initialize array for pointer chasing (creating a cycle)
    for (int i = 0; i < num_elements; ++i) {
        h_arr[i] = (i + stride) % num_elements;
    }
    
    int* d_arr;
    cudaMalloc(&d_arr, array_size);
    cudaMemcpy(d_arr, h_arr, array_size, cudaMemcpyHostToDevice);
    
    pointer_chasing_kernel<<<1, 1>>>(d_arr, num_elements, stride);
    cudaDeviceSynchronize();
    
    cudaFree(d_arr);
    free(h_arr);
    return 0;
}