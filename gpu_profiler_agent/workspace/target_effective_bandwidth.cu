#include <cuda_runtime.h>

// Measure effective VRAM and Shared Memory Bandwidth
__global__ void effective_bandwidth_kernel(float* in, float* out, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Dynamic shared memory allocation
    extern __shared__ float s_data[];
    
    if (tid < N) {
        // Coalesced global memory read
        float val = in[tid];
        
        // Dense shared memory write
        int s_idx = threadIdx.x;
        s_data[s_idx] = val * 2.0f;
        __syncthreads();
        
        // Dense shared memory read and Coalesced global memory write
        out[tid] = s_data[s_idx];
    }
}

int main() {
    // Very large array to saturate VRAM bandwidth
    int N = 20971520;
    size_t size = N * sizeof(float);
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = threadsPerBlock * sizeof(float);
    
    effective_bandwidth_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}